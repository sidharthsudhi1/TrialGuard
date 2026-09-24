"""Refresh the ctgov_live corpus against CT.gov: write, audit, publish.

  python -m trialguard.scripts.refresh                 # the scheduled run
  python -m trialguard.scripts.refresh --dry-run       # plan + gates, no writes
  python -m trialguard.scripts.refresh --allow reembed_churn   # one intended big change
  python -m trialguard.scripts.refresh --force         # ignore the /version short-circuit

A run, in order:

  1. lease   one run at a time (ingestion/ledger.py); a second exits 75.
  2. check   CT.gov's /version dataTimestamp. If CT.gov has not published since
             the last successful run, and the parser and embedding config have
             not changed, there is nothing to reconcile: one request, exit 0.
             This is what makes an hourly schedule nearly free.
  3. pull    a crawl proven complete (ctgov.pull_trials), or the run fails.
  4. plan    the content-hash diff and confirmed expiry (ingestion/publish.py).
  5. audit   gates (ingestion/audit.py). Hard gates always block; calibrated
             ones block only under TG_REFRESH_GATES=enforce. --dry-run stops here.
  6. embed   new and changed trials only, before the database is touched.
  7. publish one transaction: rows, soft expiry, ledger row, corpus version.

Every run leaves a refresh_runs row. Exit codes: 0 published / noop / skipped,
1 failed, 2 aborted by a gate, 75 another run holds the lease.

TG_REFRESH_V2=0 runs the previous date-diff refresh (scripts/refresh_v1.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import cast

from trialguard.db.cache import cache_get, cache_put
from trialguard.ingestion import audit as audit_mod
from trialguard.ingestion import ledger
from trialguard.ingestion import publish as pub
from trialguard.ingestion.ctgov import RECRUITING_STATUSES, fetch_version, lookup_ids, pull_trials
from trialguard.ingestion.embed import embed_tag
from trialguard.ingestion.normalise import normalise_trial
from trialguard.ingestion.provenance import parser_version

log = logging.getLogger("trialguard.refresh")

SOURCE = "ctgov_live"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ABORTED = 2
EXIT_LOCKED = 75

# Bootstrap commits in chunks: a first load of ~26k trials is ~90 min of CPU to
# embed, and holding all of it for one transaction at the end risks the lot on
# one dropped connection. Each chunk is an idempotent upsert, so a crash
# resumes as an ordinary diff.
BOOTSTRAP_CHUNK = 2000


def _touch_last_refresh() -> None:
    """A skipped run still reconciled the corpus: CT.gov has nothing newer.

    Without this the stamp would age through every weekend (CT.gov publishes
    Mon-Fri) and the probe would alert on a corpus that is exactly current.
    """
    prev = cache_get("corpus", "last_refresh") or {}
    cache_put("corpus", "last_refresh", {**prev, "at": pub.utcnow()})


def _unchanged_since_last_success(tag: str, pver: str) -> tuple[bool, str | None]:
    version = fetch_version().get("dataTimestamp")
    prev = ledger.last(SOURCE, ledger.SUCCESS_OUTCOMES)
    unchanged = bool(
        prev
        and version
        and prev["ctgov_data_timestamp"] == version
        and prev["embed_tag"] == tag
        and prev["parser_version"] == pver
    )
    return unchanged, version


def _mean_criteria(fresh: dict[str, dict]) -> float:
    if not fresh:
        return 0.0
    total = sum(
        len(t.get("inclusion_criteria") or []) + len(t.get("exclusion_criteria") or [])
        for t in fresh.values()
    )
    return total / len(fresh)


def run(
    *,
    dry_run: bool = False,
    allow: frozenset[str] = frozenset(),
    force: bool = False,
    bootstrap: bool | None = None,
    statuses: list[str] | None = None,
) -> tuple[int, dict]:
    """One refresh. Returns (exit code, report)."""
    statuses = statuses or list(RECRUITING_STATUSES)
    tag, pver = embed_tag(), parser_version()

    run_id: str | None = None
    if not dry_run:
        run_id, acquired = ledger.start(SOURCE, tag, pver)
        if not acquired:
            log.warning("another refresh holds the lease; exiting")
            return EXIT_LOCKED, {"outcome": "skipped_locked", "run_id": run_id}

    try:
        if not force and not dry_run:
            unchanged, version = _unchanged_since_last_success(tag, pver)
            if unchanged and run_id:
                ledger.finish(run_id, "skipped_unchanged", ctgov_data_timestamp=version)
                _touch_last_refresh()
                log.info("phase=check outcome=skipped_unchanged data=%s", version)
                return EXIT_OK, {"outcome": "skipped_unchanged", "ctgov_data_timestamp": version}

        pull = pull_trials(statuses=statuses)
        log.info(
            "phase=pull fetched=%d total=%d pages=%d retries=%d data=%s",
            len(pull.trials), pull.total_count, pull.pages, pull.retries, pull.data_timestamp,
        )
        if run_id:
            ledger.heartbeat(run_id)

        fresh = {n: normalise_trial(t) for n, t in pull.trials.items()}
        existing = pub.snapshot(SOURCE)
        active = sum(1 for e in existing.values() if not e.expired)
        is_bootstrap = (active == 0) if bootstrap is None else bootstrap

        p = pub.plan(fresh, existing, tag, pver)
        conf = pub.confirm(p.missing, existing, statuses, lookup_ids)

        previous = ledger.last(SOURCE, ("published", "noop"))
        prev_stats = ((previous or {}).get("gates") or {}).get("stats")
        a = audit_mod.AuditInput(
            fetched=len(fresh),
            total_count=pull.total_count,
            active=active,
            to_embed=len(p.to_embed),
            to_expire=len(conf.expire),
            field_missing=pull.field_missing,
            statuses=[t.get("status", "") for t in fresh.values()],
            requested_statuses=statuses,
            ids=list(fresh),
            mean_criteria=_mean_criteria(fresh),
            current_tag=tag,
            corpus_tag=pub.corpus_tag(existing),
            bootstrap=is_bootstrap,
            previous=prev_stats,
        )
        gates = audit_mod.audit(a, allow)
        gates_json = audit_mod.to_json(gates, a)
        summary = {
            "new": len(p.new),
            "changed": len(p.changed),
            "meta_only": len(p.meta_only),
            "resurrected": len(p.resurrected),
            "seen": len(p.seen),
            "missing": len(p.missing),
            "expired": len(conf.expire),
            "kept_missing": len(conf.keep),
            "lookup_failed": conf.lookup_failed,
            "embedded": len(p.to_embed),
            "corpus": len(fresh),
            "bootstrap": is_bootstrap,
            "ctgov_data_timestamp": pull.data_timestamp,
        }
        blocked = audit_mod.blocking(gates)
        report = {
            "outcome": None,
            "run_id": run_id,
            "summary": summary,
            "expired": conf.expire,
            "gates_failed": [g.name for g in audit_mod.failed(gates)],
            "gates_blocking": [g.name for g in blocked],
            "gates_mode": gates_json["mode"],
        }
        for g in audit_mod.failed(gates):
            log.warning("gate %s failed: %s (want %s)%s", g.name, g.value, g.threshold,
                        "" if g.blocks else " [not blocking]")

        if dry_run:
            report["outcome"] = "dry_run"
            report["gates"] = gates_json
            return (EXIT_ABORTED if blocked else EXIT_OK), report

        # Past the dry-run return, a run always holds its lease row.
        run_id = cast(str, run_id)
        if blocked:
            reason = "; ".join(f"{g.name} {g.value} (want {g.threshold})" for g in blocked)
            ledger.finish(
                run_id, "aborted_gate", reason=reason[:1000], counts=summary, gates=gates_json,
                ctgov_data_timestamp=pull.data_timestamp, ctgov_total_count=pull.total_count,
                fetched=len(fresh),
            )
            report["outcome"] = "aborted_gate"
            return EXIT_ABORTED, report

        def beat() -> None:
            ledger.heartbeat(run_id)

        to_embed = [fresh[n] for n in p.to_embed]
        staged = False
        if is_bootstrap and len(to_embed) > BOOTSTRAP_CHUNK:
            from trialguard.db.schema import get_conn

            for i in range(0, len(to_embed), BOOTSTRAP_CHUNK):
                part = to_embed[i : i + BOOTSTRAP_CHUNK]
                pub.embed(part, on_chunk=beat)
                with get_conn() as conn, conn.cursor() as cur:
                    pub.stage(cur, part, SOURCE)
                for t in part:
                    t.pop("embedding", None)
                log.info("phase=bootstrap staged=%d/%d", i + len(part), len(to_embed))
            staged = True
        else:
            pub.embed(to_embed, on_chunk=beat)
        log.info("phase=embed embedded=%d", len(to_embed))

        pub.publish(
            run_id=run_id, source=SOURCE, p=p, fresh=fresh, confirmation=conf,
            summary=summary, gates=gates_json, data_timestamp=pull.data_timestamp,
            total_count=pull.total_count, staged=staged,
        )
        if is_bootstrap:
            from trialguard.db.schema import ensure_vector_index

            ensure_vector_index()
        report["outcome"] = "published" if pub._changes(p, conf) else "noop"
        log.info("phase=publish outcome=%s %s", report["outcome"], json.dumps(summary))
        return EXIT_OK, report
    except Exception as e:
        if run_id:
            try:
                ledger.finish(run_id, "failed", reason=f"{type(e).__name__}: {e}"[:1000])
            except Exception:
                log.exception("could not record the failure in refresh_runs")
        raise


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Refresh the ctgov_live corpus.")
    ap.add_argument("--dry-run", action="store_true", help="plan and audit only; no writes")
    ap.add_argument("--allow", action="append", default=[], metavar="GATE",
                    help="override one named gate for this run (repeatable)")
    ap.add_argument("--force", action="store_true",
                    help="run even if CT.gov has not published since the last success")
    ap.add_argument("--bootstrap", action="store_true",
                    help="skip the churn gates and commit in chunks (first load)")
    args = ap.parse_args(argv)

    if os.environ.get("TG_REFRESH_V2", "1") == "0":
        from trialguard.scripts.refresh_v1 import refresh_v1

        try:
            return EXIT_OK if refresh_v1() is not None else EXIT_ABORTED
        except Exception:
            log.exception("refresh v1 failed")
            return EXIT_FAILED

    try:
        code, report = run(
            dry_run=args.dry_run,
            allow=frozenset(args.allow),
            force=args.force,
            bootstrap=True if args.bootstrap else None,
        )
    except Exception:
        log.exception("refresh failed")
        return EXIT_FAILED
    print(json.dumps(report, indent=2, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
