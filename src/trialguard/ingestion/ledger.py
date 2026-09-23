"""The refresh run ledger (refresh_runs) and the lease that serialises runs.

Every run leaves a row, including the ones that abort, fail or find another run
already going. A health check that can only see the last success cannot tell
"aborted twice on a schema change" from "the scheduler stopped firing".

Mutual exclusion is a lease row rather than a session advisory lock: Neon's
pooled endpoint is PgBouncer in transaction mode, where session locks are not
held. The check-and-insert is serialised by a transaction-scoped advisory lock,
which is safe there. A crashed run's lease stops blocking once its heartbeat is
older than LEASE_MINUTES.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from trialguard.db.schema import get_conn

LEASE_MINUTES = 30
FAILURE_OUTCOMES = ("failed", "aborted_gate")
SUCCESS_OUTCOMES = ("published", "noop", "skipped_unchanged")

_COLS = (
    "run_id", "source", "started_at", "heartbeat_at", "finished_at", "outcome",
    "reason", "ctgov_data_timestamp", "ctgov_total_count", "fetched", "counts",
    "gates", "embed_tag", "parser_version", "git_sha", "duration_s",
)

_WRITABLE = frozenset(
    {"reason", "ctgov_data_timestamp", "ctgov_total_count", "fetched", "counts", "gates"}
)


def git_sha() -> str | None:
    """The code that wrote the corpus. Fly sets FLY_IMAGE_REF on every machine."""
    return os.environ.get("TG_GIT_SHA") or os.environ.get("FLY_IMAGE_REF")


def _lock(cur, source: str) -> None:
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"trialguard.refresh.{source}",))


def start(source: str, embed_tag: str, parser_version: str) -> tuple[str, bool]:
    """Open a run. Returns (run_id, acquired); not acquired means another is live."""
    run_id = uuid.uuid4().hex
    with get_conn() as conn, conn.cursor() as cur:
        _lock(cur, source)
        cur.execute(
            "UPDATE refresh_runs SET outcome = 'failed', reason = 'abandoned', "
            "finished_at = NOW() WHERE source = %s AND outcome = 'running' "
            "AND heartbeat_at < NOW() - make_interval(mins => %s)",
            (source, LEASE_MINUTES),
        )
        cur.execute(
            "SELECT run_id FROM refresh_runs WHERE source = %s AND outcome = 'running'",
            (source,),
        )
        holder = cur.fetchone()
        outcome = "running" if holder is None else "skipped_locked"
        cur.execute(
            "INSERT INTO refresh_runs (run_id, source, outcome, reason, embed_tag, "
            "parser_version, git_sha, finished_at) VALUES (%s, %s, %s, %s, %s, %s, %s, "
            "CASE WHEN %s = 'running' THEN NULL ELSE NOW() END)",
            (
                run_id, source, outcome,
                None if holder is None else f"run {holder[0]} holds the lease",
                embed_tag, parser_version, git_sha(), outcome,
            ),
        )
    return run_id, holder is None


def heartbeat(run_id: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE refresh_runs SET heartbeat_at = NOW() WHERE run_id = %s", (run_id,))


def finish(run_id: str, outcome: str, cur=None, **fields: Any) -> None:
    """Close a run. Pass `cur` to write inside the caller's (publish) transaction."""
    unknown = set(fields) - _WRITABLE
    if unknown:
        raise ValueError(f"not refresh_runs columns: {sorted(unknown)}")
    fields = {k: (json.dumps(v) if k in ("counts", "gates") and v is not None else v)
              for k, v in fields.items()}
    sets = "".join(f", {k} = %({k})s" for k in fields)
    sql = (
        "UPDATE refresh_runs SET outcome = %(outcome)s, finished_at = NOW(), "  # noqa: S608 -- keys checked against _WRITABLE
        "heartbeat_at = NOW(), duration_s = EXTRACT(EPOCH FROM NOW() - started_at)"
        f"{sets} WHERE run_id = %(run_id)s"
    )
    params = {"outcome": outcome, "run_id": run_id, **fields}
    if cur is not None:
        cur.execute(sql, params)
        return
    with get_conn() as conn, conn.cursor() as c:
        c.execute(sql, params)


def _row(r) -> dict:
    d = dict(zip(_COLS, r, strict=True))
    for k in ("started_at", "heartbeat_at", "finished_at"):
        if d[k] is not None:
            d[k] = d[k].isoformat(timespec="seconds")
    return d


def last(source: str, outcomes: tuple[str, ...] | None = None) -> dict | None:
    sql = f"SELECT {', '.join(_COLS)} FROM refresh_runs WHERE source = %s"  # noqa: S608 -- constant columns
    params: list = [source]
    if outcomes:
        sql += " AND outcome = ANY(%s)"
        params.append(list(outcomes))
    sql += " ORDER BY started_at DESC LIMIT 1"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        r = cur.fetchone()
    return _row(r) if r else None


def health(source: str) -> dict | None:
    """Last attempt and failure streak for /api/health, from one indexed read."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT outcome, reason, started_at, finished_at FROM refresh_runs "
            "WHERE source = %s AND outcome <> 'skipped_locked' "
            "ORDER BY started_at DESC LIMIT 20",
            (source,),
        )
        rows = cur.fetchall()
    if not rows:
        return None
    streak = 0
    for outcome, *_ in rows:
        if outcome == "running":
            continue
        if outcome not in FAILURE_OUTCOMES:
            break
        streak += 1
    outcome, reason, started, finished = rows[0]
    return {
        "last_attempt": {
            "outcome": outcome,
            "reason": reason,
            "started_at": started.isoformat(timespec="seconds") if started else None,
            "finished_at": finished.isoformat(timespec="seconds") if finished else None,
        },
        "consecutive_failures": streak,
    }


def consecutive_failures(source: str) -> int:
    """Failed or gate-aborted runs since the last success (locked skips ignored)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT outcome FROM refresh_runs WHERE source = %s AND outcome <> ALL(%s) "
            "ORDER BY started_at DESC LIMIT 50",
            (source, ["running", "skipped_locked"]),
        )
        n = 0
        for (outcome,) in cur.fetchall():
            if outcome not in FAILURE_OUTCOMES:
                break
            n += 1
    return n
