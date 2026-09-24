"""Refresh v1: the date-diff refresh, kept for rollback (TG_REFRESH_V2=0).

Superseded by scripts/refresh.py (content-hash diff, audit gates, confirmed
soft expiry, one-transaction publish). v1 still hard-deletes trials that leave
the pull and diffs on lastUpdatePostDate, which misses same-day revisions and
parser changes. Delete once v2 has run cleanly in production.

Enrolment status is a moving target, so a loaded corpus goes stale. This re-pulls
the current oncology set from CT.gov and diffs it against the DB:

  - new       -> embed once + insert
  - expired   -> trials that left the enrolling set are deleted (no longer
                 matchable; they re-enter on a future refresh if they return)
  - revised   -> CT.gov's lastUpdatePostDate moved, so the record changed. The
                 eligibility text may have changed with it, and that text is
                 what the analyst grounds quotes against and what the embedding
                 encodes, so these are re-embedded and fully upserted.
  - restatused-> status differs with the same lastUpdatePostDate. Updated in
                 place, no re-embed.

Comparing status alone was the earlier behaviour and it left revised criteria in
the corpus indefinitely: a trial could be served with eligibility text CT.gov no
longer publishes, and a verdict grounded in a quote from it. Nothing surfaced
that, because status had not moved.

Idempotent by construction: a corpus already matching CT.gov produces no writes
and, in particular, zero embedding calls.

  python -m trialguard.scripts.refresh

Safety: the pull is proven complete before anything is diffed (pull_trials
raises IncompletePull otherwise), and if it is still a small fraction of the
existing corpus the whole refresh is aborted rather than deleting most of it.

Exit codes: 0 done, 1 failed (including an incomplete pull), 2 aborted by a
guard. A scheduled machine that exits 0 on an abort reports success while the
corpus goes stale.
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg2.extras
from rich.console import Console

from trialguard.db.cache import cache_put
from trialguard.db.schema import get_conn
from trialguard.ingestion.ctgov import pull_trials
from trialguard.ingestion.embed import eligibility_text_for_embedding, embed_batch
from trialguard.ingestion.loader import upsert_trials
from trialguard.ingestion.normalise import normalise_trial
from trialguard.ingestion.provenance import content_hash

console = Console()
SOURCE = "ctgov_live"


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def plan_refresh(
    fresh: dict[str, dict], existing: dict[str, tuple[str | None, str | None]]
) -> tuple[set[str], set[str], list[str], list[str]]:
    """Split the corpus against a fresh pull. Pure, so it can be tested directly.

    Returns (expired, new, revised, restatused). `revised` is the set that has to
    be re-embedded: CT.gov's lastUpdatePostDate is its own answer to "did this
    record change", so it is the only available signal that eligibility text may
    have moved. Status is checked second and only as a fallback, since a real
    status change moves the date too -- that branch catches an ingest that stored
    one without the other, and keeps it off the expensive path.
    """
    fresh_ids, existing_ids = set(fresh), set(existing)
    expired = existing_ids - fresh_ids
    new_ids = fresh_ids - existing_ids

    revised: list[str] = []
    restatused: list[str] = []
    for nct in sorted(fresh_ids & existing_ids):
        old_status, old_updated = existing[nct]
        if (fresh[nct].get("last_updated") or "") != (old_updated or ""):
            revised.append(nct)
        elif fresh[nct].get("status") != old_status:
            restatused.append(nct)
    return expired, new_ids, revised, restatused


def refresh_v1(max_trials: int | None = None) -> dict | None:
    console.print("Pulling current recruiting oncology set from CT.gov...")
    pull = pull_trials(max_trials=max_trials)
    fresh = {n: normalise_trial(t) for n, t in pull.trials.items()}
    fresh_ids = set(fresh)
    console.print(
        f"  {len(fresh_ids)} recruiting trials live "
        f"({pull.pages} pages, {pull.retries} retries, data {pull.data_timestamp})."
    )

    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT nct_id, status, last_updated FROM trials WHERE source=%s", (SOURCE,)
        )
        existing = {n: (st, lu) for n, st, lu in cur.fetchall()}
    existing_ids = set(existing)

    expired, new_ids, revised, restatused = plan_refresh(fresh, existing)

    # Guard: a fresh pull far smaller than the corpus means a partial/failed pull;
    # do not let that mass-expire the corpus.
    if existing_ids and len(fresh_ids) < 0.5 * len(existing_ids):
        console.print(
            f"[red]Aborting: fresh pull ({len(fresh_ids)}) is under half the corpus "
            f"({len(existing_ids)}) — likely a partial pull.[/red]"
        )
        return None

    if expired:
        with get_conn() as c, c.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "DELETE FROM trials WHERE nct_id=%s AND source=%s",
                [(n, SOURCE) for n in expired],
            )
            c.commit()

    # New and revised trials both need an embedding of text the corpus does not
    # have, so they take the same path. Nothing else does: an unchanged corpus
    # leaves this list empty and the model is never called.
    to_embed = [fresh[n] for n in sorted(new_ids) + sorted(revised)]
    for i in range(0, len(to_embed), 200):
        batch = to_embed[i : i + 200]
        embs = embed_batch([eligibility_text_for_embedding(t) for t in batch])
        for t, e in zip(batch, embs, strict=True):
            t["embedding"] = e
        upsert_trials(batch, source=SOURCE)

    if restatused:
        with get_conn() as c, c.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "UPDATE trials SET status=%s, last_updated=%s, content_hash=%s, "
                "metadata = jsonb_set(coalesce(metadata, '{}'::jsonb), '{status}', to_jsonb(%s::text)) "
                "WHERE nct_id=%s AND source=%s",
                [
                    (
                        fresh[n]["status"],
                        fresh[n].get("last_updated"),
                        content_hash(fresh[n]),
                        fresh[n]["status"],
                        n,
                        SOURCE,
                    )
                    for n in restatused
                ],
            )
            c.commit()

    summary = {
        "new": len(new_ids),
        "expired": len(expired),
        "revised": len(revised),
        "restatused": len(restatused),
        "corpus": len(fresh_ids),
        "embedded": len(to_embed),
        "ctgov_data_timestamp": pull.data_timestamp,
    }
    # Recorded so /api/health can report when the corpus was last reconciled.
    # Staleness that nothing can see is the state this work stream exists to end.
    cache_put("corpus", "last_refresh", {**summary, "at": _utcnow()})

    console.print(
        f"[green]Refresh done:[/green] {summary['new']} new, {summary['expired']} "
        f"expired, {summary['revised']} revised, {summary['restatused']} restatused."
    )
    return summary
