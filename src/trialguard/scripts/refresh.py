"""Refresh the ctgov_live corpus against the current enrolling set.

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

Safety: if the fresh pull is a small fraction of the existing corpus (a partial or
failed CT.gov pull), the whole refresh is aborted rather than deleting most of the
corpus.
"""

from __future__ import annotations

from datetime import datetime, timezone

import psycopg2.extras
from rich.console import Console

from trialguard.db.cache import cache_put
from trialguard.db.schema import get_conn
from trialguard.ingestion.ctgov import fetch_oncology_trials
from trialguard.ingestion.embed import eligibility_text_for_embedding, embed_batch
from trialguard.ingestion.loader import upsert_trials
from trialguard.ingestion.normalise import normalise_trial

console = Console()
SOURCE = "ctgov_live"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def refresh(max_trials: int = 40000) -> dict | None:
    console.print("Pulling current recruiting oncology set from CT.gov...")
    fresh = {t["nct_id"]: normalise_trial(t) for t in fetch_oncology_trials(max_trials=max_trials)}
    fresh_ids = set(fresh)
    console.print(f"  {len(fresh_ids)} recruiting trials live.")

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
        return

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
        for t, e in zip(batch, embs):
            t["embedding"] = e
        upsert_trials(batch, source=SOURCE)

    if restatused:
        with get_conn() as c, c.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "UPDATE trials SET status=%s, last_updated=%s "
                "WHERE nct_id=%s AND source=%s",
                [
                    (fresh[n]["status"], fresh[n].get("last_updated"), n, SOURCE)
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
    }
    # Recorded so /api/health can report when the corpus was last reconciled.
    # Staleness that nothing can see is the state this work stream exists to end.
    cache_put("corpus", "last_refresh", {**summary, "at": _utcnow()})

    console.print(
        f"[green]Refresh done:[/green] {summary['new']} new, {summary['expired']} "
        f"expired, {summary['revised']} revised, {summary['restatused']} restatused."
    )
    return summary


if __name__ == "__main__":
    refresh()
