"""Phase 7 WS-5: refresh the ctgov_live corpus against the current recruiting set.

Recruiting status is a moving target, so a loaded corpus goes stale. This re-pulls
the current recruiting oncology set from CT.gov and diffs it against the DB:

  - new trials  -> embed once + insert
  - expired     -> trials that left the recruiting set are deleted (no longer
                   matchable in a recruiting-only corpus; they re-enter on a future
                   refresh if they return)
  - status churn-> status/last_updated updated in place, no re-embed

Idempotent. Manual invocation; scheduling is deferred.

  python -m trialguard.scripts.refresh

Safety: if the fresh pull is a small fraction of the existing corpus (a partial or
failed CT.gov pull), expiry is aborted rather than deleting most of the corpus.
"""

from __future__ import annotations

import psycopg2.extras
from rich.console import Console

from trialguard.db.schema import get_conn
from trialguard.ingestion.ctgov import fetch_oncology_trials
from trialguard.ingestion.embed import eligibility_text_for_embedding, embed_batch
from trialguard.ingestion.loader import upsert_trials
from trialguard.ingestion.normalise import normalise_trial

console = Console()
SOURCE = "ctgov_live"


def refresh(max_trials: int = 40000) -> None:
    console.print("Pulling current recruiting oncology set from CT.gov...")
    fresh = {t["nct_id"]: normalise_trial(t) for t in fetch_oncology_trials(max_trials=max_trials)}
    fresh_ids = set(fresh)
    console.print(f"  {len(fresh_ids)} recruiting trials live.")

    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT nct_id, status FROM trials WHERE source=%s", (SOURCE,))
        existing = dict(cur.fetchall())
    existing_ids = set(existing)

    expired = existing_ids - fresh_ids
    new_ids = fresh_ids - existing_ids
    changed = [
        (fresh[n]["status"], fresh[n].get("last_updated"), n)
        for n in fresh_ids & existing_ids
        if fresh[n]["status"] != existing[n]
    ]

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

    new = [fresh[n] for n in new_ids]
    for i in range(0, len(new), 200):
        batch = new[i : i + 200]
        embs = embed_batch([eligibility_text_for_embedding(t) for t in batch])
        for t, e in zip(batch, embs):
            t["embedding"] = e
        upsert_trials(batch, source=SOURCE)

    if changed:
        with get_conn() as c, c.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "UPDATE trials SET status=%s, last_updated=%s WHERE nct_id=%s AND source=%s",
                [(s, lu, n, SOURCE) for s, lu, n in changed],
            )
            c.commit()

    console.print(
        f"[green]Refresh done:[/green] {len(new_ids)} new, {len(expired)} expired, "
        f"{len(changed)} status-updated."
    )


if __name__ == "__main__":
    refresh()
