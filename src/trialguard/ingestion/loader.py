"""Upsert normalised + embedded trials into pgvector."""

from __future__ import annotations

import json
import time

import psycopg2
import psycopg2.extras

from trialguard.db.schema import get_conn

COLUMNS = (
    "nct_id", "title", "status", "phase", "conditions", "interventions",
    "eligibility_raw", "inclusion_criteria", "exclusion_criteria",
    "min_age", "max_age", "sex", "healthy_volunteers", "last_updated",
    "embedding", "metadata", "source",
    "doc_hash", "content_hash", "embed_tag", "parser_version",
)

_CASTS = {"embedding": "::vector", "metadata": "::jsonb"}

# Named placeholders for execute_values, one per column plus the seen stamps.
TEMPLATE = (
    "("
    + ", ".join(f"%({c})s{_CASTS.get(c, '')}" for c in COLUMNS)
    + ", NOW(), NOW())"
)

_UPDATED = [c for c in COLUMNS if c != "nct_id"]

# The WHERE on the conflict branch is the source guard. The primary key is
# nct_id alone, so without it an eval-corpus load (sigir, trec_*) that shares an
# NCT id with ctgov_live silently re-labels the production row, and the next
# refresh reads it as missing. A guarded row is not updated and so not
# RETURNed, which is how upsert_trials detects the collision.
UPSERT_SQL = f"""
INSERT INTO trials ({", ".join(COLUMNS)}, first_seen_at, last_seen_at)
VALUES %s
ON CONFLICT (nct_id) DO UPDATE SET
    {", ".join(f"{c} = EXCLUDED.{c}" for c in _UPDATED)},
    last_seen_at   = NOW(),
    expired_at     = NULL,
    expired_reason = NULL,
    missing_runs   = 0,
    ingested_at    = NOW()
WHERE trials.source = EXCLUDED.source
RETURNING nct_id
"""  # noqa: S608 -- column names are module constants


class SourceCollision(RuntimeError):
    """An upsert would have overwritten a row that belongs to another source."""


def upsert_trials(trials: list[dict], source: str = "ctgov_live") -> int:
    """Upsert a batch of enriched trial dicts. Returns count inserted/updated."""
    from trialguard.ingestion.provenance import stamp

    # One statement per page now, and Postgres refuses to update a row twice in
    # one statement. A pagination-drift duplicate in a streaming ingest would
    # otherwise fail the batch; the later copy wins, as it did row by row.
    trials = list({t["nct_id"]: t for t in trials}.values())
    rows = []
    for t in trials:
        rows.append({
            "nct_id": t["nct_id"],
            "title": t.get("title"),
            "status": t.get("status"),
            "phase": t.get("phase"),
            "conditions": t.get("conditions", []),
            "interventions": t.get("interventions", []),
            "eligibility_raw": t.get("eligibility_raw"),
            "inclusion_criteria": t.get("inclusion_criteria", []),
            "exclusion_criteria": t.get("exclusion_criteria", []),
            "min_age": t.get("min_age"),
            "max_age": t.get("max_age"),
            "sex": t.get("sex"),
            "healthy_volunteers": t.get("healthy_volunteers"),
            "last_updated": t.get("last_updated"),
            "embedding": t["embedding"],
            "metadata": json.dumps({
                k: v for k, v in t.items()
                if k not in ("embedding",)
            }),
            "source": t.get("source", source),
            **stamp(t),
        })

    # Retry transient Neon drops (OperationalError/InterfaceError) with a fresh
    # connection; the upsert is idempotent (ON CONFLICT), so a retry is safe.
    for attempt in range(3):
        try:
            with get_conn() as conn, conn.cursor() as cur:
                written = psycopg2.extras.execute_values(
                    cur, UPSERT_SQL, rows, template=TEMPLATE, page_size=100, fetch=True
                )
                refused = {r["nct_id"] for r in rows} - {w[0] for w in written}
                if refused:
                    # Raising inside get_conn rolls the whole batch back.
                    raise SourceCollision(
                        f"{len(refused)} ids belong to another source: {sorted(refused)[:10]}"
                    )
                conn.commit()
            return len(rows)
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    return len(rows)
