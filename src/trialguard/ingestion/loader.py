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


# Metadata-only update: a trial whose embedded text is unchanged but whose other
# CT.gov fields (status, ages, healthy_volunteers, ...) or parser version moved.
# Rewrites every stored column except the embedding, which is what makes it
# free: no model call and no 3 KB vector per row on the wire.
_META_COLS = [c for c in COLUMNS if c not in ("nct_id", "embedding", "source")]
_META_CASTS = {
    "conditions": "::text[]", "interventions": "::text[]",
    "inclusion_criteria": "::text[]", "exclusion_criteria": "::text[]",
    "healthy_volunteers": "::boolean", "metadata": "::jsonb",
}
META_TEMPLATE = (
    "(%(nct_id)s, %(source)s, "
    + ", ".join(f"%({c})s{_META_CASTS.get(c, '')}" for c in _META_COLS)
    + ")"
)
META_UPDATE_SQL = f"""
UPDATE trials AS t SET
    {", ".join(f"{c} = v.{c}" for c in _META_COLS)},
    last_seen_at   = NOW(),
    expired_at     = NULL,
    expired_reason = NULL,
    missing_runs   = 0
FROM (VALUES %s) AS v(nct_id, source, {", ".join(_META_COLS)})
WHERE t.nct_id = v.nct_id AND t.source = v.source
"""  # noqa: S608 -- column names are module constants


def row_for(t: dict, source: str, with_embedding: bool = True) -> dict:
    """One trial dict as the column values the trials table stores."""
    from trialguard.ingestion.provenance import stamp

    row = {
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
        "metadata": json.dumps({k: v for k, v in t.items() if k != "embedding"}),
        "source": t.get("source", source),
        **stamp(t),
    }
    if with_embedding:
        row["embedding"] = t["embedding"]
    return row


def upsert_rows(cur, trials: list[dict], source: str = "ctgov_live") -> int:
    """Upsert inside the caller's transaction. Raises SourceCollision."""
    # One statement per page, and Postgres refuses to update a row twice in one
    # statement. A pagination-drift duplicate in a streaming ingest would
    # otherwise fail the batch; the later copy wins, as it did row by row.
    trials = list({t["nct_id"]: t for t in trials}.values())
    rows = [row_for(t, source) for t in trials]
    written = psycopg2.extras.execute_values(
        cur, UPSERT_SQL, rows, template=TEMPLATE, page_size=100, fetch=True
    )
    refused = {r["nct_id"] for r in rows} - {w[0] for w in written}
    if refused:
        raise SourceCollision(
            f"{len(refused)} ids belong to another source: {sorted(refused)[:10]}"
        )
    return len(rows)


def update_metadata(cur, trials: list[dict], source: str = "ctgov_live") -> int:
    """Rewrite everything but the embedding, inside the caller's transaction."""
    if not trials:
        return 0
    rows = [row_for(t, source, with_embedding=False) for t in trials]
    psycopg2.extras.execute_values(
        cur, META_UPDATE_SQL, rows, template=META_TEMPLATE, page_size=500
    )
    return len(rows)


def upsert_trials(trials: list[dict], source: str = "ctgov_live") -> int:
    """Upsert a batch of enriched trial dicts in its own transaction."""
    # Retry transient Neon drops (OperationalError/InterfaceError) with a fresh
    # connection; the upsert is idempotent (ON CONFLICT), so a retry is safe.
    # Raising inside get_conn rolls the whole batch back.
    for attempt in range(3):
        try:
            with get_conn() as conn, conn.cursor() as cur:
                n = upsert_rows(cur, trials, source)
                conn.commit()
            return n
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")
