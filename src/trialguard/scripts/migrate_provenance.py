"""Pipeline 1 hardening, WS-1: add row provenance to an existing corpus.

  python -m trialguard.scripts.migrate_provenance --dry-run
  python -m trialguard.scripts.migrate_provenance

Applies PROVENANCE_DDL (additive, idempotent), then backfills every row that has
no doc_hash yet:

- doc_hash / content_hash from the stored columns. No re-embed: the hash is of
  the text built from the criteria the row already holds.
- embed_tag is verified, not assumed. A sample of ctgov_live rows is re-embedded
  under the current config and compared with the stored vectors; only if every
  one matches (cosine >= --threshold) is the corpus stamped with the current
  tag. Otherwise it is stamped 'unknown', and the refresh re-embeds it.
- parser_version = 'unknown'. Which parser built each row was never recorded,
  so the first refresh re-parses everything and re-embeds only what moved.
- first_seen_at / last_seen_at = ingested_at.

Run it before deploying the code that writes these columns: the upsert names
them, so a refresh against an unmigrated table fails (loudly) until it is.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import psycopg2.extras

from trialguard.db.schema import PROVENANCE_DDL, get_conn
from trialguard.ingestion.embed import eligibility_text_for_embedding, embed_matrix, embed_tag
from trialguard.ingestion.provenance import CONTENT_FIELDS, content_hash, doc_hash

SOURCE = "ctgov_live"
BATCH = 2000
UNKNOWN = "unknown"

# parser_version is never known for a pre-existing row, so it is a literal.
BACKFILL_SQL = f"""
UPDATE trials AS t SET
    doc_hash       = v.doc_hash,
    content_hash   = v.content_hash,
    embed_tag      = coalesce(t.embed_tag, v.embed_tag),
    parser_version = coalesce(t.parser_version, '{UNKNOWN}'),
    first_seen_at  = coalesce(t.first_seen_at, t.ingested_at),
    last_seen_at   = coalesce(t.last_seen_at, t.ingested_at)
FROM (VALUES %s) AS v(nct_id, doc_hash, content_hash, embed_tag)
WHERE t.nct_id = v.nct_id AND t.doc_hash IS NULL
"""  # noqa: S608 -- UNKNOWN is a module constant

_ROW_COLS = ("nct_id", "source", "inclusion_criteria", "exclusion_criteria", *CONTENT_FIELDS)


def _columns_exist(cur) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'trials' AND column_name = 'doc_hash'"
    )
    return cur.fetchone() is not None


def _pending(cur, after: str, migrated: bool) -> list[dict]:
    """Rows still to backfill, keyset-paginated so no cursor outlives a statement."""
    where = "doc_hash IS NULL AND " if migrated else ""
    cur.execute(
        f"SELECT {', '.join(_ROW_COLS)} FROM trials "  # noqa: S608 -- module constants only
        f"WHERE {where}nct_id > %s ORDER BY nct_id LIMIT %s",
        (after, BATCH),
    )
    return [dict(zip(_ROW_COLS, r, strict=True)) for r in cur.fetchall()]


def verify_embed_tag(sample: int, threshold: float) -> dict:
    """Re-embed a random sample and compare with the stored vectors."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT title, inclusion_criteria, exclusion_criteria, embedding::text "
            "FROM trials WHERE source = %s AND embedding IS NOT NULL "
            "ORDER BY random() LIMIT %s",
            (SOURCE, sample),
        )
        rows = cur.fetchall()
    if not rows:
        return {"sampled": 0, "passed": False, "reason": "no embedded rows"}

    texts = [
        eligibility_text_for_embedding(
            {"title": t, "inclusion_criteria": inc or [], "exclusion_criteria": exc or []}
        )
        for t, inc, exc, _ in rows
    ]
    stored = np.array(
        [np.fromstring(e[1:-1], sep=",", dtype=np.float32) for *_, e in rows]
    )
    fresh = embed_matrix(texts)
    stored /= np.linalg.norm(stored, axis=1, keepdims=True)
    fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
    cos = np.sum(stored * fresh, axis=1)
    return {
        "sampled": len(rows),
        "tag": embed_tag(),
        "cos_min": round(float(cos.min()), 6),
        "cos_p01": round(float(np.percentile(cos, 1)), 6),
        "cos_mean": round(float(cos.mean()), 6),
        "threshold": threshold,
        "passed": bool(cos.min() >= threshold),
    }


def migrate(dry_run: bool, sample: int, threshold: float) -> dict:
    with get_conn() as conn, conn.cursor() as cur:
        migrated = _columns_exist(cur)
        if not dry_run:
            cur.execute(PROVENANCE_DDL)
            migrated = True

    check = verify_embed_tag(sample, threshold)
    live_tag = check["tag"] if check["passed"] else UNKNOWN

    total = 0
    by_source: dict[str, int] = {}
    after = ""
    while True:
        with get_conn() as conn, conn.cursor() as cur:
            rows = _pending(cur, after, migrated)
            if not rows:
                break
            after = rows[-1]["nct_id"]
            updates = [
                (
                    r["nct_id"],
                    doc_hash(r),
                    content_hash(r),
                    live_tag if r["source"] == SOURCE else UNKNOWN,
                )
                for r in rows
            ]
            for r in rows:
                by_source[r["source"]] = by_source.get(r["source"], 0) + 1
            total += len(rows)
            if dry_run:
                continue
            psycopg2.extras.execute_values(cur, BACKFILL_SQL, updates, page_size=500)

    return {"dry_run": dry_run, "embed_check": check, "stamped_tag": live_tag,
            "rows": total, "by_source": by_source}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report only; no DDL, no writes")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.999)
    args = parser.parse_args()
    report = migrate(args.dry_run, args.sample, args.threshold)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
