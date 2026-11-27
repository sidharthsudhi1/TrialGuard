"""Phase 39: add trials.study_type to an existing corpus.

  python -m trialguard.scripts.migrate_study_type --dry-run
  python -m trialguard.scripts.migrate_study_type

Applies STUDY_TYPE_DDL (additive, idempotent). No backfill: study_type is in
CONTENT_FIELDS, so every stored content_hash now differs from a fresh fetch and
the next refresh rewrites each row as a metadata-only update, with no re-embed.

Run it before deploying the code that writes the column: the upsert names it,
so a refresh against an unmigrated table fails (loudly) until it is.
"""

from __future__ import annotations

import argparse
import json
import sys

from trialguard.db.schema import STUDY_TYPE_DDL, get_conn

_EXISTS_SQL = """
SELECT 1 FROM information_schema.columns
WHERE table_name = 'trials' AND column_name = 'study_type'
"""


def migrate(dry_run: bool) -> dict:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(_EXISTS_SQL)
        existed = cur.fetchone() is not None
        if not dry_run:
            cur.execute(STUDY_TYPE_DDL)
    return {"dry_run": dry_run, "column_existed": existed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report only; no DDL")
    args = parser.parse_args()
    print(json.dumps(migrate(args.dry_run), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
