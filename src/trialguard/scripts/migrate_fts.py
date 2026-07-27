"""Phase 7 migration: Postgres FTS + retuned ivfflat on the trials table.

Applies the schema delta to an already-created DB. `CREATE TABLE IF NOT EXISTS`
in schema.py does not add columns to an existing table, so the production DB needs
this. Idempotent and re-runnable.

  python -m trialguard.scripts.migrate_fts

doc_tsv indexes title + inclusion + exclusion, matching the dense doc text
(TG_INDEX_EXCLUSION=1) and closing the old lexical/dense mismatch where in-memory
BM25 saw inclusion only. The tsvector is built through an IMMUTABLE wrapper because
`to_tsvector('english', ...)` is not accepted directly in a generated column
(the config-name lookup is not immutable); the wrapper's config is fixed, so the
assertion holds.
"""

from __future__ import annotations

from trialguard.db.schema import get_conn

_FUNC = """
CREATE OR REPLACE FUNCTION trials_doc_tsv(_title text, _incl text[], _excl text[])
RETURNS tsvector LANGUAGE sql IMMUTABLE AS $func$
    SELECT to_tsvector('english',
        coalesce(_title, '') || ' ' ||
        coalesce(array_to_string(_incl, ' '), '') || ' ' ||
        coalesce(array_to_string(_excl, ' '), ''))
$func$
"""

STEPS = [
    _FUNC,
    """ALTER TABLE trials ADD COLUMN IF NOT EXISTS doc_tsv TSVECTOR
        GENERATED ALWAYS AS (
            trials_doc_tsv(title, inclusion_criteria, exclusion_criteria)
        ) STORED""",
    "CREATE INDEX IF NOT EXISTS trials_doc_tsv_idx ON trials USING gin (doc_tsv)",
    # ivfflat lists cannot be ALTERed, so drop+recreate. Centroids are only
    # meaningful on a populated table — re-run this after the WS-2 bulk load so
    # the 161 lists cluster over real vectors, not an empty table.
    "DROP INDEX IF EXISTS trials_embedding_idx",
    """CREATE INDEX trials_embedding_idx ON trials
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 161)""",
]


def migrate() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        for sql in STEPS:
            cur.execute(sql)
        conn.commit()
    print("FTS migration applied: doc_tsv + GIN, ivfflat lists=161.")


if __name__ == "__main__":
    migrate()
