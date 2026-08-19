"""Create and manage the pgvector schema for TrialGuard."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from psycopg2 import pool

from trialguard.config import settings

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE OR REPLACE FUNCTION trials_doc_tsv(_title text, _incl text[], _excl text[])
RETURNS tsvector LANGUAGE sql IMMUTABLE AS $func$
    SELECT to_tsvector('english',
        coalesce(_title, '') || ' ' ||
        coalesce(array_to_string(_incl, ' '), '') || ' ' ||
        coalesce(array_to_string(_excl, ' '), ''))
$func$;

CREATE TABLE IF NOT EXISTS trials (
    nct_id              TEXT PRIMARY KEY,
    title               TEXT,
    status              TEXT,
    phase               TEXT,
    conditions          TEXT[],
    interventions       TEXT[],
    eligibility_raw     TEXT,
    inclusion_criteria  TEXT[],
    exclusion_criteria  TEXT[],
    min_age             TEXT,
    max_age             TEXT,
    sex                 TEXT,
    healthy_volunteers  BOOLEAN,
    last_updated        TEXT,
    embedding           VECTOR(768),
    metadata            JSONB,
    doc_tsv             TSVECTOR GENERATED ALWAYS AS (
        trials_doc_tsv(title, inclusion_criteria, exclusion_criteria)
    ) STORED,
    source              TEXT DEFAULT 'ctgov_live',
    ingested_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS trials_source_idx ON trials(source);

CREATE INDEX IF NOT EXISTS trials_embedding_idx
    ON trials USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 161);

CREATE INDEX IF NOT EXISTS trials_doc_tsv_idx
    ON trials USING gin (doc_tsv);

CREATE TABLE IF NOT EXISTS eval_patients (
    patient_id   TEXT,
    cohort       TEXT,
    description  TEXT,
    raw          JSONB,
    PRIMARY KEY (patient_id, cohort)
);

CREATE TABLE IF NOT EXISTS eval_labels (
    patient_id  TEXT,
    nct_id      TEXT,
    cohort      TEXT,
    label       TEXT,
    PRIMARY KEY (patient_id, nct_id, cohort)
);
"""

# Neon serverless proxies connections; a modest pool is correct. One retrieve()
# now fans its (keyword x backend) searches out concurrently, leasing up to
# settings.retrieval_fanout_workers connections at once, so the ceiling has to
# clear that with headroom for a few concurrent users. psycopg2's pool raises
# rather than waits when exhausted, so this is a correctness bound, not a tuning
# knob: too low turns a slow search into a failed one.
_POOL_MIN = 1
_POOL_MAX = 20

_pool: pool.ThreadedConnectionPool | None = None


def _keepalive_kwargs() -> dict:
    # Keepalives so long ingest runs don't get silently dropped by the Neon
    # serverless proxy mid-statement (observed: SSL SYSCALL timeout at ~18k rows).
    return {
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }


def _get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL is not configured")
        _pool = pool.ThreadedConnectionPool(
            _POOL_MIN,
            _POOL_MAX,
            settings.database_url,
            **_keepalive_kwargs(),
        )
    return _pool


@contextmanager
def get_conn() -> Iterator:
    """Lease a pooled connection; commit on success, return to the pool on exit.

    Callers keep `with get_conn() as conn:` — the previous per-call connect()
    leaked TCP sessions under concurrent retrieve().
    """
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def close_pool() -> None:
    """Shut down the pool (tests / process teardown)."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


def init_schema() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(DDL)
    print("Schema initialised.")
