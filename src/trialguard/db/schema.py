"""Create and manage the pgvector schema for TrialGuard."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress

import psycopg2
from psycopg2 import pool

from trialguard.config import settings

JOBS_DDL = """
-- Assess jobs and their event log (Phase 10 WS-1). Previously an in-process dict
-- with TTL eviction, so a Fly restart -- deploy, host migration, OOM -- took every
-- in-flight job with it: the user had paid for the LLM calls and the stream hung
-- until the client gave up.
--
-- instance_id is the *process* that owns the row, not the machine. fly.toml
-- suspends idle machines and suspend snapshots RAM, so a resumed process is alive
-- with a heartbeat frozen for the whole suspension; a heartbeat age test alone
-- would report it dead. A machine keeps its Fly id across a restart, so the Fly id
-- cannot make that distinction either. A per-process value can: suspend preserves
-- it, and a deploy, crash or OOM starts a new interpreter with a new one. See
-- AD-13.
CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    note              TEXT NOT NULL,
    nct_ids           TEXT[] NOT NULL,
    status            TEXT NOT NULL DEFAULT 'queued',
    error             TEXT,
    skip_cache_write  BOOLEAN NOT NULL DEFAULT TRUE,
    instance_id       TEXT NOT NULL,
    heartbeat_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS jobs_created_at_idx ON jobs(created_at);

-- seq is assigned under a FOR UPDATE lock on the parent job rather than from a
-- sequence. A BIGSERIAL is allocated before commit, so a reader can observe a
-- later id committed while an earlier one is still in flight and skip it forever
-- -- which is exactly the gap WS-2's Last-Event-ID cursor must not have.
CREATE TABLE IF NOT EXISTS job_events (
    job_id      TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    event       JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (job_id, seq)
);
"""

# Row provenance and the refresh run ledger (pipeline 1 hardening, WS-1).
# Additive and idempotent: applied by init_schema for a fresh database and by
# scripts/migrate_provenance.py for the live one. Kept apart from the CREATE
# TABLE so both paths share one definition.
#
# doc_hash is the hash of the exact string embedded, content_hash of every
# CT.gov field stored; the refresh diffs on these rather than on the
# day-granular lastUpdatePostDate. embed_tag and parser_version record what
# built the row, so a config or parser change is detectable instead of mixing
# vector spaces and parses silently. expired_at is soft expiry: a trial that
# leaves the enrolling set keeps its row, history and embedding.
PROVENANCE_DDL = """
ALTER TABLE trials ADD COLUMN IF NOT EXISTS doc_hash        TEXT;
ALTER TABLE trials ADD COLUMN IF NOT EXISTS content_hash    TEXT;
ALTER TABLE trials ADD COLUMN IF NOT EXISTS embed_tag       TEXT;
ALTER TABLE trials ADD COLUMN IF NOT EXISTS parser_version  TEXT;
ALTER TABLE trials ADD COLUMN IF NOT EXISTS first_seen_at   TIMESTAMPTZ;
ALTER TABLE trials ADD COLUMN IF NOT EXISTS last_seen_at    TIMESTAMPTZ;
ALTER TABLE trials ADD COLUMN IF NOT EXISTS expired_at      TIMESTAMPTZ;
ALTER TABLE trials ADD COLUMN IF NOT EXISTS expired_reason  TEXT;
ALTER TABLE trials ADD COLUMN IF NOT EXISTS missing_runs    SMALLINT NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS trials_active_idx ON trials(source) WHERE expired_at IS NULL;

CREATE TABLE IF NOT EXISTS refresh_runs (
    run_id                TEXT PRIMARY KEY,
    source                TEXT NOT NULL,
    started_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    heartbeat_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at           TIMESTAMPTZ,
    outcome               TEXT NOT NULL DEFAULT 'running',
    reason                TEXT,
    ctgov_data_timestamp  TEXT,
    ctgov_total_count     INTEGER,
    fetched               INTEGER,
    counts                JSONB,
    gates                 JSONB,
    embed_tag             TEXT,
    parser_version        TEXT,
    git_sha               TEXT,
    duration_s            REAL
);

CREATE INDEX IF NOT EXISTS refresh_runs_started_idx
    ON refresh_runs(source, started_at DESC);
"""

# f-string: JOBS_DDL is spliced in so the two tables have exactly one definition,
# and a test can create them without the pgvector extension the rest of this needs.
DDL = f"""
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

CREATE INDEX IF NOT EXISTS trials_doc_tsv_idx
    ON trials USING gin (doc_tsv);

{PROVENANCE_DDL}

-- Durable cache. The disk caches under data/cache/ live on the container's
-- ephemeral rootfs, so every deploy silently re-rolled them: keyword regeneration
-- changes the queries that drive ranking, which made served retrieval
-- irreproducible across deploys. Rows survive deploys and can be capped or
-- expired, which attacker-controlled filenames cannot.
CREATE TABLE IF NOT EXISTS cache_entries (
    namespace   TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (namespace, key)
);

-- Spend accounting. The JSON ledger was read-modify-write behind a lock that was
-- recreated per call, so concurrent workers lost spend; it also lived on the
-- ephemeral rootfs, so the daily cap reset on deploy and was per-machine. Both
-- go away here: the upsert below is atomic in the database, so no application
-- lock is needed at all.
CREATE TABLE IF NOT EXISTS spend_ledger (
    day     DATE PRIMARY KEY,
    usd     NUMERIC(16,8) NOT NULL DEFAULT 0,
    tokens  BIGINT        NOT NULL DEFAULT 0,
    calls   INTEGER       NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS spend_by_model (
    day        DATE NOT NULL,
    model_key  TEXT NOT NULL,          -- "provider|served_model"
    usd        NUMERIC(16,8) NOT NULL DEFAULT 0,
    tokens     BIGINT        NOT NULL DEFAULT 0,
    calls      INTEGER       NOT NULL DEFAULT 0,
    source     TEXT,
    PRIMARY KEY (day, model_key)
);

{JOBS_DDL}
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


# A connection idle longer than this is pinged before it is handed out. Neon
# suspends an idle compute after ~5 min and drops its connections, and the pool
# cannot see that: the corpus refresh embeds for minutes between DB calls, got a
# dead connection back on its next lease, and failed 277 hourly runs in a row
# (2026-09-26 to 10-07, all "connection already closed"). Busy connections skip
# the ping, so the served path pays nothing.
_IDLE_PING_S = 60.0
_last_used: dict[int, float] = {}


def _alive(conn) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.rollback()
        return True
    except psycopg2.Error:
        return False


def _lease(p: pool.ThreadedConnectionPool):
    """A live connection: dead ones are closed and replaced, never handed out."""
    for _ in range(_POOL_MAX + 1):
        conn = p.getconn()
        idle = time.monotonic() - _last_used.get(id(conn), 0.0)
        if not conn.closed and (idle < _IDLE_PING_S or _alive(conn)):
            return conn
        _last_used.pop(id(conn), None)
        p.putconn(conn, close=True)
    raise psycopg2.OperationalError("no live database connection could be leased")


@contextmanager
def get_conn() -> Iterator:
    """Lease a pooled connection; commit on success, return to the pool on exit.

    Callers keep `with get_conn() as conn:` — the previous per-call connect()
    leaked TCP sessions under concurrent retrieve().
    """
    p = _get_pool()
    conn = _lease(p)
    try:
        yield conn
        conn.commit()
    except Exception:
        # A connection the server already dropped cannot roll back, and the
        # InterfaceError that raises would replace the error worth reporting.
        with suppress(psycopg2.Error):
            conn.rollback()
        raise
    finally:
        if conn.closed:
            _last_used.pop(id(conn), None)
            p.putconn(conn, close=True)
        else:
            _last_used[id(conn)] = time.monotonic()
            p.putconn(conn)


def close_pool() -> None:
    """Shut down the pool (tests / process teardown)."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


# ivfflat centroids are trained from the rows present when the index is built
# and never move afterwards. Created by the DDL, it was trained on an empty table
# on every fresh database. lists = 161 is the Phase 7 tuning for ~26k rows
# (data/reports/phase7_retrieval.md); scripts/migrate_fts.py rebuilds it.
VECTOR_INDEX_MIN_ROWS = 1000
VECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS trials_embedding_idx
    ON trials USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 161)
"""


def ensure_vector_index() -> bool:
    """Build the ivfflat index once there is data to train it on. Returns built."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('trials_embedding_idx')")
        if cur.fetchone()[0] is not None:
            return False
        cur.execute("SELECT count(*) FROM trials WHERE embedding IS NOT NULL")
        if cur.fetchone()[0] < VECTOR_INDEX_MIN_ROWS:
            return False
        cur.execute(VECTOR_INDEX_SQL)
    return True


def init_schema() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(DDL)
    ensure_vector_index()
    print("Schema initialised.")
