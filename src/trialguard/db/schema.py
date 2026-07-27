"""Create and manage the pgvector schema for TrialGuard."""

import psycopg2

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


def get_conn():
    # Keepalives so long ingest runs don't get silently dropped by the Neon
    # serverless proxy mid-statement (observed: SSL SYSCALL timeout at ~18k rows).
    return psycopg2.connect(
        settings.database_url,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )


def init_schema() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(DDL)
        conn.commit()
    print("Schema initialised.")
