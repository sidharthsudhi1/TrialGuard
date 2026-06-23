"""Create and manage the pgvector schema for TrialGuard."""

import psycopg2
from trialguard.config import settings

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

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
    embedding           VECTOR(384),
    metadata            JSONB,
    source              TEXT DEFAULT 'ctgov_live',
    ingested_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS trials_source_idx ON trials(source);

CREATE INDEX IF NOT EXISTS trials_embedding_idx
    ON trials USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 55);

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
    return psycopg2.connect(settings.database_url)


def init_schema() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(DDL)
        conn.commit()
    print("Schema initialised.")
