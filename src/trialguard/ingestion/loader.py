"""Upsert normalised + embedded trials into pgvector."""

from __future__ import annotations

import json

import psycopg2.extras

from trialguard.db.schema import get_conn

UPSERT_SQL = """
INSERT INTO trials (
    nct_id, title, status, phase, conditions, interventions,
    eligibility_raw, inclusion_criteria, exclusion_criteria,
    min_age, max_age, sex, healthy_volunteers, last_updated,
    embedding, metadata
) VALUES (
    %(nct_id)s, %(title)s, %(status)s, %(phase)s, %(conditions)s, %(interventions)s,
    %(eligibility_raw)s, %(inclusion_criteria)s, %(exclusion_criteria)s,
    %(min_age)s, %(max_age)s, %(sex)s, %(healthy_volunteers)s, %(last_updated)s,
    %(embedding)s::vector, %(metadata)s::jsonb
)
ON CONFLICT (nct_id) DO UPDATE SET
    title               = EXCLUDED.title,
    status              = EXCLUDED.status,
    phase               = EXCLUDED.phase,
    conditions          = EXCLUDED.conditions,
    interventions       = EXCLUDED.interventions,
    eligibility_raw     = EXCLUDED.eligibility_raw,
    inclusion_criteria  = EXCLUDED.inclusion_criteria,
    exclusion_criteria  = EXCLUDED.exclusion_criteria,
    min_age             = EXCLUDED.min_age,
    max_age             = EXCLUDED.max_age,
    sex                 = EXCLUDED.sex,
    healthy_volunteers  = EXCLUDED.healthy_volunteers,
    last_updated        = EXCLUDED.last_updated,
    embedding           = EXCLUDED.embedding,
    metadata            = EXCLUDED.metadata,
    ingested_at         = NOW();
"""


def upsert_trials(trials: list[dict]) -> int:
    """Upsert a batch of enriched trial dicts. Returns count inserted/updated."""
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
        })

    with get_conn() as conn, conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, UPSERT_SQL, rows, page_size=100)
        conn.commit()

    return len(rows)
