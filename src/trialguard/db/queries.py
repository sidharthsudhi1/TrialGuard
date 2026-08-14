"""Trial row lookups for the production serving path."""

from __future__ import annotations

from trialguard.db.schema import get_conn

_TRIAL_COLS = (
    "nct_id",
    "title",
    "status",
    "phase",
    "conditions",
    "interventions",
    "eligibility_raw",
    "inclusion_criteria",
    "exclusion_criteria",
    "min_age",
    "max_age",
    "sex",
    "healthy_volunteers",
    "source",
)

_SELECT = ", ".join(_TRIAL_COLS)


def get_trials(nct_ids: list[str], source: str | None = None) -> dict[str, dict]:
    """Batch-fetch trial rows by NCT ID in one query. Returns nct_id → row dict."""
    if not nct_ids:
        return {}
    sql = f"SELECT {_SELECT} FROM trials WHERE nct_id = ANY(%s)"
    params: list = [list(nct_ids)]
    if source:
        sql += " AND source = %s"
        params.append(source)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return {row[0]: dict(zip(cols, row, strict=True)) for row in rows}


def get_trial(nct_id: str, source: str | None = None) -> dict | None:
    """Fetch a single trial row, or None if missing."""
    return get_trials([nct_id], source=source).get(nct_id)
