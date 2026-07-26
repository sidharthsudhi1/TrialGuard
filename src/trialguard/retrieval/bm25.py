"""Lexical retrieval via Postgres full-text search (tsvector + GIN).

Ranks with ts_rank_cd over the generated doc_tsv column (title + inclusion +
exclusion). Lexical search runs in the DB against a GIN index, so it scales with
the corpus instead of rebuilding an in-memory BM25 index from a full table scan on
every cold start. Keeps the (nct_id, score) contract the RRF fusion expects.

Named bm25_search for the fusion interface; ts_rank_cd is not Okapi BM25 (no IDF
saturation / length normalization), so ranking is re-validated against gold before
it is trusted (PHASE7 WS-4).
"""

from __future__ import annotations

from trialguard.db.schema import get_conn

SQL = """
SELECT nct_id, ts_rank_cd(doc_tsv, query) AS score
FROM trials, websearch_to_tsquery('english', %(q)s) AS query
WHERE doc_tsv @@ query
{source_clause}
ORDER BY score DESC
LIMIT %(top_k)s;
"""


def bm25_search(
    query_text: str,
    top_k: int = 50,
    source: str | None = None,
) -> list[tuple[str, float]]:
    """Return (nct_id, ts_rank_cd score) sorted descending."""
    source_clause = "AND source = %(source)s" if source else ""
    sql = SQL.format(source_clause=source_clause)

    params: dict = {"q": query_text, "top_k": top_k}
    if source:
        params["source"] = source

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [(row[0], float(row[1])) for row in cur.fetchall()]
