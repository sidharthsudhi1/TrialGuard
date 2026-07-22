"""Dense retrieval via pgvector cosine similarity."""

from __future__ import annotations

from trialguard.db.schema import get_conn
from trialguard.ingestion.embed import embed_text

SQL = """
SELECT nct_id, 1 - (embedding <=> %(vec)s::vector) AS score
FROM trials
WHERE embedding IS NOT NULL
{source_clause}
ORDER BY embedding <=> %(vec)s::vector
LIMIT %(top_k)s;
"""


def dense_search(
    query_text: str,
    top_k: int = 50,
    source: str | None = None,
) -> list[tuple[str, float]]:
    """Return (nct_id, cosine_similarity) sorted descending."""
    vec = embed_text(query_text, is_query=True)
    source_clause = "AND source = %(source)s" if source else ""
    sql = SQL.format(source_clause=source_clause)

    params: dict = {"vec": vec, "top_k": top_k}
    if source:
        params["source"] = source

    from trialguard.config import settings

    with get_conn() as conn, conn.cursor() as cur:
        # Widen the ivfflat search beyond the default single list; probes=1 loses
        # ~64% recall vs exact (measured in phase5_vectorstore). SET LOCAL scopes it
        # to this transaction.
        cur.execute("SET LOCAL ivfflat.probes = %s", (settings.pgvector_probes,))
        cur.execute(sql, params)
        return [(row[0], float(row[1])) for row in cur.fetchall()]
