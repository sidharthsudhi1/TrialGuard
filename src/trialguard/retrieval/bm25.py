"""BM25 retrieval using rank-bm25. Index built in-memory, cached per source."""

from __future__ import annotations

import re
from functools import lru_cache

from trialguard.db.schema import get_conn


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def _load_corpus(source: str | None) -> tuple[list[str], list[str]]:
    """Return (nct_ids, tokenized_docs) for given source scope."""
    sql = """
    SELECT nct_id,
           title || ' ' || array_to_string(inclusion_criteria, ' ')
    FROM trials
    WHERE embedding IS NOT NULL
    {clause}
    ORDER BY nct_id;
    """.format(clause="AND source = %s" if source else "")

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (source,) if source else ())
        rows = cur.fetchall()

    nct_ids = [r[0] for r in rows]
    docs = [_tokenize(r[1] or "") for r in rows]
    return nct_ids, docs


@lru_cache(maxsize=8)
def _get_index(source: str | None):
    from rank_bm25 import BM25Okapi  # type: ignore

    nct_ids, docs = _load_corpus(source)
    index = BM25Okapi(docs)
    return nct_ids, index


def bm25_search(
    query_text: str,
    top_k: int = 50,
    source: str | None = None,
) -> list[tuple[str, float]]:
    """Return (nct_id, bm25_score) sorted descending."""
    nct_ids, index = _get_index(source)
    tokens = _tokenize(query_text)
    scores = index.get_scores(tokens)

    ranked = sorted(
        zip(nct_ids, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]


def invalidate_cache() -> None:
    _get_index.cache_clear()
