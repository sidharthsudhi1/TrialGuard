"""Retrieval pipeline: dense + BM25 fused with RRF."""

from __future__ import annotations

import time

from trialguard.retrieval.bm25 import bm25_search
from trialguard.retrieval.dense import dense_search
from trialguard.retrieval.fusion import rrf


def retrieve(
    query: str,
    top_k: int = 10,
    source: str | None = None,
    dense_pool: int = 50,
    bm25_pool: int = 50,
    use_keywords: bool = False,
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    """Run hybrid retrieval. Returns (results, latency_ms_breakdown).

    results: list of (nct_id, rrf_score) sorted descending, length top_k.
    latency: {"dense_ms", "bm25_ms", "fusion_ms", "keyword_ms", "total_ms"}
    """
    t0 = time.perf_counter()

    if use_keywords:
        from trialguard.retrieval.query_transform import generate_keywords
        tk = time.perf_counter()
        queries = generate_keywords(query)
        keyword_ms = (time.perf_counter() - tk) * 1000
    else:
        queries = [query]
        keyword_ms = 0.0

    all_rankings: list[list[tuple[str, float]]] = []
    dense_ms_total = 0.0
    bm25_ms_total = 0.0

    for q in queries:
        t1 = time.perf_counter()
        all_rankings.append(dense_search(q, top_k=dense_pool, source=source))
        dense_ms_total += (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        all_rankings.append(bm25_search(q, top_k=bm25_pool, source=source))
        bm25_ms_total += (time.perf_counter() - t2) * 1000

    t3 = time.perf_counter()
    fused = rrf(all_rankings, top_k=top_k)
    fusion_ms = (time.perf_counter() - t3) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    latency = {
        "keyword_ms": round(keyword_ms, 1),
        "dense_ms": round(dense_ms_total, 1),
        "bm25_ms": round(bm25_ms_total, 1),
        "fusion_ms": round(fusion_ms, 1),
        "total_ms": round(total_ms, 1),
    }
    return fused, latency
