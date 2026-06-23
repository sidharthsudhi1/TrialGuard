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
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    """Run hybrid retrieval. Returns (results, latency_ms_breakdown).

    results: list of (nct_id, rrf_score) sorted descending, length top_k.
    latency: {"dense_ms", "bm25_ms", "fusion_ms", "total_ms"}
    """
    t0 = time.perf_counter()

    t1 = time.perf_counter()
    dense_results = dense_search(query, top_k=dense_pool, source=source)
    dense_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    bm25_results = bm25_search(query, top_k=bm25_pool, source=source)
    bm25_ms = (time.perf_counter() - t2) * 1000

    t3 = time.perf_counter()
    fused = rrf([dense_results, bm25_results], top_k=top_k)
    fusion_ms = (time.perf_counter() - t3) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    latency = {
        "dense_ms": round(dense_ms, 1),
        "bm25_ms": round(bm25_ms, 1),
        "fusion_ms": round(fusion_ms, 1),
        "total_ms": round(total_ms, 1),
    }
    return fused, latency
