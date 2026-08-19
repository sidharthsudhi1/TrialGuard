"""Retrieval pipeline: dense + BM25 fused with RRF."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

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
    handler=None,
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    """Run hybrid retrieval. Returns (results, latency_ms_breakdown).

    results: list of (nct_id, rrf_score) sorted descending, length top_k.
    latency: {"dense_ms", "bm25_ms", "fanout_ms", "fusion_ms", "keyword_ms", "total_ms"}
    """
    t0 = time.perf_counter()

    if use_keywords:
        from trialguard.retrieval.query_transform import generate_keywords
        tk = time.perf_counter()
        queries = generate_keywords(query, handler=handler)
        keyword_ms = (time.perf_counter() - tk) * 1000
    else:
        queries = [query]
        keyword_ms = 0.0

    # One task per (keyword, backend). Run concurrently: every task is a blocking
    # round trip to Postgres, so the sequential loop spent its time waiting rather
    # than working — 12 keywords meant 24 round trips end to end.
    tasks: list[tuple[str, str]] = []
    for q in queries:
        tasks.append(("dense", q))
        tasks.append(("bm25", q))

    # Position-indexed, not appended on completion. RRF sums over lists so order
    # cannot change a score, but sorted() is stable, so completion order would
    # decide exact ties — and a ranking that depends on thread scheduling is not
    # reproducible.
    rankings: list[list[tuple[str, float]]] = [[] for _ in tasks]
    elapsed = [0.0] * len(tasks)

    def _run(i: int) -> None:
        kind, q = tasks[i]
        t = time.perf_counter()
        if kind == "dense":
            rankings[i] = dense_search(q, top_k=dense_pool, source=source)
        else:
            rankings[i] = bm25_search(q, top_k=bm25_pool, source=source)
        elapsed[i] = (time.perf_counter() - t) * 1000

    tf = time.perf_counter()
    if len(tasks) == 1:
        _run(0)
    else:
        from trialguard.config import settings

        # Bounded by the connection pool: psycopg2's pool raises rather than waits
        # when every connection is leased, so more workers than connections turns
        # a slow search into a failed one.
        workers = min(len(tasks), max(1, settings.retrieval_fanout_workers))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tg-retrieve") as pool:
            list(pool.map(_run, range(len(tasks))))
    fanout_ms = (time.perf_counter() - tf) * 1000

    dense_ms_total = sum(e for e, (kind, _) in zip(elapsed, tasks, strict=True) if kind == "dense")
    bm25_ms_total = sum(e for e, (kind, _) in zip(elapsed, tasks, strict=True) if kind == "bm25")

    t3 = time.perf_counter()
    fused = rrf(rankings, top_k=top_k)
    fusion_ms = (time.perf_counter() - t3) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    latency = {
        "keyword_ms": round(keyword_ms, 1),
        # Cumulative query work, not elapsed time: the searches overlap now, so
        # these sum to more than the wall clock they actually cost. fanout_ms is
        # the elapsed figure; keeping both is what makes "is the DB slow, or are
        # we just serializing?" answerable from one response.
        "dense_ms": round(dense_ms_total, 1),
        "bm25_ms": round(bm25_ms_total, 1),
        "fanout_ms": round(fanout_ms, 1),
        "fusion_ms": round(fusion_ms, 1),
        "total_ms": round(total_ms, 1),
    }
    return fused, latency
