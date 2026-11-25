"""Retrieval pipeline: dense + BM25 fused with RRF."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from trialguard.retrieval.bm25 import bm25_search
from trialguard.retrieval.dense import dense_search
from trialguard.retrieval.fusion import importance_weights, list_pool, rrf


def _apply_demographics(query, rankings, fused, top_k, source, wide=None):
    """Drop candidates a hard age or sex gate rules out. Returns (hits, n_dropped).

    Re-fuses wider before filtering so the caller still gets top_k results rather
    than a short list — dropping from an already-sliced list would silently shrink
    the page. Fails open: any error here degrades to unfiltered results, because a
    ranking with a few impossible candidates beats no ranking at all.
    """
    from trialguard.config import settings

    if not (settings.retrieval_demographic_filter and source and settings.database_url):
        return fused, 0
    try:
        from trialguard.retrieval.demographics import filter_candidates, parse_patient

        patient = parse_patient(query)
        if patient["age"] is None and not patient["sex"]:
            return fused, 0

        from trialguard.db.queries import get_trials

        if wide is None:
            wide = rrf(rankings, top_k=top_k * 4, weights=importance_weights(len(rankings)))
        meta = get_trials([n for n, _ in wide], source=source)
        kept, dropped = filter_candidates(wide, meta, patient)
        return kept[:top_k], len(dropped)
    except Exception:
        return fused, 0


def _fanout(queries, dense_pool, bm25_pool, source):
    """Per-keyword dense and lexical lists, position-indexed so ties stay stable."""
    tasks: list[tuple[str, str]] = []
    for q in queries:
        tasks.append(("dense", q))
        tasks.append(("bm25", q))
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

        workers = min(len(tasks), max(1, settings.retrieval_fanout_workers))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tg-retrieve") as pool:
            list(pool.map(_run, range(len(tasks))))
    fanout_ms = (time.perf_counter() - tf) * 1000
    dense_ms = sum(e for e, (kind, _) in zip(elapsed, tasks, strict=True) if kind == "dense")
    bm25_ms = sum(e for e, (kind, _) in zip(elapsed, tasks, strict=True) if kind == "bm25")
    return rankings, dense_ms, bm25_ms, fanout_ms


def _dense_batch(queries, dense_pool, source):
    """One GEMM when the in-memory matrix is resident, else None."""
    from trialguard.ingestion.embed import embed_matrix
    from trialguard.retrieval.vector_cache import cached_search_queries

    vecs = embed_matrix(queries, is_query=True)
    return cached_search_queries(vecs, dense_pool, source)


def retrieve(
    query: str,
    top_k: int = 10,
    source: str | None = None,
    dense_pool: int | None = None,
    bm25_pool: int | None = None,
    use_keywords: bool = False,
    handler=None,
    on_provisional=None,
    persist_keywords: bool = True,
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    """Run hybrid retrieval. Returns (results, latency_ms_breakdown).

    results: list of (nct_id, rrf_score) sorted descending, length top_k.
    latency: {"dense_ms", "bm25_ms", "fanout_ms", "fusion_ms", "ce_ms", "keyword_ms", "total_ms"}
    """
    t0 = time.perf_counter()
    dense_pool = dense_pool or list_pool(top_k)
    bm25_pool = bm25_pool or list_pool(top_k)

    keyword_first_ms = None
    if use_keywords and on_provisional is not None:
        # Raw-note list, shown while keywords are still being written. It is not
        # the ranking that gets assessed.
        prov_rankings, _, _, _ = _fanout([query], dense_pool, bm25_pool, source)
        on_provisional(
            rrf(prov_rankings, top_k=top_k, weights=importance_weights(len(prov_rankings)))
        )

    if use_keywords:
        import os

        from trialguard.retrieval.query_transform import generate_keywords

        tk = time.perf_counter()
        stream = os.environ.get("TG_KEYWORD_STREAM", "1") != "0"
        started: list[str] = []
        bm25_futs = []
        pool = None
        if stream:
            from trialguard.config import settings

            pool = ThreadPoolExecutor(
                max_workers=max(1, settings.retrieval_fanout_workers),
                thread_name_prefix="tg-kw",
            )

            def _on_kw(kw: str) -> None:
                nonlocal keyword_first_ms
                if keyword_first_ms is None:
                    keyword_first_ms = (time.perf_counter() - tk) * 1000
                started.append(kw)
                bm25_futs.append(pool.submit(bm25_search, kw, bm25_pool, source))

        queries = generate_keywords(
            query,
            handler=handler,
            on_keyword=_on_kw if stream else None,
            persist=persist_keywords,
        )
        keyword_ms = (time.perf_counter() - tk) * 1000
        if stream and started == queries and bm25_futs:
            dense_lists = _dense_batch(queries, dense_pool, source)
            if dense_lists is None:
                dense_lists = [
                    dense_search(q, top_k=dense_pool, source=source) for q in queries
                ]
            rankings = []
            for i, dense in enumerate(dense_lists):
                rankings.append(dense)
                rankings.append(bm25_futs[i].result())
            if pool is not None:
                pool.shutdown(wait=True)
            dense_ms_total = 0.0
            bm25_ms_total = 0.0
            fanout_ms = (time.perf_counter() - tk) * 1000 - keyword_ms
        else:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            rankings, dense_ms_total, bm25_ms_total, fanout_ms = _fanout(
                queries, dense_pool, bm25_pool, source
            )
    else:
        queries = [query]
        keyword_ms = 0.0
        rankings, dense_ms_total, bm25_ms_total, fanout_ms = _fanout(
            queries, dense_pool, bm25_pool, source
        )

    from trialguard.retrieval import ce_rerank as rerank

    use_ce = rerank.enabled(top_k, use_keywords)
    t3 = time.perf_counter()
    fused = rrf(
        rankings,
        top_k=rerank.CANDIDATES if use_ce else top_k,
        weights=importance_weights(len(rankings)),
    )
    fusion_ms = (time.perf_counter() - t3) * 1000

    ce_ms, wide = 0.0, None
    if use_ce:
        from trialguard.db.queries import get_trials

        tc = time.perf_counter()
        meta = get_trials([n for n, _ in fused], source=source)
        # The whole reranked pool goes to the demographic filter, so trials it
        # drops are back-filled in cross-encoder order rather than re-fused.
        wide = rerank.rerank(queries, fused, {n: rerank.doc_text(t) for n, t in meta.items()})
        fused = wide[:top_k]
        ce_ms = (time.perf_counter() - tc) * 1000

    t4 = time.perf_counter()
    fused, n_dropped = _apply_demographics(query, rankings, fused, top_k, source, wide)
    demographics_ms = (time.perf_counter() - t4) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    latency = {
        "keyword_ms": round(keyword_ms, 1),
        "keyword_first_ms": None if keyword_first_ms is None else round(keyword_first_ms, 1),
        # Cumulative query work, not elapsed time: the searches overlap now, so
        # these sum to more than the wall clock they actually cost. fanout_ms is
        # the elapsed figure; keeping both is what makes "is the DB slow, or are
        # we just serializing?" answerable from one response.
        "dense_ms": round(dense_ms_total, 1),
        "bm25_ms": round(bm25_ms_total, 1),
        "fanout_ms": round(fanout_ms, 1),
        "fusion_ms": round(fusion_ms, 1),
        "ce_ms": round(ce_ms, 1),
        "demographics_ms": round(demographics_ms, 1),
        "demographics_dropped": n_dropped,
        "total_ms": round(total_ms, 1),
    }
    return fused, latency
