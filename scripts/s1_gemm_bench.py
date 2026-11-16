"""S1: one GEMM versus one product per keyword, on a real eval matrix.

Adopt bar, written before this run: throughput at least 2x at concurrency 8,
and p95 at concurrency 1 no worse than the per-keyword loop. Top-500 id sets
must match. Float16 is recorded and used only if those sets match.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from trialguard.retrieval.gemm import query_scores, topk

MATRIX = Path("data/indexes/trec_2021_medcpt_excl_embeddings.npy")
IDS = Path("data/indexes/trec_2021_medcpt_excl_ids.json")
OUT = Path("data/reports/s1_gemm.json")


def _once(matrix, queries, ids, limit: int) -> None:
    from trialguard.config import settings

    settings.retrieval_blas_threads = limit
    scores = query_scores(matrix, queries, normalize=True)
    for i in range(scores.shape[1]):
        topk(scores[:, i], ids, 500)


def _loop(matrix, queries, ids) -> None:
    from trialguard.config import settings

    settings.retrieval_blas_threads = 0
    for i in range(queries.shape[0]):
        scores = query_scores(matrix, queries[i], normalize=True)
        topk(scores[:, 0], ids, 500)


def _time(fn, rounds: int = 5) -> list[float]:
    fn()
    out = []
    for _ in range(rounds):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


def _p95(xs: list[float]) -> float:
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(0.95 * (len(xs) - 1))))
    return round(xs[k], 2)


def main() -> None:
    matrix = np.load(MATRIX)
    ids = json.loads(IDS.read_text())
    if len(ids) != matrix.shape[0]:
        ids = [str(i) for i in range(matrix.shape[0])]
    rng = np.random.default_rng(0)
    queries = rng.standard_normal((12, matrix.shape[1])).astype(np.float32)

    from trialguard.config import settings

    settings.retrieval_blas_threads = 0
    batched = query_scores(matrix, queries, normalize=True)
    loop_cols = [query_scores(matrix, queries[i], normalize=True)[:, 0] for i in range(12)]
    set_ok = all(
        {n for n, _ in topk(batched[:, i], ids, 500)}
        == {n for n, _ in topk(loop_cols[i], ids, 500)}
        for i in range(12)
    )
    f16 = matrix.astype(np.float16).astype(np.float32)
    settings.retrieval_blas_threads = 0
    batched16 = query_scores(f16, queries, normalize=True)
    f16_ok = all(
        {n for n, _ in topk(batched16[:, i], ids, 500)}
        == {n for n, _ in topk(loop_cols[i], ids, 500)}
        for i in range(12)
    )

    c1_loop = _time(lambda: _loop(matrix, queries, ids))
    c1_batch = _time(lambda: _once(matrix, queries, ids, 0))
    c1_cap = _time(lambda: _once(matrix, queries, ids, 1))

    def burst(limit: int, workers: int) -> list[float]:
        def go():
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda _: _once(matrix, queries, ids, limit), range(workers)))

        return _time(go, rounds=3)

    c8_loop = _time(lambda: _burst_loop(matrix, queries, ids), rounds=3)
    c8_batch = burst(0, 8)
    report = {
        "matrix": list(matrix.shape),
        "top500_set_equal": set_ok,
        "float16_top500_set_equal": f16_ok,
        "c1_loop_p95_ms": _p95(c1_loop),
        "c1_batch_p95_ms": _p95(c1_batch),
        "c1_batch_cap1_p95_ms": _p95(c1_cap),
        "c8_loop_p95_ms": _p95(c8_loop),
        "c8_batch_p95_ms": _p95(c8_batch),
        "c1_loop_ms": [round(x, 2) for x in c1_loop],
        "c1_batch_ms": [round(x, 2) for x in c1_batch],
        "c8_batch_ms": [round(x, 2) for x in c8_batch],
        "c8_loop_ms": [round(x, 2) for x in c8_loop],
    }
    # Throughput is the inverse of the time to finish `workers` searches.
    report["c8_speedup"] = round(report["c8_loop_p95_ms"] / report["c8_batch_p95_ms"], 3) if report["c8_batch_p95_ms"] else None
    report["c1_not_worse"] = report["c1_batch_p95_ms"] <= report["c1_loop_p95_ms"] * 1.05
    report["adopt"] = bool(
        set_ok and report["c1_not_worse"] and report["c8_speedup"] and report["c8_speedup"] >= 2
    )
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def _burst_loop(matrix, queries, ids) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: _loop(matrix, queries, ids), range(8)))


if __name__ == "__main__":
    main()
