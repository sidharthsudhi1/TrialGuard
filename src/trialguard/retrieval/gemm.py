"""One matrix product for every query vector in a search.

K separate products each pay a BLAS launch and, when they run on K threads,
contend for the same cores. `(N, D) @ (D, K)` is the same dots.
"""

from __future__ import annotations

import numpy as np


def query_scores(matrix: np.ndarray, queries: np.ndarray, normalize: bool) -> np.ndarray:
    """Return (N, K) scores. `queries` is (K, D) or (D,)."""
    q = np.atleast_2d(np.asarray(queries, dtype=np.float32))
    if normalize:
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q = q / norms
    from trialguard.config import settings

    limit = settings.retrieval_blas_threads
    if limit <= 0:
        return matrix @ q.T
    import threadpoolctl

    with threadpoolctl.threadpool_limits(limits=limit):
        return matrix @ q.T


def topk(scores: np.ndarray, ids: list[str], top_k: int) -> list[tuple[str, float]]:
    """Highest scores, stable on ties. `scores` is one column."""
    n = int(scores.shape[0])
    k = min(top_k, n)
    if k <= 0:
        return []
    if k == n:
        idx = np.argsort(-scores, kind="stable")
    else:
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(ids[int(i)], float(scores[int(i)])) for i in idx]
