import numpy as np

from trialguard.retrieval.gemm import query_scores, topk


def test_batch_topk_matches_one_query_at_a_time(monkeypatch):
    monkeypatch.setattr("trialguard.config.settings.retrieval_blas_threads", 0)
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((40, 8)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    queries = rng.standard_normal((5, 8)).astype(np.float32)
    ids = [f"N{i}" for i in range(40)]
    batched = query_scores(matrix, queries, normalize=True)
    for i in range(queries.shape[0]):
        alone = query_scores(matrix, queries[i], normalize=True)[:, 0]
        left = [n for n, _ in topk(batched[:, i], ids, 10)]
        right = [n for n, _ in topk(alone, ids, 10)]
        assert left == right
