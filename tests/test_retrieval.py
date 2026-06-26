from trialguard.retrieval.fusion import rrf
from trialguard.eval.retrieval_metrics import recall_at_k, mrr, ndcg_at_k


# RRF tests

def test_rrf_basic():
    r1 = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
    r2 = [("B", 0.95), ("A", 0.85), ("D", 0.6)]
    result = rrf([r1, r2], k=60)
    ids = [nct for nct, _ in result]
    # A and B both appear in both lists, should rank highest
    assert ids[0] in ("A", "B")
    assert ids[1] in ("A", "B")


def test_rrf_single_list():
    r1 = [("X", 0.9), ("Y", 0.8)]
    result = rrf([r1], k=60)
    assert result[0][0] == "X"


def test_rrf_respects_top_k():
    r1 = [(str(i), float(i)) for i in range(100)]
    result = rrf([r1], top_k=10)
    assert len(result) == 10


# Metrics tests

def test_recall_at_k_perfect():
    preds = ["A", "B", "C"]
    gold = {"A", "B"}
    assert recall_at_k(preds, gold, k=3) == 1.0


def test_recall_at_k_miss():
    preds = ["X", "Y", "Z"]
    gold = {"A"}
    assert recall_at_k(preds, gold, k=3) == 0.0


def test_recall_at_k_partial():
    preds = ["A", "X", "B", "Y"]
    gold = {"A", "B", "C"}
    # k=2: only A found in top 2, 1/3
    assert abs(recall_at_k(preds, gold, k=2) - 1/3) < 1e-6


def test_mrr_first_hit():
    preds = ["X", "A", "B"]
    gold = {"A"}
    assert abs(mrr(preds, gold) - 0.5) < 1e-6


def test_mrr_no_hit():
    preds = ["X", "Y"]
    gold = {"A"}
    assert mrr(preds, gold) == 0.0


def test_ndcg_perfect():
    preds = ["A", "B"]
    labels = {"A": "eligible", "B": "eligible"}
    score = ndcg_at_k(preds, labels, k=2)
    assert abs(score - 1.0) < 1e-6
