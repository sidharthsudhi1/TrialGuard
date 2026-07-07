import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# query_transform tests

# recall@N sweep and gold coverage tests

def test_recall_multi_k_sweep():
    """Single ranked list sliced at multiple depths gives correct per-k recall."""
    preds = ["A", "B", "C", "D", "E", "X", "Y"]
    gold = {"A", "C", "E"}
    assert abs(recall_at_k(preds, gold, k=1) - 1/3) < 1e-6   # A only
    assert abs(recall_at_k(preds, gold, k=2) - 1/3) < 1e-6   # A, B — still 1 hit
    assert abs(recall_at_k(preds, gold, k=3) - 2/3) < 1e-6   # A, B, C — 2 hits
    assert abs(recall_at_k(preds, gold, k=5) - 1.0) < 1e-6   # A..E — all 3 gold
    assert abs(recall_at_k(preds, gold, k=7) - 1.0) < 1e-6   # beyond list — still 1.0


def test_recall_multi_k_monotone():
    """recall@k is non-decreasing as k grows."""
    preds = [str(i) for i in range(20)]
    gold = {"3", "7", "15"}
    ks = [1, 5, 10, 15, 20]
    vals = [recall_at_k(preds, gold, k) for k in ks]
    for a, b in zip(vals, vals[1:]):
        assert b >= a


def test_compute_gold_coverage_full(monkeypatch):
    """All gold positives present → coverage = 1.0, missing = 0."""
    from trialguard.eval import retrieval_metrics
    fake_labels = [
        {"patient_id": "p1", "nct_id": "NCT001", "label": "eligible"},
        {"patient_id": "p1", "nct_id": "NCT002", "label": "eligible"},
        {"patient_id": "p2", "nct_id": "NCT001", "label": "eligible"},  # duplicate NCT
    ]
    monkeypatch.setattr("trialguard.eval.cohorts.load_labels", lambda c: fake_labels)
    result = retrieval_metrics.compute_gold_coverage("sigir", {"NCT001", "NCT002", "NCT003"})
    assert result["total_gold"] == 2  # unique: NCT001, NCT002
    assert result["coverage"] == 1.0
    assert result["missing_count"] == 0
    assert result["missing_sample"] == []


def test_compute_gold_coverage_partial(monkeypatch):
    """Partial coverage → correct fraction, correct missing list."""
    from trialguard.eval import retrieval_metrics
    fake_labels = [
        {"patient_id": "p1", "nct_id": "NCT001", "label": "eligible"},
        {"patient_id": "p1", "nct_id": "NCT002", "label": "eligible"},
        {"patient_id": "p1", "nct_id": "NCT003", "label": "eligible"},
        {"patient_id": "p1", "nct_id": "NCT004", "label": "excluded"},  # not gold
    ]
    monkeypatch.setattr("trialguard.eval.cohorts.load_labels", lambda c: fake_labels)
    result = retrieval_metrics.compute_gold_coverage("sigir", {"NCT001", "NCT003"})
    assert result["total_gold"] == 3
    assert result["present_in_corpus"] == 2
    assert result["missing_count"] == 1
    assert abs(result["coverage"] - 2/3) < 1e-4
    assert "NCT002" in result["missing_sample"]


def test_compute_gold_coverage_zero(monkeypatch):
    """No gold positives in corpus → coverage = 0."""
    from trialguard.eval import retrieval_metrics
    fake_labels = [
        {"patient_id": "p1", "nct_id": "NCT001", "label": "eligible"},
    ]
    monkeypatch.setattr("trialguard.eval.cohorts.load_labels", lambda c: fake_labels)
    result = retrieval_metrics.compute_gold_coverage("sigir", {"NCT999"})
    assert result["coverage"] == 0.0
    assert result["missing_count"] == 1


# rerank tests

def test_rerank_batches_predict(tmp_path, monkeypatch):
    """predict() called once with all pairs, not in a loop."""
    from trialguard.retrieval import rerank as rerank_mod
    monkeypatch.setattr(rerank_mod, "CACHE_DIR", tmp_path)

    import numpy as np
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.9, 0.5, 0.7])
    monkeypatch.setattr(rerank_mod, "_model", mock_model)

    candidates = [("NCT001", 0.8), ("NCT002", 0.6), ("NCT003", 0.7)]
    trial_texts = {"NCT001": "trial one text", "NCT002": "trial two text", "NCT003": "trial three text"}

    rerank_mod.rerank("patient note", candidates, trial_texts, top_k=3)

    mock_model.predict.assert_called_once()
    call_args = mock_model.predict.call_args[0][0]
    assert len(call_args) == 3  # all pairs in one batch, not 3 calls


def test_rerank_returns_top_k_sorted(tmp_path, monkeypatch):
    """Returns exactly top_k results sorted by cross-encoder score desc."""
    from trialguard.retrieval import rerank as rerank_mod
    monkeypatch.setattr(rerank_mod, "CACHE_DIR", tmp_path)

    import numpy as np
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.3, 0.9, 0.6, 0.1])
    monkeypatch.setattr(rerank_mod, "_model", mock_model)

    candidates = [("A", 1.0), ("B", 0.9), ("C", 0.8), ("D", 0.7)]
    trial_texts = {k: f"text {k}" for k in "ABCD"}

    result = rerank_mod.rerank("query", candidates, trial_texts, top_k=2)

    assert len(result) == 2
    assert result[0][0] == "B"  # score 0.9 — highest
    assert result[1][0] == "C"  # score 0.6 — second


def test_rerank_cache_hit_skips_model(tmp_path, monkeypatch):
    """Cache hit returns stored scores without calling cross-encoder."""
    from trialguard.retrieval import rerank as rerank_mod
    monkeypatch.setattr(rerank_mod, "CACHE_DIR", tmp_path)

    note = "cached patient note"
    note_hash = rerank_mod._note_hash(note)
    cached_scores = {"NCT001": 0.85, "NCT002": 0.42}
    (tmp_path / f"{note_hash}.json").write_text(json.dumps(cached_scores))

    mock_model = MagicMock()
    monkeypatch.setattr(rerank_mod, "_model", mock_model)

    candidates = [("NCT001", 0.7), ("NCT002", 0.5)]
    result = rerank_mod.rerank(note, candidates, {}, top_k=2)

    mock_model.predict.assert_not_called()
    assert result[0][0] == "NCT001"  # higher cached score
    assert abs(result[0][1] - 0.85) < 1e-6


def test_rerank_writes_cache(tmp_path, monkeypatch):
    """Scores written to cache after first model call."""
    from trialguard.retrieval import rerank as rerank_mod
    monkeypatch.setattr(rerank_mod, "CACHE_DIR", tmp_path)

    import numpy as np
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.7, 0.3])
    monkeypatch.setattr(rerank_mod, "_model", mock_model)

    note = "new patient note"
    candidates = [("NCT001", 0.8), ("NCT002", 0.6)]
    trial_texts = {"NCT001": "text one", "NCT002": "text two"}

    rerank_mod.rerank(note, candidates, trial_texts, top_k=2)

    cache_path = tmp_path / f"{rerank_mod._note_hash(note)}.json"
    assert cache_path.exists()
    saved = json.loads(cache_path.read_text())
    assert "NCT001" in saved and "NCT002" in saved


# query_transform tests

def test_generate_keywords_cache_hit(tmp_path, monkeypatch):
    """Cache hit returns stored keywords without LLM call."""
    from trialguard.retrieval import query_transform

    monkeypatch.setattr(query_transform, "CACHE_DIR", tmp_path)
    note = "Patient with stage IV breast cancer, HER2-positive"
    note_hash = query_transform._note_hash(note)
    cached = ["metastatic breast cancer", "HER2-positive"]
    (tmp_path / f"{note_hash}.json").write_text(json.dumps(cached))

    with patch("langchain_groq.ChatGroq") as mock_llm:
        result = query_transform.generate_keywords(note)

    mock_llm.assert_not_called()
    assert result == cached


def test_generate_keywords_parses_llm_response(tmp_path, monkeypatch):
    """LLM response parsed, deduped, capped, and cached."""
    from trialguard.retrieval import query_transform

    monkeypatch.setattr(query_transform, "CACHE_DIR", tmp_path)
    note = "Patient with NSCLC, EGFR mutation, prior platinum"

    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "keywords": ["NSCLC EGFR mutation", "prior platinum therapy", "NSCLC EGFR mutation"]
    })
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = mock_response

    with patch("trialguard.retrieval.query_transform.ChatGroq", return_value=mock_llm_instance):
        result = query_transform.generate_keywords(note, n_max=10)

    assert "nsclc egfr mutation" in result
    assert result.count("nsclc egfr mutation") == 1  # deduped
    note_hash = query_transform._note_hash(note)
    assert (tmp_path / f"{note_hash}.json").exists()


def test_generate_keywords_fallback_on_llm_failure(tmp_path, monkeypatch):
    """LLM error falls back to raw note — never crashes."""
    from trialguard.retrieval import query_transform

    monkeypatch.setattr(query_transform, "CACHE_DIR", tmp_path)
    note = "Patient with AML, refractory"

    with patch("trialguard.retrieval.query_transform.ChatGroq", side_effect=Exception("rate limit")):
        result = query_transform.generate_keywords(note)

    assert result == [note]


def test_generate_keywords_strips_markdown_fences(tmp_path, monkeypatch):
    """Markdown code fences in LLM output are stripped before JSON parse."""
    from trialguard.retrieval import query_transform

    monkeypatch.setattr(query_transform, "CACHE_DIR", tmp_path)
    note = "Patient with CLL"

    mock_response = MagicMock()
    mock_response.content = "```json\n" + json.dumps({"keywords": ["chronic lymphocytic leukemia"]}) + "\n```"
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = mock_response

    with patch("trialguard.retrieval.query_transform.ChatGroq", return_value=mock_llm_instance):
        result = query_transform.generate_keywords(note)

    assert result == ["chronic lymphocytic leukemia"]


def test_generate_keywords_cap_at_n_max(tmp_path, monkeypatch):
    """Keywords capped at n_max regardless of LLM output length."""
    from trialguard.retrieval import query_transform

    monkeypatch.setattr(query_transform, "CACHE_DIR", tmp_path)
    note = "Complex patient"

    many_kws = [f"keyword {i}" for i in range(20)]
    mock_response = MagicMock()
    mock_response.content = json.dumps({"keywords": many_kws})
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = mock_response

    with patch("trialguard.retrieval.query_transform.ChatGroq", return_value=mock_llm_instance):
        result = query_transform.generate_keywords(note, n_max=5)

    assert len(result) == 5
