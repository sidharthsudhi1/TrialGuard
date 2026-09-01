"""Cross-encoder reranker (eval-only, not on the production path)."""

from unittest.mock import MagicMock, patch

from trialguard.retrieval import rerank as R


def test_cache_key_is_scoped_to_the_model():
    """Scores are model-specific. Sharing one key across models means a second
    model silently reads back the first model's ranking and reports it as its own."""
    note = "58-year-old woman with stage III NSCLC"
    assert R._cache_key(note, "cross-encoder/ms-marco-MiniLM-L-6-v2") != R._cache_key(
        note, "ncbi/MedCPT-Cross-Encoder"
    )


def test_cache_key_is_stable_for_the_same_pair():
    assert R._cache_key("n", "m") == R._cache_key("n", "m")


def test_a_second_model_does_not_reuse_the_first_models_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "CACHE_DIR", tmp_path)
    texts = {"NCT1": "trial one", "NCT2": "trial two"}
    cands = [("NCT1", 0.0), ("NCT2", 0.0)]

    def fake_model(scores):
        m = MagicMock()
        m.predict.return_value = MagicMock(tolist=lambda: scores)
        return m

    # First model ranks NCT1 top; second ranks NCT2 top.
    with patch.object(R, "_get_model", return_value=fake_model([0.9, 0.1])):
        first = R.rerank("note", cands, texts, top_k=2, model_name="model-a")
    with patch.object(R, "_get_model", return_value=fake_model([0.1, 0.9])):
        second = R.rerank("note", cands, texts, top_k=2, model_name="model-b")

    assert first[0][0] == "NCT1"
    assert second[0][0] == "NCT2", "model-b read back model-a's cached scores"
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_default_model_is_unchanged(tmp_path, monkeypatch):
    """Omitting model_name must behave exactly as before this parameter existed."""
    monkeypatch.setattr(R, "CACHE_DIR", tmp_path)
    seen = {}

    def spy(name):
        seen["name"] = name
        m = MagicMock()
        m.predict.return_value = MagicMock(tolist=lambda: [0.5])
        return m

    with patch.object(R, "_get_model", side_effect=spy):
        R.rerank("note", [("NCT1", 0.0)], {"NCT1": "t"}, top_k=1)
    assert seen["name"] == R.RERANK_MODEL
