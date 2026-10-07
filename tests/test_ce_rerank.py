from unittest.mock import patch

from trialguard.config import settings
from trialguard.eval import r2_crossencoder as R2
from trialguard.retrieval import ce_rerank as C
from trialguard.retrieval import pipeline as P


def test_off_by_default_and_only_for_deep_keyword_queries(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_ce_model", "")
    assert not C.enabled(100, True)
    monkeypatch.setattr(settings, "retrieval_ce_model", "data/models/r2_ce")
    assert C.enabled(100, True)
    assert not C.enabled(25, True)
    assert not C.enabled(100, False)


def test_matches_the_measured_ce_rrf_deep_ordering():
    ids = [f"NCT{i}" for i in range(20)]
    scores = [((i * 7) % 11) / 10 for i in range(20)]  # includes ties
    measured = R2._orders({"candidates": ids, "served": ids}, scores)["ce_rrf_deep"]
    with patch.object(C, "score", return_value=scores):
        got = C.rerank(["kw"], [(n, 0.0) for n in ids], {n: n for n in ids})
    assert [n for n, _ in got] == measured


def test_trial_without_text_is_demoted_not_dropped():
    pool = [("A", 0.0), ("B", 0.0), ("C", 0.0)]
    with patch.object(C, "score", return_value=[0.1, 0.9]):
        got = C.rerank(["kw"], pool, {"B": "b", "C": "c"})
    assert [n for n, _ in got][-1] == "A" and len(got) == 3


def test_doc_text_is_what_training_read():
    t = {"title": "T", "inclusion_criteria": ["a", "b"], "exclusion_criteria": None}
    assert R2._doc(t) == C.doc_text(t) == "T. Inclusion: a b Exclusion: "


def test_retrieve_reranks_deep_pool_and_slices(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_ce_model", "x")
    monkeypatch.setattr(settings, "retrieval_demographic_filter", False)
    dense = [(f"NCT{i}", 1.0 - i / 300) for i in range(150)]
    rows = {n: {"title": n} for n, _ in dense}
    with (
        patch.object(P, "dense_search", return_value=dense),
        patch.object(P, "bm25_search", return_value=[]),
        patch("trialguard.retrieval.query_transform.generate_keywords", return_value=["kw"]),
        patch("trialguard.db.queries.get_trials", return_value=rows),
        # Reverse the deep order: the last candidate scores highest.
        patch.object(C, "score", side_effect=lambda q, d: [float(i) for i in range(len(d))]),
    ):
        hits, lat = P.retrieve("note", top_k=100, source="ctgov_live", use_keywords=True)
    assert len(hits) == 100
    assert "ce_ms" in lat
    assert [n for n, _ in hits] != [n for n, _ in dense[:100]]
    assert "NCT149" in {n for n, _ in hits}
