import os

from trialguard.retrieval.pipeline import retrieve
from trialguard.retrieval.query_transform import _parse_keywords, partial_keywords


def test_partial_keywords_drop_an_unclosed_tail():
    raw = '{"keywords": ["breast cancer", "ecog 1", "her2'
    assert partial_keywords(raw, 12) == ["breast cancer", "ecog 1"]


def test_parse_keywords_ignores_prose_around_the_object():
    raw = (
        'Here is the list:\n```\n{"keywords": ["breast cancer", "ecog 1"]}\n```\n'
        "Note: omitted age."
    )
    assert _parse_keywords(raw, 12) == ["breast cancer", "ecog 1"]


def test_partial_keywords_match_the_full_parse():
    raw = '{"keywords": ["Breast Cancer", "ecog 1", "Breast Cancer"]}'
    assert partial_keywords(raw, 12) == _parse_keywords(raw, 12)


def test_streamed_keywords_rank_like_the_blocking_path(monkeypatch):
    calls = []

    def fake_dense(q, top_k=50, source=None):
        calls.append(("dense", q))
        return [("NCT1", 1.0)] if "breast" in q else [("NCT2", 1.0)]

    def fake_bm25(q, top_k=50, source=None):
        calls.append(("bm25", q))
        return [("NCT1", 0.5)] if "breast" in q else [("NCT3", 0.5)]

    def fake_gen(note, n_max=12, handler=None, on_keyword=None):
        kws = ["breast cancer", "ecog 1"]
        if on_keyword:
            for kw in kws:
                on_keyword(kw)
        return kws

    monkeypatch.setattr("trialguard.retrieval.pipeline.dense_search", fake_dense)
    monkeypatch.setattr("trialguard.retrieval.pipeline.bm25_search", fake_bm25)
    monkeypatch.setattr("trialguard.retrieval.pipeline._dense_batch", lambda *a, **k: None)
    monkeypatch.setattr(
        "trialguard.retrieval.query_transform.generate_keywords", fake_gen
    )
    monkeypatch.setenv("TG_KEYWORD_STREAM", "0")
    off, _ = retrieve("note", top_k=5, use_keywords=True)
    monkeypatch.setenv("TG_KEYWORD_STREAM", "1")
    on, lat = retrieve("note", top_k=5, use_keywords=True)
    assert [n for n, _ in on] == [n for n, _ in off]
    assert lat["keyword_first_ms"] is not None
    assert os.environ["TG_KEYWORD_STREAM"] == "1"


def test_a_partial_stream_is_discarded(monkeypatch):
    seen = []

    def fake_dense(q, top_k=50, source=None):
        seen.append(q)
        return [("NCT1", 1.0)]

    def fake_bm25(q, top_k=50, source=None):
        return [("NCT9", 0.2)]

    def fake_gen(note, n_max=12, handler=None, on_keyword=None):
        if on_keyword:
            on_keyword("dropped fragment")
        return ["breast cancer"]

    monkeypatch.setattr("trialguard.retrieval.pipeline.dense_search", fake_dense)
    monkeypatch.setattr("trialguard.retrieval.pipeline.bm25_search", fake_bm25)
    monkeypatch.setattr("trialguard.retrieval.pipeline._dense_batch", lambda *a, **k: None)
    monkeypatch.setattr(
        "trialguard.retrieval.query_transform.generate_keywords", fake_gen
    )
    monkeypatch.setenv("TG_KEYWORD_STREAM", "1")
    retrieve("note", top_k=5, use_keywords=True)
    assert seen == ["breast cancer"]


def test_provisional_callback_does_not_change_the_final_list(monkeypatch):
    def fake_dense(q, top_k=50, source=None):
        return [("NCT1", 1.0)] if q == "note" else [("NCT2", 0.4)]

    def fake_bm25(q, top_k=50, source=None):
        return [("NCT3", 0.2)]

    def fake_gen(note, n_max=12, handler=None, on_keyword=None):
        kws = ["breast cancer"]
        if on_keyword:
            on_keyword(kws[0])
        return kws

    monkeypatch.setattr("trialguard.retrieval.pipeline.dense_search", fake_dense)
    monkeypatch.setattr("trialguard.retrieval.pipeline.bm25_search", fake_bm25)
    monkeypatch.setattr("trialguard.retrieval.pipeline._dense_batch", lambda *a, **k: None)
    monkeypatch.setattr(
        "trialguard.retrieval.query_transform.generate_keywords", fake_gen
    )
    monkeypatch.setenv("TG_KEYWORD_STREAM", "1")
    plain, _ = retrieve("note", top_k=5, use_keywords=True)
    seen = {}

    def on_prov(hits):
        seen["hits"] = hits

    final, _ = retrieve("note", top_k=5, use_keywords=True, on_provisional=on_prov)
    assert [n for n, _ in final] == [n for n, _ in plain]
    assert seen["hits"]
