import json

import pytest

from trialguard.retrieval import listwise


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(listwise, "CACHE_DIR", tmp_path / "listwise")


def test_parse_ranked_drops_out_of_range_and_duplicates():
    raw = '```json\n{"ranked": [2, 0, 99, 2, -1]}\n```'
    assert listwise._parse_ranked(raw, n_candidates=3) == [2, 0]


def test_summarize_flattens_and_truncates():
    out = listwise._summarize("a\n\n  b   c", max_chars=4)
    assert out == "a b "


def test_cache_key_is_pool_specific():
    a = listwise._cache_key("note", ["NCT1", "NCT2"], "m")
    b = listwise._cache_key("note", ["NCT1", "NCT3"], "m")
    c = listwise._cache_key("note", ["NCT1", "NCT2"], "other-model")
    assert a != b and a != c


def test_rerank_falls_back_to_retrieval_order_on_failure(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("provider down")

    monkeypatch.setattr(listwise, "_rank_batch", _boom)
    candidates = [("NCT1", 0.9), ("NCT2", 0.8), ("NCT3", 0.7)]
    out = listwise.listwise_rerank("note", candidates, {}, top_k=2)
    assert out == candidates[:2]


def test_rerank_interleaves_batches_by_within_batch_rank(monkeypatch):
    # Batch 2's best must outrank batch 1's second, or a deep batch is starved
    # purely by where retrieval happened to place it.
    def _fake(note, batch, handler=None):
        return [n for n, _ in batch]

    monkeypatch.setattr(listwise, "_rank_batch", _fake)
    candidates = [(f"NCT{i}", 1.0) for i in range(4)]
    out = listwise.listwise_rerank("note", candidates, {}, top_k=4, batch_size=2)
    assert [n for n, _ in out] == ["NCT0", "NCT2", "NCT1", "NCT3"]


def test_rerank_reads_cache_without_calling_the_model(monkeypatch, tmp_path):
    candidates = [("NCT1", 0.9), ("NCT2", 0.8)]
    from trialguard.llm.provider import active_model

    listwise.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = listwise._cache_key("note", ["NCT1", "NCT2"], active_model())
    (listwise.CACHE_DIR / f"{key}.json").write_text(json.dumps(["NCT2", "NCT1"]))

    def _boom(*a, **kw):
        raise AssertionError("cache hit must not reach the model")

    monkeypatch.setattr(listwise, "_rank_batch", _boom)
    out = listwise.listwise_rerank("note", candidates, {}, top_k=2)
    assert [n for n, _ in out] == ["NCT2", "NCT1"]


def test_empty_candidates_short_circuits():
    assert listwise.listwise_rerank("note", [], {}) == []
