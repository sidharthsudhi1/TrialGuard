"""A caller that already holds the keyword list must not call the extractor."""

import numpy as np


class _BM25:
    def get_scores(self, _tokens):
        return np.array([1.0, 0.2])


def test_passed_queries_skip_extraction(monkeypatch):
    from trialguard.eval.file_index import FileIndex
    from trialguard.ingestion import embed as embed_mod

    idx = FileIndex("sigir")
    idx._loaded = True
    idx._nct_ids = ["A", "B"]
    idx._chunk_owners = ["A", "B"]
    idx._matrix = np.zeros((2, 768), dtype=np.float32)
    idx._matrix[0, 0] = 1.0
    idx._bm25 = _BM25()

    def boom(_note):
        raise AssertionError("keyword model called")

    monkeypatch.setattr("trialguard.retrieval.query_transform.generate_keywords", boom)
    monkeypatch.setattr("trialguard.config.settings.retrieval_ce_model", "")

    def fake_embed(texts, is_query=False):
        assert texts == ["her2 positive"]
        assert is_query
        return np.zeros((1, 768), dtype=np.float32)

    monkeypatch.setattr(embed_mod, "embed_matrix", fake_embed)
    hits = idx.search("the note", top_k=1, queries=["her2 positive"])
    assert [nct for nct, _ in hits] == ["A"]
