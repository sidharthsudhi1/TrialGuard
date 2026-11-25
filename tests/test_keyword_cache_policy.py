"""Keywords derived from a free-text note are not persisted."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from trialguard.retrieval import query_transform as qt


class _Ledger:
    def check(self, *a, **k):
        pass

    def record(self, *a, **k):
        pass


class _LLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        return SimpleNamespace(content='{"keywords": ["breast cancer", "ECOG 1"]}')


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    llm = _LLM()
    monkeypatch.setattr(qt, "CACHE_DIR", tmp_path)
    monkeypatch.setattr("trialguard.llm.provider.get_chat_model", lambda *_: llm)
    monkeypatch.setattr("trialguard.llm.provider.extract_usage", lambda r: {})
    monkeypatch.setattr("trialguard.llm.cost.active_ledger", lambda: _Ledger())
    monkeypatch.setattr("trialguard.tracing.trace_config", lambda *a, **k: {})
    return llm


def test_free_text_keywords_are_not_written_anywhere(fake_llm, tmp_path):
    with (
        patch("trialguard.db.cache.cache_get", return_value=None),
        patch("trialguard.db.cache.cache_put") as put,
    ):
        kws = qt.generate_keywords("free text note", persist=False)
        again = qt.generate_keywords("free text note", persist=False)

    assert kws == again == ["breast cancer", "ecog 1"]
    put.assert_not_called()
    assert list(tmp_path.iterdir()) == []
    # Nothing was stored, so the second call paid for its own extraction.
    assert fake_llm.calls == 2


def test_persisted_keywords_are_written_whole(fake_llm, tmp_path):
    with (
        patch("trialguard.db.cache.cache_get", return_value=None),
        patch("trialguard.db.cache.cache_put") as put,
    ):
        qt.generate_keywords("preset note")
        qt.generate_keywords("preset note")

    put.assert_called_once()
    assert [p.suffix for p in tmp_path.iterdir()] == [".json"]
    assert fake_llm.calls == 1
