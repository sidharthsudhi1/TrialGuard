"""WS-6: what reads the corpus follows the refresh that writes it."""

from __future__ import annotations

import numpy as np

from trialguard.agent import analyst as A
from trialguard.retrieval import vector_cache as V

CRITERIA = [{"text": "Adults", "kind": "inclusion"}]


def test_the_analyst_key_ignores_criteria_by_default(monkeypatch):
    """Eval cohorts: committed keys must stay byte-identical."""
    monkeypatch.delenv("TG_CACHE_KEY_CRITERIA", raising=False)
    assert A._cache_key("note", "NCT1", CRITERIA) == A._cache_key("note", "NCT1")


def test_the_served_key_moves_when_the_criteria_do(monkeypatch):
    """F10: a revised trial keeps its nct_id; its cached answers must not survive."""
    monkeypatch.setenv("TG_CACHE_KEY_CRITERIA", "1")
    revised = [{"text": "Adults aged 21 or over", "kind": "inclusion"}]

    assert A._cache_key("note", "NCT1", CRITERIA) != A._cache_key("note", "NCT1", revised)
    assert A._cache_key("note", "NCT1", CRITERIA) == A._cache_key("note", "NCT1", CRITERIA)


def _resident(version: str) -> V.VectorCache:
    cache = V.VectorCache("ctgov_live")
    cache._ids = ["NCT1"]
    cache._matrix = np.full((1, 768), 1 / np.sqrt(768), dtype=np.float32)  # unit rows, as load() leaves them
    cache.version = version
    return cache


def test_a_new_corpus_version_triggers_a_reload(monkeypatch):
    monkeypatch.setattr("trialguard.config.settings.vector_cache_check_s", 0.0)
    monkeypatch.setattr(V, "corpus_version", lambda: "v2")
    cache = _resident("v1")
    reloads = []
    monkeypatch.setattr(cache, "load", lambda reload=False: reloads.append(reload))

    cache.maybe_reload().join()

    assert reloads == [True]


def test_the_same_version_does_not_reload(monkeypatch):
    monkeypatch.setattr("trialguard.config.settings.vector_cache_check_s", 0.0)
    monkeypatch.setattr(V, "corpus_version", lambda: "v1")
    cache = _resident("v1")
    reloads = []
    monkeypatch.setattr(cache, "load", lambda reload=False: reloads.append(reload))

    cache.maybe_reload().join()

    assert reloads == []


def test_the_check_is_throttled(monkeypatch):
    monkeypatch.setattr("trialguard.config.settings.vector_cache_check_s", 300.0)
    cache = _resident("v1")
    cache._last_check = __import__("time").time()

    assert cache.maybe_reload() is None


def test_time_spent_suspended_counts_toward_the_throttle(monkeypatch):
    """CLOCK_MONOTONIC stops while a Fly machine is suspended. A check made just
    before suspend must not hold the next one off after resume."""
    import time

    monkeypatch.setattr("trialguard.config.settings.vector_cache_check_s", 300.0)
    monkeypatch.setattr(V, "corpus_version", lambda: "v2")
    cache = _resident("v1")
    reloads = []
    monkeypatch.setattr(cache, "load", lambda reload=False: reloads.append(reload))
    cache._last_check = time.time() - 3600  # an hour ago on the wall clock
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)  # but no running time

    cache.maybe_reload().join()

    assert reloads == [True]


def test_a_health_poll_starts_the_version_check(monkeypatch):
    """Reloads hung off search traffic alone, so an idle deployment kept the
    pre-publish matrix until someone searched and was answered from it."""
    calls = []
    monkeypatch.setattr("trialguard.config.settings.retrieval_vector_cache", True)
    monkeypatch.setattr(V.VectorCache, "maybe_reload", lambda self: calls.append(self.source))

    V.kick()

    assert calls == ["ctgov_live"]


def test_searches_keep_the_old_matrix_while_a_reload_is_in_flight():
    cache = _resident("v1")
    cache._loading = True

    assert cache.load(reload=True) is True  # refused: one load at a time
    [(nct, score)] = cache.search(np.ones(768, dtype=np.float32), 1)
    assert nct == "NCT1" and abs(score - 1.0) < 1e-5
