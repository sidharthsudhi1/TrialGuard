"""Per-keyword list depth fed to RRF (R1b): 200 for top_k >= 100, else 50."""

from __future__ import annotations

from unittest.mock import patch

from trialguard.retrieval import pipeline as P
from trialguard.retrieval.fusion import list_pool


def test_depth_follows_the_request_and_env_pins_it(monkeypatch):
    monkeypatch.delenv("TG_LIST_POOL", raising=False)
    assert [list_pool(k) for k in (5, 25, 99, 100, 500)] == [50, 50, 50, 200, 200]
    monkeypatch.setenv("TG_LIST_POOL", "50")
    assert list_pool(500) == 50


def _pools_seen(monkeypatch, env=None, **kw):
    seen = {}
    if env:
        monkeypatch.setenv("TG_LIST_POOL", env)
    else:
        monkeypatch.delenv("TG_LIST_POOL", raising=False)

    def dense(q, top_k, source=None):
        seen["dense"] = top_k
        return [("NCT1", 0.9)]

    def lex(q, top_k, source=None):
        seen["lex"] = top_k
        return [("NCT1", 0.5)]

    with patch.object(P, "dense_search", dense), patch.object(P, "bm25_search", lex):
        P.retrieve("note", **{"top_k": 5, **kw})
    return seen


def test_demo_depth_keeps_shallow_lists_and_deep_requests_go_deep(monkeypatch):
    assert _pools_seen(monkeypatch) == {"dense": 50, "lex": 50}
    assert _pools_seen(monkeypatch, top_k=100) == {"dense": 200, "lex": 200}
    assert _pools_seen(monkeypatch, env="50", top_k=100) == {"dense": 50, "lex": 50}


def test_explicit_pool_still_wins(monkeypatch):
    assert _pools_seen(monkeypatch, dense_pool=30, bm25_pool=40) == {"dense": 30, "lex": 40}
