"""Per-keyword list depth fed to RRF (R1b): default 200, TG_LIST_POOL=50 reproduces."""

from __future__ import annotations

from unittest.mock import patch

from trialguard.retrieval import pipeline as P
from trialguard.retrieval.fusion import list_pool


def test_default_is_200_and_env_restores_50(monkeypatch):
    monkeypatch.delenv("TG_LIST_POOL", raising=False)
    assert list_pool() == 200
    monkeypatch.setenv("TG_LIST_POOL", "50")
    assert list_pool() == 50


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
        P.retrieve("note", top_k=5, **kw)
    return seen


def test_served_retrieve_uses_the_flag(monkeypatch):
    assert _pools_seen(monkeypatch) == {"dense": 200, "lex": 200}
    assert _pools_seen(monkeypatch, env="50") == {"dense": 50, "lex": 50}


def test_explicit_pool_still_wins(monkeypatch):
    assert _pools_seen(monkeypatch, dense_pool=30, bm25_pool=40) == {"dense": 30, "lex": 40}
