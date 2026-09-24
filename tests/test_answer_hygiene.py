"""Duplicate answers, retries that lose grounding, empty cache rows, no source."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from trialguard.agent import analyst
from trialguard.agent import graph as G

CRIT = [{"text": "Stage IV disease", "kind": "inclusion"}]
NOTE = "62-year-old man, Stage IV disease."


@pytest.fixture(autouse=True)
def _fresh_graph():
    G._GRAPH = None
    yield
    G._GRAPH = None


def _a(verdict, quote="", grounded=False, criterion="Stage IV disease"):
    return {"criterion": criterion, "verdict": verdict, "quote": quote, "grounded": grounded}


# --- R8: one answer per criterion ---


def test_contradictory_grounded_answers_collapse_to_a_conflict():
    """Picking either side would be a verdict nothing supports over the other."""
    out = G._dedup([_a("met", "stage iv", True), _a("not_met", "stage iv", True)])

    assert len(out) == 1
    assert out[0]["verdict"] == "cannot_determine"
    assert out[0]["conflict"] is True


def test_a_grounded_duplicate_is_preferred_over_an_abstention():
    out = G._dedup([_a("cannot_determine"), _a("met", "stage iv", True)])

    assert [a["verdict"] for a in out] == ["met"]


def test_a_duplicate_not_met_no_longer_excludes_the_trial(monkeypatch):
    monkeypatch.delenv("TG_DEDUP_ANSWERS", raising=False)

    def twice(note, nct_id, criteria, **kw):
        return [
            {"criterion": "Stage IV disease", "verdict": "met", "quote": "Stage IV disease"},
            {"criterion": "Stage IV disease", "verdict": "not_met", "quote": "62-year-old man"},
        ]

    with patch.object(G, "analyze_trial", side_effect=twice):
        state = G.assess(NOTE, "NCT1", CRIT, "Stage IV disease", max_retries=0)

    assert state["trial_verdict"] != "excluded"
    assert len(state["assessments"]) == 1


def test_dedup_escape_hatch(monkeypatch):
    monkeypatch.setenv("TG_DEDUP_ANSWERS", "0")

    def twice(note, nct_id, criteria, **kw):
        return [
            {"criterion": "Stage IV disease", "verdict": "met", "quote": "Stage IV disease"},
            {"criterion": "Stage IV disease", "verdict": "not_met", "quote": "62-year-old man"},
        ]

    with patch.object(G, "analyze_trial", side_effect=twice):
        state = G.assess(NOTE, "NCT1", CRIT, "Stage IV disease", max_retries=0)

    assert state["trial_verdict"] == "excluded"


# --- R7: a retry cannot undo grounding ---


def test_backfill_keeps_a_grounded_answer_over_an_ungrounded_retry(monkeypatch):
    monkeypatch.delenv("TG_RETRY_KEEP_GROUNDED", raising=False)
    prior = [_a("met", "stage iv", True)]
    retried = [{**_a("unverifiable", "made up"), "grounding_failure": True}]

    out = G._backfill(retried, prior, CRIT)

    assert out == prior


def test_backfill_still_takes_a_grounded_retry(monkeypatch):
    monkeypatch.delenv("TG_RETRY_KEEP_GROUNDED", raising=False)
    prior = [_a("met", "stage iv", True)]
    retried = [_a("not_met", "no stage iv", True)]

    assert G._backfill(retried, prior, CRIT) == retried


def test_keep_grounded_escape_hatch(monkeypatch):
    monkeypatch.setenv("TG_RETRY_KEEP_GROUNDED", "0")
    retried = [{**_a("unverifiable", "made up"), "grounding_failure": True}]

    assert G._backfill(retried, [_a("met", "stage iv", True)], CRIT) == retried


# --- R5: an empty response is not an answer ---


def _llm_returning(payload):
    m = MagicMock()
    m.invoke.return_value.content = json.dumps({"assessments": payload})
    return m


def test_an_empty_response_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(analyst, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("TG_SKIP_ANALYST_CACHE_WRITE", raising=False)
    with (
        patch("trialguard.db.cache.cache_get", return_value=None),
        patch("trialguard.db.cache.cache_put") as cp,
        patch("trialguard.llm.cost.active_ledger"),
        patch.object(analyst, "_llm", return_value=_llm_returning([])),
    ):
        assert analyst.analyze_trial(NOTE, "NCT1", CRIT) == []

    assert not list(tmp_path.glob("*.json"))
    cp.assert_not_called()


def test_an_empty_cache_entry_is_a_miss_outside_cached_only(tmp_path, monkeypatch):
    monkeypatch.setattr(analyst, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("TG_CACHED_ONLY", raising=False)
    key = analyst.retry_cache_key(NOTE, "NCT1", CRIT, "", analyst.prompt_version())
    (tmp_path / f"{key}.json").write_text("[]")
    fresh = [{"criterion": "Stage IV disease", "verdict": "met", "quote": "Stage IV disease"}]
    with (
        patch("trialguard.db.cache.cache_get", return_value=None),
        patch("trialguard.db.cache.cache_put"),
        patch("trialguard.llm.cost.active_ledger"),
        patch.object(analyst, "_llm", return_value=_llm_returning(fresh)) as llm,
    ):
        got = analyst.analyze_trial(NOTE, "NCT1", CRIT)

    llm.assert_called()
    assert got[0]["verdict"] == "met"


def test_cached_only_still_honours_an_empty_entry(tmp_path, monkeypatch):
    """A cached-only run must never make a fresh call."""
    monkeypatch.setattr(analyst, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("TG_CACHED_ONLY", "1")
    key = analyst.retry_cache_key(NOTE, "NCT1", CRIT, "", analyst.prompt_version())
    (tmp_path / f"{key}.json").write_text("[]")
    with patch.object(analyst, "_llm") as llm:
        assert analyst.analyze_trial(NOTE, "NCT1", CRIT) == []
    llm.assert_not_called()


# --- R10: no eligibility text ---


def test_a_trial_without_eligibility_text_grounds_against_its_criteria():
    """Otherwise the patient note is the only source and a trial-text quote
    reads as note-sourced or fails."""

    def quotes_criterion(note, nct_id, criteria, **kw):
        return [{"criterion": "Stage IV disease", "verdict": "met",
                 "quote": "Stage IV disease"}]

    with patch.object(G, "analyze_trial", side_effect=quotes_criterion):
        state = G.assess("62-year-old man.", "NCT1", CRIT, "", max_retries=0)

    assert state["assessments"][0]["grounded_in"] == "trial"
