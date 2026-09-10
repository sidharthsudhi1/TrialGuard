"""P1: re-ask for criteria the analyst never answered (TG_RETRY_MISSING)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from trialguard.agent import graph as G

TRIAL = "Inclusion: Age 18 or older. Stage IV disease. Exclusion: Prior chemotherapy."
CRITERIA = [
    {"text": "Age 18 or older", "kind": "inclusion"},
    {"text": "Stage IV disease", "kind": "inclusion"},
    {"text": "Prior chemotherapy", "kind": "exclusion"},
]
NOTE = "62-year-old man, Stage IV disease, no chemotherapy."


@pytest.fixture(autouse=True)
def _fresh_graph():
    G._GRAPH = None
    yield
    G._GRAPH = None


def test_missing_criteria_are_identified_by_normalized_text():
    """Whitespace and casing drift is not a skipped criterion."""
    answered = [{"criterion": "age 18 or older "}, {"criterion": "Stage  IV disease"}]

    missing = G._missing_criteria(answered, CRITERIA)

    assert [c["text"] for c in missing] == ["Prior chemotherapy"]


def test_the_escape_hatch_restores_the_previous_behaviour(monkeypatch):
    """On by default, but this changes the retry prompt and therefore every
    retry cache key, so reproducing numbers committed before 2026-09-10 needs a
    way back -- the same role TG_KEYWORD_DECAY=0 plays for rankings."""
    monkeypatch.setenv("TG_RETRY_MISSING", "0")
    calls = []

    def short(note, nct_id, criteria, **kw):
        calls.append(note)
        return [{"criterion": "Age 18 or older", "verdict": "met", "quote": "62-year-old"}]

    with patch.object(G, "analyze_trial", side_effect=short):
        state = G.assess(NOTE, "NCT1", CRITERIA, TRIAL, max_retries=2)

    assert len(calls) == 1
    assert len(state["assessments"]) == 1


def test_a_short_answer_is_re_asked_by_default(monkeypatch):
    """A criterion with no assessment cannot fail grounding, so it never reached
    the retry edge. That is why the shortfall was invisible."""
    monkeypatch.delenv("TG_RETRY_MISSING", raising=False)
    calls = []

    def short_then_full(note, nct_id, criteria, **kw):
        calls.append(note)
        if len(calls) == 1:
            return [{"criterion": "Age 18 or older", "verdict": "met", "quote": "62-year-old"}]
        return [
            {"criterion": "Age 18 or older", "verdict": "met", "quote": "62-year-old"},
            {"criterion": "Stage IV disease", "verdict": "met", "quote": "Stage IV disease"},
            {"criterion": "Prior chemotherapy", "verdict": "not_met",
             "quote": "no chemotherapy"},
        ]

    with patch.object(G, "analyze_trial", side_effect=short_then_full):
        state = G.assess(NOTE, "NCT1", CRITERIA, TRIAL, max_retries=2)

    assert len(calls) == 2
    assert len(state["assessments"]) == 3
    # The re-ask names what was skipped, and says it was skipped rather than
    # telling the model its quote was not verbatim.
    assert "were not answered at all" in calls[1]
    assert "- Stage IV disease" in calls[1]
    assert "- Prior chemotherapy" in calls[1]


def test_the_retry_stops_once_everything_is_answered(monkeypatch):
    monkeypatch.setenv("TG_RETRY_MISSING", "1")
    calls = []

    def full(note, nct_id, criteria, **kw):
        calls.append(note)
        return [
            {"criterion": c["text"], "verdict": "cannot_determine", "quote": ""}
            for c in CRITERIA
        ]

    with patch.object(G, "analyze_trial", side_effect=full):
        G.assess(NOTE, "NCT1", CRITERIA, TRIAL, max_retries=2)

    assert len(calls) == 1


def test_the_retry_is_still_bounded_when_the_analyst_keeps_skipping(monkeypatch):
    """Never an unbounded loop, whatever the model does."""
    monkeypatch.setenv("TG_RETRY_MISSING", "1")
    calls = []

    def always_short(note, nct_id, criteria, **kw):
        calls.append(note)
        return [{"criterion": "Age 18 or older", "verdict": "met", "quote": "62-year-old"}]

    with patch.object(G, "analyze_trial", side_effect=always_short):
        state = G.assess(NOTE, "NCT1", CRITERIA, TRIAL, max_retries=2)

    assert len(calls) == 3  # first attempt plus max_retries
    assert state["retries"] == 2


def test_a_recovered_criterion_survives_the_partial_retry_merge(monkeypatch):
    """_merge_retry walks the prior list, so a criterion that was missing is not
    in it and would be dropped -- throwing away the recovery this exists for."""
    monkeypatch.setenv("TG_RETRY_MISSING", "1")
    monkeypatch.setenv("TG_RETRY_FAILED_ONLY", "1")

    def short_then_subset(note, nct_id, criteria, **kw):
        if "were not answered at all" not in note:
            return [{"criterion": "Age 18 or older", "verdict": "met", "quote": "62-year-old"}]
        return [
            {"criterion": "Stage IV disease", "verdict": "met", "quote": "Stage IV disease"},
            {"criterion": "Prior chemotherapy", "verdict": "not_met", "quote": "no chemotherapy"},
        ]

    with patch.object(G, "analyze_trial", side_effect=short_then_subset):
        state = G.assess(NOTE, "NCT1", CRITERIA, TRIAL, max_retries=2)

    got = {a["criterion"] for a in state["assessments"]}
    assert got == {"Age 18 or older", "Stage IV disease", "Prior chemotherapy"}


def test_a_criterion_the_model_invents_on_retry_is_not_added(monkeypatch):
    """Recovery is limited to criteria that really were missing."""
    monkeypatch.setenv("TG_RETRY_MISSING", "1")
    monkeypatch.setenv("TG_RETRY_FAILED_ONLY", "1")

    def short_then_invented(note, nct_id, criteria, **kw):
        if "were not answered at all" not in note:
            return [{"criterion": "Age 18 or older", "verdict": "met", "quote": "62-year-old"}]
        return [{"criterion": "Must own a bicycle", "verdict": "met", "quote": "62-year-old man"}]

    with patch.object(G, "analyze_trial", side_effect=short_then_invented):
        state = G.assess(NOTE, "NCT1", CRITERIA, TRIAL, max_retries=2)

    assert "Must own a bicycle" not in {a["criterion"] for a in state["assessments"]}
