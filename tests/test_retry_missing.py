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


def test_a_missing_only_retry_still_re_asks_the_whole_list(monkeypatch):
    """Narrowing to just the skipped criteria was measured and is worse: 144 of
    1,761 TREC criteria left unanswered against 71, because a narrowed ask has
    to be merged back by text match and what the merge cannot place is lost."""
    monkeypatch.delenv("TG_RETRY_MISSING", raising=False)
    monkeypatch.delenv("TG_RETRY_FAILED_ONLY", raising=False)
    asked = []

    def short_then_full(note, nct_id, criteria, **kw):
        asked.append([c["text"] for c in criteria])
        if len(asked) == 1:
            return [{"criterion": "Age 18 or older", "verdict": "met", "quote": "62-year-old"}]
        return [
            {"criterion": c["text"], "verdict": "cannot_determine", "quote": ""}
            for c in CRITERIA
        ]

    with patch.object(G, "analyze_trial", side_effect=short_then_full):
        G.assess(NOTE, "NCT1", CRITERIA, TRIAL, max_retries=2)

    assert asked[1] == [c["text"] for c in CRITERIA]


def test_a_grounding_failure_still_re_asks_the_whole_list(monkeypatch):
    """L4 measured the narrowed failed-retry and rejected it. Narrowing the
    omission case must not quietly adopt it for the failure case too."""
    monkeypatch.delenv("TG_RETRY_MISSING", raising=False)
    monkeypatch.delenv("TG_RETRY_FAILED_ONLY", raising=False)
    asked = []

    def ungrounded_then_fixed(note, nct_id, criteria, **kw):
        asked.append([c["text"] for c in criteria])
        quote = "62-year-old" if len(asked) > 1 else "not in the source at all"
        return [
            {"criterion": "Age 18 or older", "verdict": "met", "quote": quote},
            {"criterion": "Stage IV disease", "verdict": "met", "quote": "Stage IV disease"},
            {"criterion": "Prior chemotherapy", "verdict": "not_met", "quote": "no chemotherapy"},
        ]

    with patch.object(G, "analyze_trial", side_effect=ungrounded_then_fixed):
        G.assess(NOTE, "NCT1", CRITERIA, TRIAL, max_retries=2)

    assert len(asked) == 2
    assert asked[1] == [c["text"] for c in CRITERIA]


def test_an_omission_only_retry_does_not_claim_a_quote_failed(monkeypatch):
    """The prompt is built from parts, so a retry with nothing ungrounded must
    not open with a verbatim-quote complaint followed by an empty list."""
    monkeypatch.delenv("TG_RETRY_MISSING", raising=False)
    notes = []

    def short(note, nct_id, criteria, **kw):
        notes.append(note)
        return [{"criterion": "Age 18 or older", "verdict": "met", "quote": "62-year-old"}]

    with patch.object(G, "analyze_trial", side_effect=short):
        G.assess(NOTE, "NCT1", CRITERIA, TRIAL, max_retries=1)

    assert "need a verbatim quote" not in notes[1]
    assert "were not answered at all" in notes[1]


def test_a_retry_can_never_lose_a_criterion_attempt_one_answered(monkeypatch):
    """The full-list retry replaced attempt one wholesale, so a retry that came
    back shorter made coverage worse than not retrying at all. Measured at 71,
    144 and 269 unanswered across three runs of one configuration, against 229
    with no retry: that spread was output length varying, not the prompt."""
    monkeypatch.delenv("TG_RETRY_MISSING", raising=False)
    monkeypatch.delenv("TG_RETRY_FAILED_ONLY", raising=False)
    calls = []

    def two_then_one(note, nct_id, criteria, **kw):
        calls.append(note)
        if len(calls) == 1:
            return [
                {"criterion": "Age 18 or older", "verdict": "met", "quote": "62-year-old"},
                {"criterion": "Stage IV disease", "verdict": "met", "quote": "Stage IV disease"},
            ]
        # A shorter retry. Without the backfill this would drop "Stage IV
        # disease" entirely and leave the trial worse off than before.
        return [{"criterion": "Prior chemotherapy", "verdict": "not_met",
                 "quote": "no chemotherapy"}]

    with patch.object(G, "analyze_trial", side_effect=two_then_one):
        state = G.assess(NOTE, "NCT1", CRITERIA, TRIAL, max_retries=1)

    assert {a["criterion"] for a in state["assessments"]} == {
        "Age 18 or older", "Stage IV disease", "Prior chemotherapy"
    }


def test_the_retry_answer_wins_where_both_attempts_answered(monkeypatch):
    """Backfill fills gaps; it must not resurrect a verdict the retry replaced."""
    monkeypatch.delenv("TG_RETRY_MISSING", raising=False)
    calls = []

    def ungrounded_then_grounded(note, nct_id, criteria, **kw):
        calls.append(note)
        quote = "62-year-old" if len(calls) > 1 else "nowhere in any source"
        return [{"criterion": "Age 18 or older", "verdict": "met", "quote": quote}]

    with patch.object(G, "analyze_trial", side_effect=ungrounded_then_grounded):
        state = G.assess(NOTE, "NCT1", CRITERIA, TRIAL, max_retries=1)

    age = [a for a in state["assessments"] if a["criterion"] == "Age 18 or older"]
    assert len(age) == 1
    assert age[0]["verdict"] == "met"
    assert age[0]["grounded"] is True


def test_backfill_does_not_duplicate_on_casing_drift():
    from trialguard.agent.graph import _backfill

    out = _backfill(
        [{"criterion": "Age 18 or Older", "verdict": "met"}],
        [{"criterion": "age 18 or older", "verdict": "cannot_determine"}],
        [{"text": "Age 18 or older", "kind": "inclusion"}],
    )

    assert len(out) == 1
    assert out[0]["verdict"] == "met"


def test_a_rephrased_criterion_is_not_answered_twice(monkeypatch):
    """Deduplicating on the model's echoed text is not enough: a retry that
    rephrases a criterion does not match attempt one, and the criterion ends up
    answered twice. Two entries with opposing verdicts flip a trial to excluded
    on a disqualifier that does not exist."""
    monkeypatch.delenv("TG_RETRY_MISSING", raising=False)
    calls = []

    def rephrase(note, nct_id, criteria, **kw):
        calls.append(note)
        if len(calls) == 1:
            return [{"criterion": "Prior chemotherapy", "verdict": "not_met",
                     "quote": "no chemotherapy"}]
        # Same criterion, different wording, opposite verdict.
        return [{"criterion": "Prior  CHEMOTHERAPY.", "verdict": "met",
                 "quote": "no chemotherapy"}]

    with patch.object(G, "analyze_trial", side_effect=rephrase):
        state = G.assess(NOTE, "NCT1", CRITERIA, TRIAL, max_retries=1)

    chemo = [a for a in state["assessments"]
             if "chemo" in a["criterion"].lower()]
    assert len(chemo) == 1


def test_backfill_never_returns_more_entries_than_criteria_asked():
    """criterion_unanswered is asked minus answered, so a duplicate made it
    negative -- which is how this was caught."""
    from trialguard.agent.graph import _backfill

    out = _backfill(
        [{"criterion": "Age  18 or older", "verdict": "met"}],
        [{"criterion": "age 18 or older", "verdict": "cannot_determine"},
         {"criterion": "Stage IV disease", "verdict": "met"}],
        CRITERIA,
    )

    assert len(out) <= len(CRITERIA)
    assert len(out) == 2
