"""Pairing assessments to the criteria they answer (fix 1, 2026-09-11)."""

from __future__ import annotations

from trialguard.agent.schema import align_assessments

TYPED = [
    {"text": "General: Age equal to or greater than 18.", "kind": "inclusion"},
    {"text": "Histologically confirmed NSCLC", "kind": "inclusion"},
    {"text": "Prior systemic chemotherapy", "kind": "exclusion"},
]


def _texts(slots):
    return [None if a is None else a["criterion"] for a in slots]


def test_an_exactly_echoed_criterion_is_paired():
    slots, left = align_assessments(
        [{"criterion": "Histologically confirmed NSCLC", "verdict": "met"}], TYPED
    )

    assert _texts(slots) == [None, "Histologically confirmed NSCLC", None]
    assert left == []


def test_a_criterion_answered_without_its_label_is_paired():
    """The measured case: 7.2% of TREC criteria were called unanswered when the
    model had answered them and dropped a "General:" style prefix."""
    slots, left = align_assessments(
        [{"criterion": "Age equal to or greater than 18", "verdict": "met"}], TYPED
    )

    assert slots[0] is not None
    assert left == []


def test_an_assessment_for_something_never_asked_is_left_over():
    slots, left = align_assessments(
        [{"criterion": "Must own a bicycle", "verdict": "met"}], TYPED
    )

    assert slots == [None, None, None]
    assert [a["criterion"] for a in left] == ["Must own a bicycle"]


def test_one_assessment_cannot_answer_two_criteria():
    """Two entries against one criterion is how a trial gets excluded on a
    disqualifier that does not exist, so pairing is one-to-one."""
    typed = [
        {"text": "Age 18 or older", "kind": "inclusion"},
        {"text": "Age 18 or older and under 75", "kind": "inclusion"},
    ]

    slots, left = align_assessments(
        [{"criterion": "Age 18 or older", "verdict": "met"}], typed
    )

    assert sum(1 for a in slots if a is not None) == 1
    assert left == []


def test_the_most_specific_criterion_gets_first_claim():
    """A short echo must not take the long criterion when a longer echo answers
    it exactly."""
    typed = [
        {"text": "Age 18 or older", "kind": "inclusion"},
        {"text": "Age 18 or older and under 75 with measurable disease",
         "kind": "inclusion"},
    ]

    slots, left = align_assessments(
        [
            {"criterion": "Age 18 or older", "verdict": "met"},
            {"criterion": "Age 18 or older and under 75 with measurable disease",
             "verdict": "not_met"},
        ],
        typed,
    )

    assert slots[0]["verdict"] == "met"
    assert slots[1]["verdict"] == "not_met"
    assert left == []


def test_exact_matching_wins_over_containment():
    typed = [
        {"text": "Measurable disease", "kind": "inclusion"},
        {"text": "Measurable disease by RECIST 1.1", "kind": "inclusion"},
    ]

    slots, _ = align_assessments(
        [
            {"criterion": "Measurable disease by RECIST 1.1", "verdict": "met"},
            {"criterion": "Measurable disease", "verdict": "not_met"},
        ],
        typed,
    )

    assert slots[0]["verdict"] == "not_met"
    assert slots[1]["verdict"] == "met"


def test_an_empty_echo_never_claims_a_criterion():
    slots, left = align_assessments([{"criterion": "", "verdict": "met"}], TYPED)

    assert slots == [None, None, None]
    assert len(left) == 1


def test_the_retry_does_not_re_ask_a_criterion_answered_under_another_wording():
    from trialguard.agent.graph import _missing_criteria

    missing = _missing_criteria(
        [{"criterion": "Age equal to or greater than 18", "verdict": "met"}], TYPED
    )

    assert [c["text"] for c in missing] == [
        "Histologically confirmed NSCLC",
        "Prior systemic chemotherapy",
    ]
