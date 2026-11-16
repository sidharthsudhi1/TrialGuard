from trialguard.verify.entailment import (
    badge_reasons,
    context_conflict,
    numeric_conflict,
    uncovered_conjunct,
)
from trialguard.verify.grounding import ground_assessments


def test_negated_quote_conflicts_with_a_presence_claim():
    reason = context_conflict(
        "diarrhea",
        "The patient denies diarrhea and keeps fluids down.",
        "met",
    )
    assert reason == "context_negated"


def test_affirmed_quote_conflicts_with_an_absence_claim():
    reason = context_conflict(
        "shortness of breath",
        "complaining of shortness of breath",
        "not_met",
    )
    assert reason == "context_affirmed"


def test_history_criterion_does_not_treat_history_as_a_conflict():
    reason = context_conflict(
        "history of docetaxel",
        "history of docetaxel for metastatic disease",
        "met",
    )
    assert reason is None


def test_numeric_age_disagreement():
    reason = numeric_conflict(
        "Age >= 40 years",
        "not_met",
        "A 63-year-old man with chest pain.",
    )
    assert reason == "numeric_age"


def test_numeric_age_agreement_is_silent():
    assert numeric_conflict(
        "Age >= 40 years",
        "met",
        "A 63-year-old man with chest pain.",
    ) is None


def test_missing_number_is_not_a_flag():
    assert numeric_conflict("Age >= 40 years", "met", "A man with chest pain.") is None


def test_uncovered_conjunct_names_a_miss():
    reason = uncovered_conjunct(
        "HER2-positive disease and ECOG performance status 0",
        "HER2-positive disease",
        "met",
    )
    assert reason == "conjunct_uncovered"


def test_single_clause_is_not_a_conjunct_check():
    assert uncovered_conjunct("HER2-positive disease", "HER2-positive disease", "met") is None


def test_badge_is_off_unless_the_flag_is_set(monkeypatch):
    monkeypatch.delenv("TG_WEAK_ENTAILMENT", raising=False)
    note = "complaining of shortness of breath"
    out = ground_assessments(
        [{
            "criterion": "shortness of breath",
            "verdict": "not_met",
            "quote": note,
            "kind": "inclusion",
        }],
        note,
        patient_text=note,
    )
    assert "weak_entailment" not in out[0]
    monkeypatch.setenv("TG_WEAK_ENTAILMENT", "1")
    out = ground_assessments(
        [{
            "criterion": "shortness of breath",
            "verdict": "not_met",
            "quote": note,
            "kind": "inclusion",
        }],
        note,
        patient_text=note,
    )
    assert out[0]["verdict"] == "not_met"
    assert "context_affirmed" in out[0]["weak_entailment"]


def test_reasons_collect_without_changing_a_verdict():
    reasons = badge_reasons(
        "shortness of breath",
        "complaining of shortness of breath",
        "not_met",
        "complaining of shortness of breath",
    )
    assert reasons == ["context_affirmed"]
