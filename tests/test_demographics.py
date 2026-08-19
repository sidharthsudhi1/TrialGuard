"""Demographic gating — must only ever act on explicit evidence."""

from __future__ import annotations

import pytest

from trialguard.retrieval.demographics import (
    exclusion_reason,
    filter_candidates,
    parse_age,
    parse_patient,
    parse_sex,
)


@pytest.mark.parametrize(
    "note,expected",
    [
        ("A 58-year-old African-American woman presents to the ER", 58),
        ("An 8-year-old male presents in March", 8),
        ("64-year-old obese female with diabetes", 64),
        ("A 2-year-old boy is brought to the emergency department", 2),
        ("Patient is 47 yo with chest pain", 47),
        ("A 6-month-old infant with fever", 0.5),
        ("Patient with chest pain, no age given", None),
    ],
)
def test_age_parsing(note, expected):
    got = parse_age(note)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, abs=0.01)


def test_sex_takes_the_first_mention():
    """'her father' must not make a woman ambiguous."""
    assert parse_sex("A 58-year-old woman whose father had a myocardial infarction") == "female"
    assert parse_sex("An 8-year-old male whose mother reports fever") == "male"
    assert parse_sex("Patient presents with cough") is None


def test_explicit_contradiction_excludes():
    child = {"age": 8, "sex": "male"}
    assert exclusion_reason({"min_age": "18 Years"}, child)
    assert exclusion_reason({"sex": "FEMALE"}, child)
    assert exclusion_reason({"max_age": "5 Years"}, child)


def test_absence_never_excludes():
    """A null is not a negative — CT.gov populates these fields inconsistently."""
    patient = {"age": 58, "sex": "female"}
    assert exclusion_reason({}, patient) is None
    assert exclusion_reason({"min_age": None, "max_age": None, "sex": None}, patient) is None
    assert exclusion_reason({"min_age": "N/A", "sex": "ALL"}, patient) is None
    # And an unparseable note gates nothing at all.
    assert exclusion_reason({"min_age": "18 Years", "sex": "MALE"}, {"age": None, "sex": None}) is None


def test_month_and_year_limits_are_comparable():
    infant = {"age": 0.5, "sex": None}
    assert exclusion_reason({"min_age": "6 Months"}, infant) is None
    assert exclusion_reason({"min_age": "18 Years"}, infant)
    assert exclusion_reason({"min_age": "12 Months"}, infant)


def test_filter_keeps_candidates_with_no_metadata():
    """The filter may only act on evidence it has."""
    hits = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
    trials = {"A": {"min_age": "18 Years"}, "B": {}}  # C absent entirely
    kept, dropped = filter_candidates(hits, trials, {"age": 8, "sex": None})
    assert [n for n, _ in kept] == ["B", "C"]
    assert "A" in dropped and "below min" in dropped["A"]


def test_filter_is_inert_without_patient_demographics():
    hits = [("A", 0.9)]
    trials = {"A": {"min_age": "18 Years", "sex": "MALE"}}
    kept, dropped = filter_candidates(hits, trials, parse_patient("no demographics here"))
    assert kept == hits and dropped == {}


def test_filter_preserves_ranking_order():
    hits = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
    trials = {n: {} for n in "ABC"}
    kept, _ = filter_candidates(hits, trials, {"age": 40, "sex": "female"})
    assert kept == hits
