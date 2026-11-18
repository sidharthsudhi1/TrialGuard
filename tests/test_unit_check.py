import pytest

from trialguard.verify.unit_check import criterion_clauses, numeric_conflict, quote_values


@pytest.mark.parametrize(
    "criterion,verdict,quote,flag",
    [
        ("Absolute Neutrophil Count ≥ 1.5x10^9/L", "not_met", "ANC 1594/uL", True),
        ("Initial haemoglobin concentration <90 g/L", "met", "Hb is 9.0 g/dL", True),
        ("Platelets ≥ 100,000/mm3", "met", "platelets of 87,000/mm3", True),
        ("Age 18-75 years", "not_met", "72-year-old", True),
        ("Age 18-75 years", "met", "72-year-old", False),
        ("ECOG 0 or 1", "met", "ECOG performance status of 2", True),
        ("ANC ≥1500/mm3, platelets ≥ 100,000/mm3", "not_met", "platelets 150,000/mm3", False),
        ("ANC ≥1500/mm3, platelets ≥ 100,000/mm3", "met", "platelets 80,000/mm3", True),
        ("Platelets ≥ 100,000/mm3", "cannot_determine", "platelets 87,000/mm3", False),
        (
            "Creatinine ≤ 1.5 mg/dL",
            "met",
            "creatinine 1.2 mg/dL, later 2.0 mg/dL creatinine",
            False,
        ),
        ("Creatinine ≤ 1.5 mg/dL", "met", "creatinine 1.2 mg/dL then creatinine 2.0 mg/dL", False),
    ],
)
def test_numeric_conflict(criterion, verdict, quote, flag):
    assert bool(numeric_conflict(criterion, verdict, quote)) is flag


@pytest.mark.parametrize(
    "criterion",
    [
        "Creatinine clearance ≥ 60 mL/min",
        "Bilirubin ≤ 1.5 x ULN",
        "Hemoglobin A1c ≥ 7 g/dL",
        "age at diagnosis < 40 years",
        "ANC ≥ 1500/mm3 unless due to marrow involvement",
    ],
)
def test_out_of_scope(criterion):
    assert criterion_clauses(criterion) is None


def test_quote_needs_a_unit_for_labs():
    assert quote_values("platelets 87") == {}
    assert quote_values("Hb 85 g/L") == {"hemoglobin": 8.5}
