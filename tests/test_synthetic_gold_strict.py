import pytest

from trialguard.eval.synthetic_gold import holds_strict, parse_strict, score_rows


@pytest.mark.parametrize(
    "line,attr,value,expected",
    [
        ("Bilirubin ≤ 1.5 mg/dL", "bilirubin", 1.47, True),
        ("Total bilirubin: ≤ 1.5 mg/dL", "bilirubin", 1.64, False),
        ("Haemoglobin ≥90 g/L.", "hemoglobin", 8.6, False),
        ("Platelets < 50,000/ul", "platelets", 49, True),
        ("Platelet count ≥ 80 x 109/L", "platelets", 80, True),
        ("Absolute neutrophil count (ANC) =< 1.0 x 10^9/L", "anc", 1000, True),
        ("Absolute neutrophil count (ANC) of >1.5 x 10^9/L", "anc", 1500, False),
        ("Age 3 years or older.", "age", 3, True),
        ("Age: 18 to75 years old;", "age", 75, True),
        ("Under 18 years,", "age", 18, False),
        ("ECOG performance status (PS) of 0, 1, or 2", "ecog", 3, False),
        ("ECOG score 0 or 1.", "ecog", 1, True),
    ],
)
def test_strict_parse_and_truth(line, attr, value, expected):
    spec = parse_strict(line)
    assert spec is not None and spec["attr"] == attr
    assert holds_strict(spec, float(value)) is expected


@pytest.mark.parametrize(
    "line",
    [
        "creatinine clearance ≥ 40 mL/min",
        "Bilirubin ≤ 1.5 x ULN",
        "Hemoglobin A1c (HbA1c) between ≥7",
        "Hemoglobin level ≥ 8 mg/dl within 4 weeks before enrollment",
        "Creatinine > 177 μmol/L (2 mg/dL) at screening",
        "Preterm infants with gestational age 24 to 32 weeks",
        "Platelets 100",
    ],
)
def test_strict_rejects_unsafe_lines(line):
    assert parse_strict(line) is None


def test_score_rows_counts_abstention_as_wrong():
    gold = [
        {"id": "a", "holds": True, "spec": {"attr": "age"}},
        {"id": "b", "holds": False, "spec": {"attr": "age"}},
        {"id": "c", "holds": True, "spec": {"attr": "anc"}},
        {"id": "d", "holds": False, "spec": {"attr": "anc"}},
    ]
    calls = {
        "a": {"verdict": "met"},
        "b": {"verdict": "met"},
        "c": {"verdict": "not_met"},
        "d": {"verdict": "unverifiable"},
    }
    out = score_rows(gold, calls)["overall"]
    assert out["accuracy"]["point"] == 0.25
    assert out["coverage"]["point"] == 0.75
    assert out["false_exclusion"]["point"] == 0.5
    assert out["false_inclusion"]["point"] == 0.5
