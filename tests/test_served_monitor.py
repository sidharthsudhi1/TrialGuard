import json
from pathlib import Path

from trialguard.eval.served_monitor import check, compute


def _state(verdicts: list[str], retries: int = 0) -> dict:
    return {
        "retries": retries,
        "trial_verdict": "cannot_determine",
        "assessments": [{"verdict": v} for v in verdicts],
    }


def test_compute_rates():
    states = [
        _state(["met", "not_met", "cannot_determine", "unverifiable"], retries=1),
        _state(["met", "met"], retries=0),
    ]
    m = compute(states)
    assert m["n_trials"] == 2
    assert m["n_criteria"] == 6
    assert m["abstention_rate"] == round(2 / 6, 4)
    assert m["unverifiable_rate"] == round(1 / 6, 4)
    assert m["retry_rate"] == 0.5


def test_check_passes_inside_bands():
    m = compute([_state(["met", "cannot_determine"]) for _ in range(20)])
    outcome = check(m)
    assert outcome["checked"] is True
    assert outcome["passed"] is True


def test_check_diverges_on_abstain_everything(tmp_path):
    m = compute([_state(["cannot_determine"] * 5) for _ in range(20)])
    outcome = check(m)
    assert outcome["passed"] is False
    failed = [r["metric"] for r in outcome["results"] if not r["passed"]]
    assert "abstention_rate" in failed


def test_check_withholds_judgment_below_min_trials():
    m = compute([_state(["cannot_determine"] * 5)])
    outcome = check(m)
    assert outcome["checked"] is False
    assert outcome["passed"] is True


def test_bands_cover_committed_eval_anchors():
    # The bands must not flag the system the eval reports describe.
    bands = {b["metric"]: b for b in json.loads(
        Path("data/reports/served_baselines.json").read_text())["bands"]}
    for anchor in (0.5558, 0.6092, 0.593):
        b = bands["abstention_rate"]
        assert b["min"] <= anchor <= b["max"]
    for anchor in (0.0276, 0.0471):
        assert anchor <= bands["unverifiable_rate"]["max"]
