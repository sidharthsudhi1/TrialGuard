import json
from pathlib import Path

import pytest

from trialguard.eval.regression_gate import FIXTURE, evaluate, stress_test
from trialguard.verify.grounding import is_grounded

CASES = json.loads(FIXTURE.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["quote"][:30])
def test_golden_cases_ground_as_declared(case):
    # Offline lock on the grounding logic itself: every fixture quote grounds or is
    # rejected exactly as the golden file declares. Independent of Groq/cache.
    assert is_grounded(case["quote"], case["source"]) is case["grounded"]


def test_stress_catch_rate_is_100_percent():
    s = stress_test(CASES)
    assert s["n_corrupted"] > 0
    assert s["verifier_catch_rate"] == 1.0
    assert s["verifier_false_rejection_rate"] == 0.0


def test_gate_passes_on_committed_reports():
    assert evaluate()["passed"] is True


def test_gate_bites_on_regressed_report(tmp_path):
    # Prove the gate actually fails red: point it at a report whose verified
    # unsupported rate blew past the ceiling, keep everything else passing.
    good = json.loads(Path("data/reports/phase4_agent_sigir.json").read_text())
    good["verified"]["unsupported_verdict_rate"] = 0.5
    bad_report = tmp_path / "regressed.json"
    bad_report.write_text(json.dumps(good))

    baselines = json.loads(Path("data/reports/baselines.json").read_text())
    baselines["report"] = str(bad_report)
    bad_baselines = tmp_path / "baselines.json"
    bad_baselines.write_text(json.dumps(baselines))

    outcome = evaluate(baselines_path=bad_baselines)
    assert outcome["passed"] is False
    failed = [r["metric"] for r in outcome["results"] if not r["passed"]]
    assert "verified.unsupported_verdict_rate" in failed
