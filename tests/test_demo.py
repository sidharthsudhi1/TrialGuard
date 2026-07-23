from unittest.mock import patch

from trialguard import demo
from trialguard.agent.ratelimit import BudgetExhausted

RESULT = {
    "note": "58-year-old woman with stage IV NSCLC, ECOG 1.",
    "results": [
        {
            "nct_id": "NCT001",
            "score": 0.42,
            "trial_verdict": "eligible",
            "assessments": [
                {
                    "criterion": "Stage IV NSCLC",
                    "verdict": "met",
                    "quote": "stage IV NSCLC",
                    "grounded": True,
                },
            ],
        },
        {
            "nct_id": "NCT002",
            "score": 0.31,
            "trial_verdict": "cannot_determine",
            "assessments": [
                {
                    "criterion": "EGFR mutation",
                    "verdict": "unverifiable",
                    "quote": "EGFR positive",
                    "grounded": False,
                    "grounding_failure": True,
                },
            ],
        },
    ],
}


def test_render_shows_badges_citation_and_unverifiable():
    md = demo.render(RESULT)
    assert "🟢 Eligible" in md
    assert "🟡 Cannot determine" in md
    assert "grounded citation" in md and "stage IV NSCLC" in md
    assert "unverifiable" in md.lower()


def test_render_empty():
    assert "No candidate trials" in demo.render({"results": []})


def test_run_empty_note_prompts():
    assert "synthetic patient note" in demo.run("   ").lower()


def test_run_renders_assessment():
    with patch.object(demo, "assess_note", return_value=RESULT):
        out = demo.run("some synthetic note")
    assert "NCT001" in out and "🟢 Eligible" in out


def test_run_handles_budget_exhausted():
    with patch.object(demo, "assess_note", side_effect=BudgetExhausted("cap")):
        out = demo.run("some synthetic note")
    assert "budget" in out.lower()


def test_run_handles_rate_limit():
    with patch.object(demo, "assess_note", side_effect=Exception("429 rate_limit")):
        out = demo.run("some synthetic note")
    assert "rate limit" in out.lower()
