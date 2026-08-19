"""End-to-end eval scoring — the composition of retrieval and verdicts."""

from __future__ import annotations

from trialguard.eval.end_to_end import score


def _row(pid, gold_elig, retrieved, verdicts, labels=None, incomplete=False):
    row = {
        "patient_id": pid,
        "gold_eligible": gold_elig,
        "gold_labels": labels or {n: "eligible" for n in gold_elig},
        "retrieved": retrieved,
        "verdicts": {
            n: {"trial_verdict": v, "n_criteria": 4, "n_grounded": 4, "n_unverifiable": 0}
            for n, v in verdicts.items()
        },
    }
    if incomplete:
        row["incomplete"] = True
    return row


def test_end_to_end_is_bounded_by_retrieval():
    """A trial never retrieved cannot be assessed, so the agent cannot recover it."""
    rows = [_row("p1", ["A", "B"], ["A"], {"A": "eligible"})]
    m = score(rows)
    assert m["retrieval_recall"] == 0.5
    assert m["end_to_end_recall"] == 0.5
    assert m["agent_loss"] == 0.0


def test_agent_loss_separates_the_stages():
    """Retrieved but abstained is an agent loss, not a retrieval loss."""
    rows = [_row("p1", ["A", "B"], ["A", "B"], {"A": "eligible", "B": "cannot_determine"})]
    m = score(rows)
    assert m["retrieval_recall"] == 1.0
    assert m["end_to_end_recall"] == 0.5
    assert m["agent_loss"] == 0.5


def test_precision_counts_wrong_eligible_calls():
    """Calling an excluded trial eligible must cost precision, not recall."""
    rows = [
        _row(
            "p1", ["A"], ["A", "X"],
            {"A": "eligible", "X": "eligible"},
            labels={"A": "eligible", "X": "excluded"},
        )
    ]
    m = score(rows)
    assert m["end_to_end_recall"] == 1.0
    assert m["eligible_precision"] == 0.5


def test_empty_gold_never_divides_by_zero():
    assert score([])["end_to_end_recall"] == 0.0
    assert score([_row("p1", [], [], {})])["retrieval_recall"] == 0.0


def test_incomplete_runs_are_flagged():
    """A budget stop mid-run must be visible, not scored as a complete result."""
    rows = [_row("p1", ["A"], ["A"], {}, incomplete=True)]
    assert score(rows)["incomplete_patients"] == 1


def test_unverifiable_rate_is_criterion_level():
    rows = [_row("p1", ["A"], ["A"], {"A": "eligible"})]
    rows[0]["verdicts"]["A"]["n_unverifiable"] = 1
    assert score(rows)["criterion_unverifiable_rate"] == 0.25
