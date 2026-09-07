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


def test_the_verdict_distribution_is_reported_not_just_the_abstention_rate():
    """L4 improved the faithfulness proxy 26% while making the system strictly
    worse; only the criterion-level split showed the vanished grounding failures
    had become abstentions rather than recoveries."""
    rows = [_row("p1", ["A"], ["A"], {"A": "eligible"})]
    rows[0]["verdicts"]["A"]["verdicts"] = {
        "met": 2, "not_met": 1, "cannot_determine": 1, "unverifiable": 0
    }

    m = score(rows)

    assert m["criterion_verdicts"] == {
        "met": 2, "not_met": 1, "cannot_determine": 1, "unverifiable": 0
    }
    assert m["criterion_grounded"] == 4
    assert m["criterion_total"] == 4


def test_criteria_the_analyst_never_answered_are_counted():
    """A prompt that answers fewer criteria than it was handed shrinks every
    denominator below, so the rates look unchanged while coverage fell."""
    rows = [_row("p1", ["A"], ["A"], {"A": "eligible"})]
    rows[0]["verdicts"]["A"]["n_criteria_asked"] = 6

    m = score(rows)

    assert m["criterion_asked"] == 6
    assert m["criterion_unanswered"] == 2
    # Over what was asked, not over what came back.
    assert m["criterion_grounded_rate"] == round(4 / 6, 4)


def test_an_arm_that_answers_everything_reports_no_shortfall():
    m = score([_row("p1", ["A"], ["A"], {"A": "eligible"})])

    assert m["criterion_unanswered"] == 0
    assert m["criterion_grounded_rate"] == 1.0


def _assess_rows():
    return [
        {
            "patient_id": "p1",
            "note": "62 M, mCRC",
            "gold_eligible": ["A", "B"],
            "gold_labels": {"A": "eligible", "B": "eligible"},
            "retrieved": ["A", "B", "C"],
        }
    ]


_CORPUS = {
    n: {
        "nct_id": n,
        "inclusion_criteria": ["Age >= 18"],
        "exclusion_criteria": ["Prior chemo"],
        "eligibility_raw": "Age >= 18. Prior chemo.",
    }
    for n in ("A", "B", "C")
}


def _run_assess(workers, monkeypatch, assess_impl):
    from unittest.mock import patch

    from trialguard.eval import end_to_end as E

    monkeypatch.setenv("TG_EVAL_WORKERS", str(workers))
    rows = _assess_rows()
    with patch.object(E, "_load_corpus", return_value=_CORPUS), patch(
        "trialguard.agent.graph.assess", side_effect=assess_impl
    ):
        counts = E.assess_retrieved(rows, "sigir")
    return rows, counts


def test_concurrent_and_serial_arms_produce_the_same_verdicts(monkeypatch):
    """Opting into workers must move the wall clock and nothing else."""
    def _ok(note, nct, criteria, source, **kw):
        return {
            "trial_verdict": "eligible",
            "trial_tier": "eligible",
            "assessments": [{"verdict": "met", "grounded": True}] * 2,
        }

    serial, c1 = _run_assess(1, monkeypatch, _ok)
    parallel, c2 = _run_assess(8, monkeypatch, _ok)

    assert serial[0]["verdicts"] == parallel[0]["verdicts"]
    assert c1["assessed"] == c2["assessed"] == 3


def test_a_budget_stop_marks_every_patient_left_unfinished(monkeypatch):
    """Pooling loses which patient the exhausted call belonged to, so a run that
    stops early must not report any patient as scored over its full pool."""
    from trialguard.agent.ratelimit import BudgetExhausted

    calls = {"n": 0}

    def _die(note, nct, criteria, source, **kw):
        calls["n"] += 1
        if calls["n"] > 1:
            raise BudgetExhausted("cap")
        return {"trial_verdict": "eligible", "trial_tier": "eligible", "assessments": []}

    rows, counts = _run_assess(1, monkeypatch, _die)

    assert counts["budget_stops"] == 1
    assert rows[0]["incomplete"] is True
    assert score(rows)["incomplete_patients"] == 1


def test_one_failing_trial_does_not_void_the_run(monkeypatch):
    def _flaky(note, nct, criteria, source, **kw):
        if nct == "B":
            raise RuntimeError("provider 500")
        return {"trial_verdict": "eligible", "trial_tier": "eligible", "assessments": []}

    rows, counts = _run_assess(1, monkeypatch, _flaky)

    assert counts["skipped"] == 1
    assert set(rows[0]["verdicts"]) == {"A", "C"}
