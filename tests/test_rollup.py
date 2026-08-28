"""Tiered trial roll-up (A5): the verdict, and what stands between it and eligible."""

from trialguard.agent.schema import rollup_trial, rollup_trial_verdict


def _a(kind, verdict, text="c"):
    return {"kind": kind, "verdict": verdict, "criterion": text}


def test_all_met_is_the_eligible_tier():
    roll = rollup_trial([_a("inclusion", "met"), _a("exclusion", "not_met")])
    assert roll["tier"] == "eligible"
    assert roll["verdict"] == "eligible"
    assert roll["n_unknown"] == 0


def test_unknowns_alone_are_review_not_exclusion():
    """The distinction the tier exists for: nothing disqualifies this patient."""
    roll = rollup_trial([
        _a("inclusion", "met"),
        _a("inclusion", "cannot_determine", "ejection fraction"),
        _a("exclusion", "unverifiable", "prior therapy"),
    ])
    assert roll["tier"] == "needs_review"
    assert roll["n_unknown"] == 2
    assert roll["unknown"] == ["ejection fraction", "prior therapy"]
    assert roll["n_disqualifying"] == 0


def test_a_disqualifier_outranks_any_number_of_unknowns():
    roll = rollup_trial([
        _a("exclusion", "met", "brain metastases"),
        _a("inclusion", "cannot_determine"),
    ])
    assert roll["tier"] == "excluded"
    assert roll["disqualifying"] == ["brain metastases"]


def test_exclusion_semantics_stay_inverted():
    """Exclusion met disqualifies; exclusion not_met is a pass, not a failure."""
    assert rollup_trial([_a("exclusion", "met")])["tier"] == "excluded"
    assert rollup_trial([_a("exclusion", "not_met")])["tier"] == "eligible"
    assert rollup_trial([_a("inclusion", "not_met")])["tier"] == "excluded"


def test_untyped_assessments_default_to_inclusion():
    assert rollup_trial([{"verdict": "not_met", "criterion": "x"}])["tier"] == "excluded"


def test_empty_never_claims_eligible():
    roll = rollup_trial([])
    assert roll["tier"] == "needs_review"
    assert roll["verdict"] == "cannot_determine"
    assert roll["n_criteria"] == 0


def test_verdict_wrapper_is_unchanged_for_existing_callers():
    """The regression gate and faithfulness floors read this, not the tier."""
    for ass, expected in [
        ([_a("inclusion", "met")], "eligible"),
        ([_a("inclusion", "not_met")], "excluded"),
        ([_a("inclusion", "cannot_determine")], "cannot_determine"),
        ([], "cannot_determine"),
    ]:
        assert rollup_trial_verdict(ass) == expected


def test_needs_review_rows_can_be_ranked_by_unknown_count():
    few = rollup_trial([_a("inclusion", "met"), _a("inclusion", "cannot_determine")])
    many = rollup_trial([_a("inclusion", "cannot_determine") for _ in range(5)])
    assert few["n_unknown"] < many["n_unknown"]
