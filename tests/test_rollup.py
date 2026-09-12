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


# --- attach_kinds anchoring (A2) ---

from trialguard.agent.schema import attach_kinds  # noqa: E402


def _typed():
    return [
        {"text": "Age 18 years or older", "kind": "inclusion"},
        {"text": "History of brain metastases", "kind": "exclusion"},
    ]


def test_exact_text_anchors_kind():
    got = attach_kinds([{"criterion": "History of brain metastases"}], _typed())
    assert got[0]["kind"] == "exclusion"


def test_whitespace_and_case_drift_still_anchors():
    """Recovers 7.3% of assessments that used to fall through to a positional guess."""
    got = attach_kinds([{"criterion": "  history of BRAIN metastases  "}], _typed())
    assert got[0]["kind"] == "exclusion"


def test_position_is_used_only_when_counts_agree():
    same_len = attach_kinds(
        [{"criterion": "unrecognisable A"}, {"criterion": "unrecognisable B"}], _typed()
    )
    assert [a["kind"] for a in same_len] == ["inclusion", "exclusion"]


def test_a_short_response_is_not_aligned_by_position():
    """The model dropped a criterion, so index i is no longer typed[i]."""
    got = attach_kinds([{"criterion": "unrecognisable B"}], _typed())
    assert got[0]["kind"] == "unknown"


def test_unknown_kind_can_never_disqualify_a_trial():
    """A guess here would flip 'patient lacks the disqualifier' into an exclusion."""
    ass = attach_kinds(
        [{"criterion": "unrecognisable", "verdict": "not_met"}], _typed()
    )
    roll = rollup_trial(ass)
    assert roll["tier"] == "needs_review"
    assert roll["n_disqualifying"] == 0
    assert roll["n_unknown"] == 1


# --- criteria budget (2026-09-12) --------------------------------------------


def test_a_long_trial_still_gets_its_exclusion_criteria_assessed():
    """Filling the cap with inclusion criteria dropped every disqualifier: 7.2%
    of the live corpus had zero exclusion criteria assessed, so those trials
    could only ever come back eligible or cannot_determine."""
    from trialguard.agent.schema import MAX_CRITERIA, build_typed_criteria

    trial = {
        "inclusion_criteria": [f"inc {i}" for i in range(28)],
        "exclusion_criteria": [f"exc {i}" for i in range(27)],
    }

    criteria, truncated = build_typed_criteria(trial)

    assert truncated is True
    assert len(criteria) == MAX_CRITERIA
    kinds = [c["kind"] for c in criteria]
    assert kinds.count("exclusion") > 0, "every disqualifier was dropped"
    assert kinds.count("inclusion") > 0


def test_an_unused_half_of_the_budget_goes_to_the_other_kind():
    """A trial with few inclusion criteria must still spend the whole cap."""
    from trialguard.agent.schema import MAX_CRITERIA, build_typed_criteria

    criteria, truncated = build_typed_criteria(
        {
            "inclusion_criteria": ["inc 0", "inc 1", "inc 2"],
            "exclusion_criteria": [f"exc {i}" for i in range(40)],
        }
    )

    assert truncated is True
    assert len(criteria) == MAX_CRITERIA
    assert sum(1 for c in criteria if c["kind"] == "inclusion") == 3
    assert sum(1 for c in criteria if c["kind"] == "exclusion") == MAX_CRITERIA - 3


def test_one_sided_trials_are_unaffected():
    from trialguard.agent.schema import MAX_CRITERIA, build_typed_criteria

    criteria, truncated = build_typed_criteria(
        {"inclusion_criteria": [f"inc {i}" for i in range(30)], "exclusion_criteria": []}
    )

    assert truncated is True
    assert len(criteria) == MAX_CRITERIA
    assert all(c["kind"] == "inclusion" for c in criteria)


def test_a_trial_under_the_cap_keeps_every_criterion_and_the_original_order():
    """Inclusion first is the order the analyst prompt has always presented;
    changing it would change every fresh assessment for no reason."""
    from trialguard.agent.schema import build_typed_criteria

    criteria, truncated = build_typed_criteria(
        {"inclusion_criteria": ["a", "b"], "exclusion_criteria": ["c"]}
    )

    assert truncated is False
    assert [c["text"] for c in criteria] == ["a", "b", "c"]
    assert [c["kind"] for c in criteria] == ["inclusion", "inclusion", "exclusion"]
