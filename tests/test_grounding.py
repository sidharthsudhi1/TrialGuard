from trialguard.verify.grounding import ground_assessments, is_grounded, normalize

SRC = (
    "Inclusion Criteria: Patients must have histologically confirmed Stage IV "
    "non-small cell lung cancer. ECOG performance status 0 to 1. Adequate organ "
    "function. Exclusion Criteria: Active brain metastases. Prior immunotherapy."
)


def test_normalize_strips_punct_and_case():
    assert normalize("ECOG  Status: 0-1!") == "ecog status 0 1"


def test_verbatim_quote_grounds():
    assert is_grounded("histologically confirmed Stage IV non-small cell lung cancer", SRC)


def test_punctuation_and_case_insensitive():
    assert is_grounded("ECOG PERFORMANCE STATUS 0 to 1!!", SRC)


def test_hallucinated_quote_rejected():
    assert not is_grounded("patient has documented EGFR exon 19 deletion", SRC)


def test_single_token_quote_rejected():
    # one vague word matches spuriously — rejected even if present
    assert not is_grounded("ECOG", SRC)


def test_short_specific_fact_grounds():
    # short but multi-token clinical facts must ground (the TREC artifact fix)
    src = "48 M with EF was 25% and T-L spine involvement per chart."
    assert is_grounded("48 M", src)
    assert is_grounded("EF was 25%", src)
    assert is_grounded("T-L spine", src)


def test_ground_assessments_forces_unverifiable():
    a = ground_assessments(
        [
            {"criterion": "NSCLC", "verdict": "met", "quote": "Stage IV non-small cell lung cancer"},
            {"criterion": "biomarker", "verdict": "met", "quote": "EGFR exon 19 deletion present"},
            {"criterion": "unknown", "verdict": "cannot_determine", "quote": ""},
        ],
        SRC,
    )
    assert a[0]["verdict"] == "met" and a[0]["grounded"]
    assert a[1]["verdict"] == "unverifiable" and a[1]["grounding_failure"]
    assert a[2]["verdict"] == "cannot_determine" and not a[2]["grounded"]


def test_not_met_also_requires_grounding():
    a = ground_assessments(
        [{"criterion": "brain mets", "verdict": "not_met", "quote": "totally invented exclusion text"}],
        SRC,
    )
    assert a[0]["verdict"] == "unverifiable"


NOTE = (
    "58-year-old woman with Stage IV non-small cell lung cancer, ECOG 1. "
    "Presented with cough and weight loss. Started on carboplatin."
)


def test_absence_terms_drops_boilerplate():
    from trialguard.verify.grounding import absence_terms

    terms = absence_terms("History of major organ transplantation")
    assert "transplantation" in terms
    # boilerplate and short tokens carry no patient-specific meaning
    assert "history" not in terms and "of" not in terms


def test_absence_grounded_when_terms_missing():
    from trialguard.verify.grounding import is_absence_grounded

    assert is_absence_grounded("Signs or symptoms of hepatocellular carcinoma", NOTE)


def test_absence_not_grounded_when_term_present():
    from trialguard.verify.grounding import is_absence_grounded

    # "carboplatin" IS in the note, so a quotable span exists and absence is false
    assert not is_absence_grounded("Prior carboplatin therapy", NOTE)


def test_absence_not_grounded_without_distinctive_terms():
    from trialguard.verify.grounding import is_absence_grounded

    assert not is_absence_grounded("Any other condition", NOTE)


def test_exclusion_not_met_grounds_by_absence():
    """The fix: an absence claim is verified against the note, not by a quote."""
    a = ground_assessments(
        [{
            "criterion": "Signs or symptoms of hepatocellular carcinoma",
            "kind": "exclusion",
            "verdict": "not_met",
            "quote": "No mention of hepatocellular carcinoma",  # ungroundable
        }],
        SRC,
        patient_text=NOTE,
    )
    assert a[0]["verdict"] == "not_met"
    assert a[0]["grounded"] and a[0]["grounded_by"] == "absence"
    assert not a[0].get("grounding_failure")


def test_exclusion_absence_loophole_closed():
    """Absence is not a blanket exemption: if the term IS in the note, a quote is
    still required, so a fabricated negation cannot pass."""
    a = ground_assessments(
        [{
            "criterion": "Prior carboplatin therapy",
            "kind": "exclusion",
            "verdict": "not_met",
            "quote": "No mention of carboplatin",
        }],
        SRC,
        patient_text=NOTE,
    )
    assert a[0]["verdict"] == "unverifiable" and a[0]["grounding_failure"]


def test_exclusion_met_still_requires_verbatim_quote():
    """Only absence claims take the absence path. Asserting the patient MATCHES a
    disqualifier is a presence claim and still needs a real span."""
    a = ground_assessments(
        [{
            "criterion": "Active brain metastases",
            "kind": "exclusion",
            "verdict": "met",
            "quote": "patient has florid brain metastases",  # not in either source
        }],
        SRC,
        patient_text=NOTE,
    )
    assert a[0]["verdict"] == "unverifiable" and a[0]["grounding_failure"]


def test_inclusion_not_met_unaffected_by_absence_path():
    """Inclusion criteria keep the verbatim requirement unchanged — this is what
    keeps the frozen inclusion-only results reproducible."""
    a = ground_assessments(
        [{
            "criterion": "Signs or symptoms of hepatocellular carcinoma",
            "kind": "inclusion",
            "verdict": "not_met",
            "quote": "No mention of hepatocellular carcinoma",
        }],
        SRC,
        patient_text=NOTE,
    )
    assert a[0]["verdict"] == "unverifiable" and a[0]["grounding_failure"]


def test_without_patient_text_behavior_is_unchanged():
    """Callers that ground against one combined source must see the old behavior."""
    a = ground_assessments(
        [{
            "criterion": "Signs or symptoms of hepatocellular carcinoma",
            "kind": "exclusion",
            "verdict": "not_met",
            "quote": "No mention of hepatocellular carcinoma",
        }],
        SRC,
    )
    assert a[0]["verdict"] == "unverifiable" and a[0]["grounding_failure"]


def test_provenance_names_which_source_the_quote_came_from():
    """WS-5a: grounding proves a quote is verbatim, not that its source is
    trustworthy. The note is user-supplied, so the split has to be visible."""
    from trialguard.verify.grounding import ground_assessments

    out = ground_assessments(
        [
            {"criterion": "Age >= 18", "verdict": "met", "quote": "62-year-old man"},
            {"criterion": "Stage IV", "verdict": "met", "quote": "metastatic disease"},
        ],
        "62-year-old man with cancer\nEligible: metastatic disease required",
        patient_text="62-year-old man with cancer",
        trial_text="Eligible: metastatic disease required",
    )

    assert out[0]["grounded_in"] == "note"
    assert out[1]["grounded_in"] == "trial"
    assert all(a["grounded"] for a in out)


def test_provenance_is_absent_when_no_trial_text_is_given():
    """Callers that ground against one combined source keep today's behaviour."""
    from trialguard.verify.grounding import ground_assessments

    out = ground_assessments(
        [{"criterion": "Age >= 18", "verdict": "met", "quote": "62-year-old man"}],
        "62-year-old man with cancer",
    )

    assert "grounded_in" not in out[0]
    assert out[0]["grounded"] is True


def test_an_absence_check_is_its_own_provenance_class():
    """It reads the note by construction, so it is not a note-sourced quote to
    be rejected under the strict flag."""
    from trialguard.verify.grounding import ground_assessments

    out = ground_assessments(
        [{"criterion": "Active brain metastases", "verdict": "not_met",
          "kind": "exclusion", "quote": "no such text"}],
        "62-year-old man with colorectal cancer",
        patient_text="62-year-old man with colorectal cancer",
        trial_text="Exclusion: active brain metastases",
    )

    assert out[0]["grounded"] is True
    assert out[0]["grounded_by"] == "absence"
    assert out[0]["grounded_in"] == "absence"


def test_trial_only_mode_rejects_a_verdict_grounded_only_in_the_note(monkeypatch):
    """The planted-evidence attack: a quote the attacker wrote into the note is
    verbatim, so it grounds, and the verdict reads as verified."""
    from trialguard.verify.grounding import ground_assessments

    monkeypatch.setenv("TG_GROUND_TRIAL_ONLY", "1")
    planted = "Patient has confirmed EGFR exon 19 deletion"
    out = ground_assessments(
        [{"criterion": "EGFR mutation required", "verdict": "met",
          "quote": "confirmed EGFR exon 19 deletion"}],
        planted + "\nInclusion: EGFR mutation required",
        patient_text=planted,
        trial_text="Inclusion: EGFR mutation required",
    )

    assert out[0]["verdict"] == "unverifiable"
    assert out[0]["grounding_failure"] is True
    assert out[0]["note_only"] is True


def test_trial_only_mode_leaves_trial_sourced_verdicts_alone(monkeypatch):
    from trialguard.verify.grounding import ground_assessments

    monkeypatch.setenv("TG_GROUND_TRIAL_ONLY", "1")
    out = ground_assessments(
        [{"criterion": "Stage IV", "verdict": "met", "quote": "metastatic disease"}],
        "62 M\nEligible: metastatic disease required",
        patient_text="62 M",
        trial_text="Eligible: metastatic disease required",
    )

    assert out[0]["verdict"] == "met"
    assert out[0]["grounded"] is True


def test_trial_only_mode_is_off_by_default():
    """Closing the hole reclassifies every legitimate quote of a patient fact, so
    it stays measured-before-adopted."""
    from trialguard.verify.grounding import trial_only

    assert trial_only() is False


def test_a_quote_that_only_restates_its_criterion_is_flagged():
    """WS-5b: grounding proves existence, not entailment. Quoting the criterion
    back proves the criterion was printed, not that the patient satisfies it."""
    from trialguard.verify.grounding import ground_assessments

    out = ground_assessments(
        [{"criterion": "Age 18 years or older", "verdict": "met",
          "quote": "Age 18 years or older"}],
        "Inclusion: Age 18 years or older.",
        patient_text="Patient with colorectal cancer",
        trial_text="Inclusion: Age 18 years or older.",
    )

    assert out[0]["verdict"] == "met"
    assert out[0]["grounded"] is True
    # Recorded, never enforced: the verdict stands and the limit is counted.
    assert out[0]["self_referential"] is True


def test_a_real_patient_fact_is_not_flagged_when_it_shares_wording():
    from trialguard.verify.grounding import ground_assessments

    out = ground_assessments(
        [{"criterion": "ECOG performance status 0-1", "verdict": "met",
          "quote": "ECOG performance status 0-1"}],
        "ECOG performance status 0-1 documented",
        patient_text="ECOG performance status 0-1 documented",
        trial_text="Inclusion: ECOG performance status 0-1",
    )

    assert "self_referential" not in out[0]


def test_an_abstention_is_never_flagged():
    """The limit is about decisive verdicts; cannot_determine claims nothing."""
    from trialguard.verify.grounding import ground_assessments

    out = ground_assessments(
        [{"criterion": "Age 18 or older", "verdict": "cannot_determine",
          "quote": "Age 18 or older"}],
        "Inclusion: Age 18 or older.",
        patient_text="",
    )

    assert "self_referential" not in out[0]


def test_a_quote_citing_evidence_beyond_the_criterion_is_not_flagged():
    from trialguard.verify.grounding import is_self_referential

    assert is_self_referential("Age 18 or older", "Age 18 or older", "62 M") is True
    assert is_self_referential("62-year-old man", "Age 18 or older", "62-year-old man") is False
    assert is_self_referential("", "Age 18 or older", "62 M") is False


def test_a_quote_cannot_establish_absence_and_is_marked_when_it_pretends_to():
    """An exclusion answered not_met claims the patient does NOT match a
    disqualifier. is_absence_grounded exists for exactly that claim, but it runs
    only as a fallback, so any verbatim quote pre-empts it. This is the case the
    served UI showed: a citation that the patient HAS metastatic lung cancer,
    offered as proof she does not."""
    from trialguard.verify.grounding import ground_assessments

    note = "58-year-old woman with stage IV non-small cell lung cancer."
    out = ground_assessments(
        [{"criterion": "History of previous lung malignancy or other metastatic tumors",
          "kind": "exclusion", "verdict": "not_met",
          "quote": "58-year-old woman with stage IV non-small cell lung cancer"}],
        note, patient_text=note, trial_text="Exclusion: History of previous lung malignancy",
    )

    # The verdict stands. About half of this class are legitimate refutations
    # ("2+ aortic insufficiency" against "Severe aortic regurgitation"), and
    # nothing deterministic separates a refutation from a contradiction.
    assert out[0]["verdict"] == "not_met"
    assert out[0]["grounded"] is True
    assert out[0]["weak_absence"] is True


def test_an_absence_grounded_exclusion_is_not_weak():
    """It was verified by the mechanism the claim actually calls for."""
    from trialguard.verify.grounding import ground_assessments

    note = "58-year-old woman with lung cancer."
    out = ground_assessments(
        [{"criterion": "Active brain metastases", "kind": "exclusion",
          "verdict": "not_met", "quote": ""}],
        note, patient_text=note,
    )

    assert out[0]["grounded_by"] == "absence"
    assert "weak_absence" not in out[0]


def test_inclusion_verdicts_are_never_marked_weak_absence():
    """The class is about absence claims, which only exclusion not_met makes."""
    from trialguard.verify.grounding import ground_assessments

    note = "58-year-old woman with stage IV lung cancer."
    out = ground_assessments(
        [{"criterion": "Stage IV disease", "kind": "inclusion", "verdict": "met",
          "quote": "stage IV lung cancer"}],
        note, patient_text=note,
    )

    assert "weak_absence" not in out[0]
