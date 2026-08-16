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
