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
