from trialguard.ingestion.normalise import _split_criteria, normalise_trial


def test_split_inclusion_exclusion():
    raw = """
    Inclusion Criteria:
    - Age >= 18
    - Confirmed HER2-positive breast cancer

    Exclusion Criteria:
    - Prior chemotherapy within 6 months
    - Active CNS metastases
    """
    inc, exc = _split_criteria(raw)
    assert any("HER2" in c for c in inc)
    assert any("chemotherapy" in c for c in exc)


def test_split_no_markers():
    raw = "Must be 18 or older with confirmed diagnosis"
    inc, exc = _split_criteria(raw)
    assert len(inc) > 0
    assert exc == []


def test_split_empty():
    inc, exc = _split_criteria("")
    assert inc == []
    assert exc == []


def test_normalise_trial_adds_fields():
    trial = {
        "nct_id": "NCT000001",
        "title": "Test",
        "eligibility_raw": "Inclusion Criteria:\n- Age >= 18\nExclusion Criteria:\n- Pregnant",
    }
    result = normalise_trial(trial)
    assert "inclusion_criteria" in result
    assert "exclusion_criteria" in result
    assert result["nct_id"] == "NCT000001"
