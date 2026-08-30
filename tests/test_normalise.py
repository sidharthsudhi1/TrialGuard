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


def test_bare_colon_header_splits_exclusions():
    """SIGIR's corpus lost the words 'exclusion criteria'; a lone ':' is all that
    marks the boundary. Filing those as inclusions inverts their roll-up."""
    raw = (
        "inclusion criteria: \n\n Reported pain greater than or equal to 3 out of 10 \n\n"
        " \n: \n\n Less than 18 years of age \n\n Decreased level of consciousness \n"
    )
    inc, exc = _split_criteria(raw)
    assert inc == ["Reported pain greater than or equal to 3 out of 10"]
    assert exc == ["Less than 18 years of age", "Decreased level of consciousness"]


def test_bare_colon_ignored_without_an_inclusion_header():
    """Nothing says which side of the colon is which, so neither side moves."""
    raw = ": \n\n Individuals with known prior cancer of the colon \n"
    inc, exc = _split_criteria(raw)
    assert exc == []
    assert inc == ["Individuals with known prior cancer of the colon"]


def test_spelled_out_exclusion_header_still_wins():
    """CT.gov spells the header out; the bare-colon path must not shadow it."""
    raw = (
        "Inclusion Criteria:\n- Age 18 years or older\n:\n- decoy line long enough to parse\n"
        "Exclusion Criteria:\n- Pregnant or nursing mothers\n"
    )
    inc, exc = _split_criteria(raw)
    assert exc == ["Pregnant or nursing mothers"]
    assert any("Age 18 years or older" in c for c in inc)


def test_bare_colon_conserves_every_criterion():
    """The split must move criteria between lists, never drop or invent one."""
    raw = "inclusion criteria:\n- alpha criterion text\n:\n- beta criterion text\n"
    inc, exc = _split_criteria(raw)
    assert inc + exc == ["alpha criterion text", "beta criterion text"]


# --- disjunctive groups (A4) ---


def test_indented_disjunction_becomes_one_criterion():
    """Split into siblings these are ANDed, so a patient matching one fails the rest."""
    raw = (
        "Inclusion Criteria:\n"
        "- Blunt traumatic event with any of the following:\n"
        "   - Extremity paralysis\n"
        "   - Multiple long bone fractures\n"
        "- Separate unrelated criterion here\n"
    )
    inc, _ = _split_criteria(raw)
    assert len(inc) == 2
    assert "Extremity paralysis" in inc[0] and "Multiple long bone fractures" in inc[0]
    assert inc[1] == "Separate unrelated criterion here"


def test_a_dedented_sibling_is_not_swallowed_into_the_group():
    raw = (
        "Inclusion Criteria:\n"
        "- Meets any of the following:\n"
        "   - first alternative here\n"
        "- back out at parent level\n"
    )
    inc, _ = _split_criteria(raw)
    assert inc[-1] == "back out at parent level"


def test_except_carve_out_stays_with_its_parent():
    """Flattened, the exception is promoted into a disqualifier of its own."""
    raw = (
        "Exclusion Criteria:\n"
        "- Any prior malignancy, except:\n"
        "   - adequately treated non-melanoma skin cancer\n"
    )
    _, exc = _split_criteria(raw)
    assert len(exc) == 1
    assert "non-melanoma skin cancer" in exc[0]


def test_flat_text_is_never_grouped():
    """The eval corpora were flattened upstream; extent is then unknowable."""
    raw = (
        "inclusion criteria: \n\n Meets any of the following: \n\n"
        " first alternative here \n\n second alternative here \n"
    )
    inc, _ = _split_criteria(raw)
    assert len(inc) == 3


def test_one_space_of_drift_is_not_nesting():
    raw = "Inclusion Criteria:\n- Meets any of the following:\n - only one space in\n"
    inc, _ = _split_criteria(raw)
    assert len(inc) == 2


def test_an_oversized_group_is_left_flat():
    """A 9k-character criterion would crowd out the other 23 in the prompt."""
    child = "x" * 400
    raw = "Inclusion Criteria:\n- Meets any of the following:\n" + "".join(
        f"   - {child}\n" for _ in range(5)
    )
    inc, _ = _split_criteria(raw)
    assert len(inc) == 6
    assert all(len(c) <= 1500 for c in inc)
