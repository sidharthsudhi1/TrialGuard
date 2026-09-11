"""The parser emits things that are not criteria (fix 2, 2026-09-11)."""

from __future__ import annotations

import pytest

from trialguard.ingestion.normalise import _is_not_a_criterion, normalise_trial

# Taken verbatim from the eval cohorts, by frequency. "inclusion criteria:"
# survived as its own criterion 91 times on TREC.
JUNK = [
    "inclusion criteria:",
    "inclusion criteria",
    "DISEASE CHARACTERISTICS:",
    "PATIENT CHARACTERISTICS:",
    "PROTOCOL ENTRY CRITERIA:",
    "Performance status:",
    "Life expectancy:",
    "Hematopoietic:",
    "Cardiovascular:",
    "Disease Characteristics--",
    "Prior/Concurrent Therapy--",
    "Not specified",
    "No eligibility criteria",
]

# Also from the cohorts. Every one is three words or fewer, which is why the
# rule is structural: a length filter would delete all of them.
REAL = [
    "Karnofsky 60-100%",
    "Contrast allergy",
    "Nonpalpable femoral pulses",
    "Age below 18",
    "Metformin treatment",
    "nodular goiter",
    "African American",
    "Pregnant women",
    "Clinical gout.",
    "Negative pregnancy test",
    "Written informed consent.",
    "Supraventricular tachycardia",
]


@pytest.mark.parametrize("line", JUNK)
def test_parser_artifacts_are_not_criteria(line):
    assert _is_not_a_criterion(line) is True


@pytest.mark.parametrize("line", REAL)
def test_short_criteria_are_kept(line):
    """Short does not mean junk, and a word-count filter would delete these."""
    assert _is_not_a_criterion(line) is False


def test_a_label_with_content_after_the_colon_is_kept():
    assert _is_not_a_criterion("General: Age equal to or greater than 18.") is False


def test_a_group_header_is_never_dropped():
    """It carries a disjunction the roll-up depends on; _GROUP_HEADER owns what
    happens to it, and dropping it here would delete that decision."""
    assert _is_not_a_criterion("Patients must meet any of the following:") is False
    assert _is_not_a_criterion("except:") is False


def test_a_stranded_header_is_dropped_from_a_real_trial():
    raw = (
        "Inclusion Criteria:\n"
        "  DISEASE CHARACTERISTICS:\n"
        "  Histologically confirmed non-small cell lung cancer\n"
        "  Performance status:\n"
        "  ECOG performance status 0-1\n"
        "Exclusion Criteria:\n"
        "  Not specified\n"
        "  Active brain metastases\n"
    )
    out = normalise_trial({"nct_id": "NCT1", "eligibility_raw": raw})

    assert out["inclusion_criteria"] == [
        "Histologically confirmed non-small cell lung cancer",
        "ECOG performance status 0-1",
    ]
    assert out["exclusion_criteria"] == ["Active brain metastases"]


def test_the_escape_hatch_restores_the_previous_parse(monkeypatch):
    """Dropping artifacts changes the question the analyst is asked, so it
    changes the analyst cache namespace with it; reproducing anything cached
    before 2026-09-11 needs the old parse and the old keys together."""
    monkeypatch.setenv("TG_STRICT_CRITERIA", "0")
    raw = "Inclusion Criteria:\n  Performance status:\n  ECOG performance status 0-1\n"

    out = normalise_trial({"nct_id": "NCT1", "eligibility_raw": raw})

    assert "Performance status:" in out["inclusion_criteria"]


def test_the_cache_namespace_moves_with_the_parser(monkeypatch):
    """A pre-fix entry must not be replayed against a different criteria list."""
    from trialguard.agent.analyst import _cache_key

    monkeypatch.setenv("TG_STRICT_CRITERIA", "0")
    old = _cache_key("note", "NCT1")
    monkeypatch.setenv("TG_STRICT_CRITERIA", "1")

    assert _cache_key("note", "NCT1") != old


def test_a_real_criterion_under_eleven_characters_is_still_lost():
    """Not this fix, and recorded so it is not mistaken for it. _parse_block
    drops any line of 10 characters or fewer, which removes junk like "Age:"
    and "Other:" but also real criteria: "Pregnancy", "Prisoners", "Children",
    "CALGB 0-2". Measured at 100 dropped lines on TREC and 187 on SIGIR, of
    which roughly a quarter are real.

    Relaxing the threshold is a separate change: the structural filter above
    now catches the junk the length rule was crudely catching, but relaxing it
    also admits wrapped-line fragments ("and over", "RELATIVE", "Main"), so it
    needs its own measurement rather than being bundled into this one.
    """
    out = normalise_trial(
        {"nct_id": "NCT1", "eligibility_raw": "Exclusion Criteria:\n  Pregnancy\n"}
    )

    assert out["exclusion_criteria"] == []
