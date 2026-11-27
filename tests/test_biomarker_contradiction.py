"""Biomarker contradiction: explicit statuses in the note against what a trial's inclusion requires."""

from __future__ import annotations

import pytest

from trialguard.eval.biomarker_contradiction import (
    contradictions,
    flip_keywords,
    flip_text,
    inclusion_section,
    note_statuses,
    observational_share,
    score,
    trial_requirements,
)

# --- note statuses ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The tumor was HER2-positive and ER/PR negative.",
         {"HER2": "pos", "ER": "neg", "PR": "neg", "HR": "neg"}),
        ("HER2: positive", {"HER2": "pos"}),
        ("Hormone receptors: ER+, PR-", {"ER": "pos", "PR": "neg", "HR": "pos"}),
        ("Hormone receptors: PR positive", {"PR": "pos", "HR": "pos"}),
        ("ER+/PR+/HER2-", {"ER": "pos", "PR": "pos", "HR": "pos", "HER2": "neg"}),
        ("triple-negative breast cancer",
         {"ER": "neg", "PR": "neg", "HER2": "neg", "HR": "neg"}),
        ("triple negative breast cancer (TNBC)",
         {"ER": "neg", "PR": "neg", "HER2": "neg", "HR": "neg"}),
        ("triple-positive breast cancer",
         {"ER": "pos", "PR": "pos", "HER2": "pos", "HR": "pos"}),
        ("hormone receptor-positive disease", {"HR": "pos"}),
        ("HER2 amplified", {"HER2": "pos"}),
        ("HER2 not amplified", {"HER2": "neg"}),
        ("estrogen receptor negative", {"ER": "neg"}),
        ("EGFR exon 19 deletion was not tested; KRAS wild-type", {"KRAS": "neg"}),
        ("EGFR-mutant adenocarcinoma", {"EGFR": "pos"}),
        ("harbouring an EGFR mutation", {"EGFR": "pos"}),
        ("no EGFR mutation", {"EGFR": "neg"}),
        ("negative for EGFR mutation", {"EGFR": "neg"}),
        ("ALK rearrangement positive", {"ALK": "pos"}),
        ("BRAF V600E mutation", {"BRAF": "pos"}),
        ("germline BRCA1 mutation carrier", {"BRCA": "pos"}),
    ],
)
def test_note_statuses_reads_explicit_status(text, expected):
    assert note_statuses(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "presented to the ER with chest pain",
        "pulse 88 per hr, BP 120/80",
        "prior anti-HER2 therapy",
        "treated with an EGFR inhibitor",
        "HER2 status unknown",
        "Disorder: glaucoma\nVisual acuity: 0.3",
    ],
)
def test_note_statuses_ignores_mentions_without_status(text):
    assert note_statuses(text) == {}


def test_conflicting_statuses_in_one_text_drop_the_marker():
    assert "HER2" not in note_statuses("Initially HER2-negative; on rebiopsy HER2-positive.")


def test_hr_is_not_inferred_negative_from_one_receptor():
    assert note_statuses("ER negative") == {"ER": "neg"}


# --- trial requirements ----------------------------------------------------


TRIAL = (
    "Summary: A study of drug X in HER2-negative patients.\n"
    "Inclusion criteria: inclusion criteria: \n\n"
    " Histologically confirmed HER2-positive breast cancer \n\n"
    " ECOG 0-1 \n\n"
    "Exclusion criteria: \n\n Triple-negative disease \n"
)


def test_inclusion_section_skips_summary_and_exclusion():
    section = inclusion_section(TRIAL)
    assert "HER2-positive" in section
    assert "HER2-negative" not in section
    assert "Triple-negative" not in section


def test_inclusion_section_ctgov_layout():
    text = "Inclusion Criteria:\n\n- EGFR mutation\n\nExclusion Criteria:\n\n- ALK fusion"
    assert inclusion_section(text).strip() == "- EGFR mutation"


def test_inclusion_section_without_headers_is_whole_text():
    assert inclusion_section("HER2-positive") == "HER2-positive"


def test_trial_requirements_reads_inclusion_only():
    assert trial_requirements(TRIAL) == {"HER2": "pos"}


def test_trial_allowing_either_status_requires_neither():
    text = "Inclusion Criteria:\n- HER2-positive or HER2-negative breast cancer"
    assert trial_requirements(text) == {}


def test_trial_requirements_closes_triple_negative():
    text = "Inclusion Criteria:\n- triple negative breast cancer"
    assert trial_requirements(text) == {"ER": "neg", "PR": "neg", "HER2": "neg", "HR": "neg"}


@pytest.mark.parametrize(
    "text",
    [
        "Inclusion Criteria:\n- ER-positive and/or PR-positive, HER2-negative breast cancer",
        "Inclusion Criteria:\n- ER and/or PR positive, HER2-negative",
        "Inclusion Criteria:\n- estrogen receptor positive or progesterone receptor positive; HER2-negative",
    ],
)
def test_receptor_disjunction_requires_hormone_receptor_not_each(text):
    assert trial_requirements(text) == {"HR": "pos", "HER2": "neg"}


def test_plus_minus_is_no_requirement():
    text = "Inclusion Criteria:\n- invasive breast cancer (ER+/-, PR+/-, HER2+/-)"
    assert trial_requirements(text) == {}


def test_trial_accepting_hr_positive_or_triple_negative_requires_only_her2():
    """NCT05747794: HR+ or HR- (TNBC), and HER2-negative."""
    text = (
        "Inclusion Criteria:\n- Metastatic HR+ positive (estrogen receptor positive and/or "
        "progesterone receptor positive) or hormone receptor negative (HR\u02d7), and HER2-neg "
        "breast adenocarcinoma\n- Participants with HR\u02d7 MBC (i.e. triple-negative breast "
        "cancer [TNBC])"
    )
    assert trial_requirements(text) == {"HER2": "neg"}


@pytest.mark.parametrize("minus", ["\u2212", "\u02d7"])
def test_unicode_minus_is_negative(minus):
    assert note_statuses(f"HER2{minus}") == {"HER2": "neg"}


# --- contradictions --------------------------------------------------------


def test_contradiction_is_a_disagreement_on_a_shared_marker():
    note = {"HER2": "pos", "ER": "neg", "PR": "neg", "HR": "neg"}
    assert contradictions(note, {"HR": "pos"}) == ["HR"]
    assert contradictions(note, {"HER2": "pos"}) == []


def test_absence_is_never_a_contradiction():
    assert contradictions({}, {"HER2": "pos"}) == []
    assert contradictions({"HER2": "pos"}, {}) == []


# --- flips -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "flipped"),
    [
        ("The tumor was HER2-positive and ER/PR negative.",
         "The tumor was HER2-negative and ER/PR positive."),
        ("HER2: positive", "HER2: negative"),
        ("Hormone receptors: ER+, PR-", "Hormone receptors: ER-, PR+"),
        ("triple-negative breast cancer", "triple-positive breast cancer"),
        ("her2 positive breast cancer", "her2 negative breast cancer"),
        ("KRAS wild-type", "KRAS mutant"),
        ("EGFR-mutant", "EGFR-wild-type"),
        ("glaucoma", "glaucoma"),
    ],
)
def test_flip_text(text, flipped):
    assert flip_text(text) == flipped


def test_flip_inverts_every_status():
    note = "HER2-positive, ER+/PR-, KRAS wild-type"
    before, after = note_statuses(note), note_statuses(flip_text(note))
    assert set(before) == set(after)
    # HR is implied: ER+/PR- and ER-/PR+ are both hormone-receptor positive.
    assert all(before[m] != after[m] for m in before if m != "HR")


def test_flip_keywords_drops_a_keyword_the_flipped_note_contradicts():
    # ER+/PR- flips to ER-/PR+, which is still hormone-receptor positive.
    note = flip_text("Hormone receptors: ER+, PR-")
    kept, dropped = flip_keywords(["er positive breast cancer", "hormone receptor positive"], note)
    assert kept == ["er negative breast cancer"]
    assert dropped == ["hormone receptor negative"]


# --- scoring ---------------------------------------------------------------


def _trial(req: str) -> str:
    return f"Inclusion Criteria:\n- {req}\n"


def test_score_pools_top_k_and_bootstraps_patients():
    trials = {
        "NCT1": _trial("HER2-positive disease"),
        "NCT2": _trial("HER2-negative disease"),
        "NCT3": _trial("adult"),
    }
    rows = [
        {"patient_id": "a", "note": "HER2-positive", "retrieved": ["NCT1", "NCT2", "NCT3"]},
        {"patient_id": "b", "note": "no marker stated", "retrieved": ["NCT2", "NCT3", "NCT1"]},
    ]
    out = score(rows, trials, k=2)
    assert out["surfaced"] == 4
    assert out["contradicted"] == 1
    assert out["rate"]["point"] == 0.25
    assert out["rate"]["n"] == 2
    assert out["patients_with_marker"] == 1
    assert out["rate_given_marker"]["point"] == 0.5
    assert out["per_patient"][0]["contradicted_ids"] == ["NCT2"]


def test_score_refuses_a_ranking_with_missing_trial_text():
    rows = [{"patient_id": "a", "note": "HER2-positive", "retrieved": ["NCT9"]}]
    with pytest.raises(KeyError):
        score(rows, {}, k=1)


def test_observational_share_counts_known_types_and_reports_missing():
    types = {"NCT1": "OBSERVATIONAL", "NCT2": "INTERVENTIONAL", "NCT3": "INTERVENTIONAL",
             "NCT4": "INTERVENTIONAL", "NCT5": None}
    out = observational_share(["NCT1", "NCT2", "NCT3", "NCT4", "NCT5", "NCT6"], types)
    assert out["n"] == 4
    assert out["observational"] == 1
    assert out["share"] == 0.25
    assert out["missing"] == 2
    lo, hi = out["ci95"]
    assert 0 < lo < 0.25 < hi < 1


def test_observational_share_counts_expanded_access_as_not_observational():
    out = observational_share(["NCT1"], {"NCT1": "EXPANDED_ACCESS"})
    assert out["observational"] == 0
    assert out["by_type"] == {"EXPANDED_ACCESS": 1}


def test_score_counts_an_unmarked_note_without_a_ranking():
    """A note stating no status cannot contradict a trial, so its top-k needs no retrieval."""
    rows = [{"patient_id": "b", "note": "no marker stated", "retrieved": None}]
    out = score(rows, {}, k=25)
    assert out["surfaced"] == 25
    assert out["contradicted"] == 0
    assert out["per_patient"][0]["retrieved"] is False


def test_score_refuses_a_marked_note_without_a_ranking():
    rows = [{"patient_id": "a", "note": "HER2-positive", "retrieved": None}]
    with pytest.raises(ValueError):
        score(rows, {}, k=25)
