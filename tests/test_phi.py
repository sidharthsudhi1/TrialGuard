"""PHI refusal gate: must fire on real identifiers, never on synthetic notes."""

import pytest

from trialguard.agent.sanitize import detect_phi


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Patient SSN 123-45-6789 on file", "us_ssn"),
        ("Contact jane.doe@hospital.org for records", "email"),
        ("MRN: 0042213 admitted today", "record_number"),
        ("DOB: 1961-04-12, presents with dyspnea", "date_of_birth"),
        ("Seen on 04/12/1961 for follow-up", "full_date"),
        ("Call (415) 555-0142 to arrange", "phone"),
        ("Resides at 42 Sycamore Street", "street_address"),
        ("Portal at https://records.example.org/p/9", "url"),
        ("Patient name: Rivera, seen in clinic", "named_patient"),
    ],
)
def test_detects_each_identifier_class(text, expected):
    assert expected in detect_phi(text)


def test_returns_categories_never_the_matched_text():
    found = detect_phi("SSN 123-45-6789 and bob@x.org")
    assert found == ["email", "us_ssn"]
    # The whole point: an error message built from this cannot leak the PHI it
    # is refusing, because the value never leaves the detector.
    assert not any("123-45-6789" in f or "bob@x.org" in f for f in found)


@pytest.mark.parametrize(
    "note",
    [
        "45-year-old man with anaplastic astrocytoma of the spine",
        "Patient is a 58-year-old woman, ECOG 1, EF was 25%",
        "Stage IV NSCLC, EGFR exon 19 deletion, progressed on osimertinib 80 mg",
        "Hgb 9.2, platelets 140, creatinine 1.1, treated in 2019 and 2021",
        "Enrolled in NCT03633552 previously; ANC 1500/mm3",
    ],
)
def test_clinical_narrative_is_not_flagged(note):
    # Ages, scores, labs, doses, years and NCT ids are what these notes are made
    # of. A gate that trips on them would be switched off, which is worse than
    # no gate.
    assert detect_phi(note) == []


def test_no_false_positives_on_any_committed_cohort_note():
    """The load-bearing test: zero flags across every real cohort note.

    These are the synthetic notes the system is built to accept. If the gate
    fires on even one, it would refuse legitimate traffic in production.
    """
    from trialguard.eval.cohorts import load_patients

    flagged = []
    total = 0
    for cohort in ("sigir", "trec_2021", "trec_2022"):
        for p in load_patients(cohort):
            total += 1
            hits = detect_phi(p["description"])
            if hits:
                flagged.append((cohort, p["patient_id"], hits))
    # 59 SIGIR + 75 TREC 2021 + 50 TREC 2022. Asserted so the test cannot
    # quietly pass on an empty or truncated cohort load.
    assert total == 184, f"expected the full cohort set, saw {total}"
    assert not flagged, f"false positives on synthetic notes: {flagged[:5]}"
