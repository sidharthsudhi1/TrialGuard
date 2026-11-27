"""PHI refusal gate: must fire on real identifiers, never on synthetic notes."""

import json
import os
from pathlib import Path

import pytest

from trialguard.agent import sanitize
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


# --- Phase 40: unlabelled names (de-id model + context rules) -----------------
# The model is obi/deid_bert_i2b2. Tests replay its outputs recorded on Modal
# (data/reports/d1_entities.json) so they run without loading it;
# test_live_model_matches_the_recording checks the recording in CI.

RECORDED = Path("data/reports/d1_entities.json")
JOBS = Path("data/reports/d1_jobs.json")


class _Fake:
    def __init__(self, by_text):
        self.by_text = by_text

    def __call__(self, text):
        return [
            {"entity_group": e["label"], "score": e["score"], "start": e["start"],
             "end": e["end"], "word": e["word"]}
            for e in self.by_text.get(text, [])
        ]


@pytest.fixture
def no_model(monkeypatch):
    monkeypatch.setattr(sanitize, "_deid", lambda: _Fake({}))


@pytest.fixture(scope="module")
def recorded():
    entities = json.loads(RECORDED.read_text())[sanitize.DEID_MODEL]["entities"]
    jobs = json.loads(JOBS.read_text())["jobs"]
    return {j["note"]: entities[j["id"]] for j in jobs}, jobs


@pytest.fixture
def replay(monkeypatch, recorded):
    monkeypatch.setattr(sanitize, "_deid", lambda: _Fake(recorded[0]))
    return recorded[1]


@pytest.mark.parametrize(
    "text",
    [
        "Mr. Okafor was seen today with chest pain.",
        "Ms. Nguyen declined further imaging.",
        "Her daughter Amara provides most of the history.",
        "Lives with his wife, Mary Thompson, who drives him to clinic.",
        "Maria Lopez is a 45-year-old woman with stage IV NSCLC.",
        "Karen is a 72-year-old woman with hypertension.",
        "Referred by her sister Hope Sharma after the diagnosis.",
    ],
)
def test_context_rules_catch_names_without_the_model(no_model, text):
    assert "person_name" in detect_phi(text)


@pytest.mark.parametrize(
    "text",
    [
        "Tenderness and positive Murphy sign on examination.",
        "Urinary retention s/p Foley catheter.",
        "History of Burkitt lymphoma and Hodgkin lymphoma.",
        "Long-standing Crohn's disease, on infliximab.",
        "Patient is a 45-year-old man with dyspnea.",
        "Wegener's granulomatosis treated with rituximab.",
        "Daughter reported increased agitation for 2-3 days.",
        "Mother has breast cancer; father had colon cancer.",
        "Started Lasix 40 mg and Vanco po for Cdiff.",
        "A 61-year-old woman, BRCA1 carrier, ECOG 1.",
        "Moved from Colorado to California last year.",
        "Down syndrome with Graves disease.",
    ],
)
def test_context_rules_leave_eponyms_drugs_and_relatives_alone(no_model, text):
    assert "person_name" not in detect_phi(text)


def _with(monkeypatch, text, label, word, score):
    start = text.index(word)
    ent = {"label": label, "score": score, "start": start, "end": start + len(word), "word": word}
    monkeypatch.setattr(sanitize, "_deid", lambda: _Fake({text: [ent]}))


def test_model_person_span_refuses(monkeypatch):
    text = "Spoke with Wei by phone about the results."
    _with(monkeypatch, text, "PATIENT", "Wei", 0.91)
    assert detect_phi(text) == ["person_name"]


@pytest.mark.parametrize("label", ["DATE", "AGE", "HOSP", "LOC"])
def test_model_non_person_labels_do_not_refuse(monkeypatch, label):
    text = "Spoke with Wei by phone about the results."
    _with(monkeypatch, text, label, "Wei", 0.99)
    assert "person_name" not in detect_phi(text)


def test_model_score_below_threshold_does_not_refuse(monkeypatch):
    text = "Spoke with Wei by phone about the results."
    _with(monkeypatch, text, "PATIENT", "Wei", sanitize.DEID_THRESHOLD - 0.01)
    assert "person_name" not in detect_phi(text)


@pytest.mark.parametrize(
    "text,word",
    [
        ("The Kayser-Fleischer' ring was detected in both eyes.", "Kayser"),
        ("Ranked as mild, Hoehn-Yahr Stage I.", "Hoehn"),
        ("Positive Murphy sign.", "Murphy"),
    ],
)
def test_model_span_followed_by_an_eponym_suffix_does_not_refuse(monkeypatch, text, word):
    # The model's span can stop mid-word ("Kayser" of "Kayser-Fleischer'"),
    # so the suffix check starts after the whole hyphenated word.
    _with(monkeypatch, text, "PATIENT", word, 0.83)
    assert "person_name" not in detect_phi(text)


# These two cohort notes open with the patient's name ("Karen is a 72-year-old",
# "Fernandez is a 41 year man"). Unlabelled names are exactly what this pass
# exists to refuse, so flagging them is a true positive, not a false one.
NAMED_COHORT_NOTES = {"sigir-201515", "trec-202148"}


def test_no_false_positives_on_any_committed_cohort_note(replay):
    """The load-bearing test: zero flags across every real cohort note.

    These are the synthetic notes the system is built to accept. If the gate
    fires on even one, it would refuse legitimate traffic in production.
    """
    from trialguard.eval.cohorts import load_patients

    flagged = []
    total = 0
    named = set()
    for cohort in ("sigir", "trec_2021", "trec_2022"):
        for p in load_patients(cohort):
            total += 1
            hits = detect_phi(p["description"])
            if p["patient_id"] in NAMED_COHORT_NOTES:
                assert hits == ["person_name"], (p["patient_id"], hits)
                named.add(p["patient_id"])
            elif hits:
                flagged.append((cohort, p["patient_id"], hits))
    # 59 SIGIR + 75 TREC 2021 + 50 TREC 2022. Asserted so the test cannot
    # quietly pass on an empty or truncated cohort load.
    assert total == 184, f"expected the full cohort set, saw {total}"
    assert not flagged, f"false positives on synthetic notes: {flagged[:5]}"
    assert named == NAMED_COHORT_NOTES


def test_no_false_positives_on_trec_2023_notes(replay):
    from trialguard.eval.cohorts import load_patients

    flagged = [(p["patient_id"], detect_phi(p["description"]))
               for p in load_patients("trec_2023") if detect_phi(p["description"])]
    assert not flagged


def test_name_recall_on_the_untouched_holdout(replay):
    """Gate (2) on holdout 2, which no rule or threshold was chosen on."""
    held = [j for j in replay if j["id"].startswith("holdout_2|")]
    hits = sum("person_name" in detect_phi(j["note"]) for j in held)
    assert len(held) >= 200
    assert hits / len(held) >= 0.95, f"{hits}/{len(held)}"


@pytest.mark.skipif(os.environ.get("TG_DEID_LIVE") != "1", reason="loads the de-id model")
def test_live_model_matches_the_recording(recorded):
    by_text, jobs = recorded
    sample = [j for j in jobs if j["id"].startswith(("cohort|", "holdout_2|"))][::12]
    for job in sample:
        live = {(e["entity_group"], e["start"], e["end"]) for e in sanitize._deid()(job["note"])}
        want = {(e["label"], e["start"], e["end"]) for e in by_text[job["note"]]}
        assert live == want, job["id"]


def test_warm_models_loads_the_deid_model(monkeypatch):
    # The first refused or accepted note must not pay the model load.
    from trialguard import demo

    calls = []
    monkeypatch.setattr("trialguard.ingestion.embed.embed_text", lambda *a, **k: None)
    monkeypatch.setattr(sanitize, "_deid", lambda: calls.append(1) or (lambda text: []))
    demo.warm_models()
    assert calls
