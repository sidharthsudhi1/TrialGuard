from xml.etree.ElementTree import Element, SubElement, tostring

from trialguard.eval.trec_2023 import (
    parse_qrels,
    parse_topics,
    patient_id,
    render_topic,
    trial_from_xml,
)


def test_topic_note_skips_blank_fields(tmp_path):
    path = tmp_path / "topics.xml"
    path.write_text(
        """<topics>
          <topic number="1" template="glaucoma">
            <field name="definitive diagnosis">POAG</field>
            <field name="intraocular pressure"></field>
            <field name="visual acuity">20/50</field>
          </topic>
        </topics>"""
    )
    patients = parse_topics(path)
    assert patients[0]["_id"] == "trec-20231"
    assert patients[0]["text"] == (
        "Disorder: glaucoma\n"
        "Definitive diagnosis: POAG\n"
        "Visual acuity: 20/50"
    )
    assert "intraocular" not in patients[0]["text"]


def test_qrels_gain_the_cohort_id_and_a_header(tmp_path):
    path = tmp_path / "qrels.txt"
    path.write_text("1 0 NCT00000119 0\n1 0 NCT00000125 2\n")
    assert parse_qrels(path) == [
        ("trec-20231", "NCT00000119", "0"),
        ("trec-20231", "NCT00000125", "2"),
    ]
    assert patient_id("40") == "trec-202340"


def test_xml_trial_keeps_criteria_lines():
    root = Element("clinical_study")
    id_info = SubElement(root, "id_info")
    SubElement(id_info, "nct_id").text = "NCT00704457"
    SubElement(root, "brief_title").text = "Sacral neuromodulation"
    elig = SubElement(root, "eligibility")
    criteria = SubElement(elig, "criteria")
    block = SubElement(criteria, "textblock")
    block.text = "Inclusion Criteria:\n\n          -  Adult patients\n\n        Exclusion Criteria:\n\n          -  Male"
    trial = trial_from_xml(tostring(root))
    assert trial["_id"] == "NCT00704457"
    assert trial["title"] == "Sacral neuromodulation"
    assert "Inclusion Criteria:" in trial["text"]
    assert "Adult patients" in trial["text"]
    assert "Exclusion Criteria:" in trial["text"]


def test_render_topic_with_no_template():
    assert render_topic("", [("age", "39")]) == "Age: 39"
