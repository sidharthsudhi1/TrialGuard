"""Parse the TREC CT 2023 download into the 2021/2022 eval file shapes.

Topics are synthetic questionnaires. Qrels are the NIST graded file. The corpus
is the May 2023 ClinicalTrials.gov XML snapshot, streamed out of the zips.
No embedding.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

EVAL_DIR = Path("data/eval/trec_2023")
ZIPS = tuple(
    EVAL_DIR / f"ClinicalTrials.2023-05-08.trials{i}.zip" for i in range(6)
)

_WS = re.compile(r"[ \t]+")


def patient_id(number: str) -> str:
    """Match the trec-20211 id shape: year and topic number, no extra hyphen."""
    return f"trec-2023{number}"


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    out: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if out and not blank:
                out.append("")
            blank = True
            continue
        blank = False
        out.append(line)
    return "\n".join(out).strip()


def render_topic(template: str, fields: list[tuple[str, str]]) -> str:
    """Questionnaire fields as a note. Empty answers are omitted."""
    lines = [f"Disorder: {template}"] if template else []
    for name, value in fields:
        value = value.strip()
        if not value:
            continue
        label = name[:1].upper() + name[1:] if name else name
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def parse_topics(path: Path) -> list[dict]:
    # NIST snapshot, parsed offline. ElementTree does not fetch external entities.
    root = ET.parse(path).getroot()  # noqa: S314
    patients = []
    for topic in root.findall("topic"):
        number = (topic.get("number") or "").strip()
        template = (topic.get("template") or "").strip()
        fields = [
            ((field.get("name") or "").strip(), "".join(field.itertext()))
            for field in topic.findall("field")
        ]
        patients.append({
            "_id": patient_id(number),
            "text": render_topic(template, fields),
            "metadata": {"template": template, "topic": number},
        })
    return patients


def parse_qrels(path: Path) -> list[tuple[str, str, str]]:
    """NIST qrels are `topic iteration nct score`, space-separated, no header."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        topic, _iteration, nct, score = line.split()
        rows.append((patient_id(topic), nct, score))
    return rows


def _child_text(root: ET.Element, path: str) -> str:
    node = root.find(path)
    if node is None:
        return ""
    return _clean("".join(node.itertext()))


def trial_from_xml(raw: bytes) -> dict | None:
    try:
        root = ET.fromstring(raw)  # noqa: S314  same snapshot as parse_topics
    except ET.ParseError:
        return None
    nct = _child_text(root, "id_info/nct_id")
    if not nct:
        return None
    conditions = [_clean("".join(node.itertext())) for node in root.findall("condition")]
    conditions = [c for c in conditions if c]
    return {
        "_id": nct,
        "title": _child_text(root, "brief_title"),
        "text": _child_text(root, "eligibility/criteria/textblock"),
        "metadata": {
            "overall_status": _child_text(root, "overall_status"),
            "phase": _child_text(root, "phase"),
            "minimum_age": _child_text(root, "eligibility/minimum_age"),
            "maximum_age": _child_text(root, "eligibility/maximum_age"),
            "gender": _child_text(root, "eligibility/gender"),
            "healthy_volunteers": _child_text(root, "eligibility/healthy_volunteers"),
            "conditions": conditions,
        },
    }


def iter_zip_trials(zips: tuple[Path, ...] = ZIPS):
    seen: set[str] = set()
    for path in zips:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.endswith(".xml"):
                    continue
                trial = trial_from_xml(archive.read(info))
                if trial is None or trial["_id"] in seen:
                    continue
                seen.add(trial["_id"])
                yield trial


def write_queries(patients: list[dict], dest: Path) -> None:
    with dest.open("w") as handle:
        for patient in patients:
            handle.write(json.dumps(patient, ensure_ascii=False) + "\n")


def write_qrels(rows: list[tuple[str, str, str]], dest: Path) -> None:
    with dest.open("w") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for query_id, nct, score in rows:
            handle.write(f"{query_id}\t{nct}\t{score}\n")


def write_corpus(dest: Path, zips: tuple[Path, ...] = ZIPS) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".jsonl.partial")
    n = empty = 0
    with tmp.open("w") as handle:
        for trial in iter_zip_trials(zips):
            if not trial["text"]:
                empty += 1
            handle.write(json.dumps(trial, ensure_ascii=False) + "\n")
            n += 1
            if n % 20000 == 0:
                print(f"  {n} trials", flush=True)
    tmp.replace(dest)
    return {"trials": n, "empty_eligibility": empty}


def prepare(dest_dir: Path = EVAL_DIR) -> dict:
    patients = parse_topics(dest_dir / "topics2023.xml")
    qrels = parse_qrels(dest_dir / "qrels2023.txt")
    write_queries(patients, dest_dir / "queries.jsonl")
    write_qrels(qrels, dest_dir / "qrels.tsv")
    corpus = write_corpus(dest_dir / "trec_2023_corpus.jsonl")
    stats = {
        "patients": len(patients),
        "qrels": len(qrels),
        **corpus,
    }
    (dest_dir / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    return stats
