"""Download and parse SIGIR 2016 + TREC CT 2021/2022 gold eval cohorts.

Source: https://github.com/ncbi-nlp/TrialGPT (dataset/)

qrels format (TSV): query-id  corpus-id  score
Scores: 0=irrelevant, 1=excluded, 2=eligible
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import httpx

EVAL_DIR = Path("data/eval")

# Non-negotiable: no real patient data, ever. Patient notes may only come from
# these vetted, published, synthetic cohorts. Every patient/label loader validates
# against this allowlist so a real-PHI source cannot be loaded even by mistake.
SYNTHETIC_COHORTS = frozenset({"sigir", "trec_2021", "trec_2022"})


def _require_synthetic(cohort: str) -> None:
    if cohort not in SYNTHETIC_COHORTS:
        raise ValueError(
            f"cohort {cohort!r} is not an allowed synthetic cohort "
            f"{sorted(SYNTHETIC_COHORTS)}; loading non-synthetic patient data is forbidden"
        )


COHORTS = {
    "sigir": {
        "queries": "https://raw.githubusercontent.com/ncbi-nlp/TrialGPT/main/dataset/sigir/queries.jsonl",
        "qrels": "https://raw.githubusercontent.com/ncbi-nlp/TrialGPT/main/dataset/sigir/qrels/test.tsv",
    },
    "trec_2021": {
        "queries": "https://raw.githubusercontent.com/ncbi-nlp/TrialGPT/main/dataset/trec_2021/queries.jsonl",
        "qrels": "https://raw.githubusercontent.com/ncbi-nlp/TrialGPT/main/dataset/trec_2021/qrels/test.tsv",
    },
    "trec_2022": {
        "queries": "https://raw.githubusercontent.com/ncbi-nlp/TrialGPT/main/dataset/trec_2022/queries.jsonl",
        "qrels": "https://raw.githubusercontent.com/ncbi-nlp/TrialGPT/main/dataset/trec_2022/qrels/test.tsv",
    },
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"Downloading {dest.name}...")
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)


def download_cohorts() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for cohort, urls in COHORTS.items():
        cohort_dir = EVAL_DIR / cohort
        cohort_dir.mkdir(exist_ok=True)
        for name, url in urls.items():
            ext = url.rsplit(".", 1)[-1]
            _download(url, cohort_dir / f"{name}.{ext}")
    print("All cohort files downloaded.")


def load_patients(cohort: str) -> list[dict]:
    """Return list of patient dicts: patient_id, description, cohort."""
    _require_synthetic(cohort)
    path = EVAL_DIR / cohort / "queries.jsonl"
    patients = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            patients.append({
                "patient_id": str(obj.get("_id") or obj.get("id", "")),
                "cohort": cohort,
                "description": obj.get("text", ""),
                "raw": obj,
            })
    return patients


def load_labels(cohort: str) -> list[dict]:
    """Return list of label dicts: patient_id, nct_id, label, cohort.

    qrels TSV columns: query-id  corpus-id  score
    Scores: 0=irrelevant, 1=excluded, 2=eligible
    """
    _require_synthetic(cohort)
    path = EVAL_DIR / cohort / "qrels.tsv"
    score_map = {"0": "irrelevant", "1": "excluded", "2": "eligible"}
    labels = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            labels.append({
                "patient_id": row["query-id"],
                "nct_id": row["corpus-id"],
                "cohort": cohort,
                "label": score_map.get(str(row["score"]).strip(), "irrelevant"),
            })
    return labels
