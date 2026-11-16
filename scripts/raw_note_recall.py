"""C2: recall@100 of a raw note versus keyword retrieval, same FileIndex.

The final keyword ranking is what gets assessed. This only reports how much of
the gold the provisional list already holds.
"""

from __future__ import annotations

import json
from pathlib import Path

from trialguard.eval.cohorts import load_patients
from trialguard.eval.end_to_end import _gold_by_patient
from trialguard.eval.file_index import get_index
from trialguard.eval.retrieval_metrics import recall_at_k

OUT = Path("data/reports/c2_raw_note_recall.json")


def _recall(cohort: str, k: int) -> dict:
    gold = _gold_by_patient(cohort)
    idx = get_index(cohort)
    corpus = set(idx.corpus_ids())
    raw_vals, kw_vals = [], []
    n = 0
    for patient in load_patients(cohort):
        labels = gold.get(patient["patient_id"], {})
        eligible = {nct for nct, lab in labels.items() if lab == "eligible" and nct in corpus}
        if not eligible:
            continue
        note = patient["description"]
        raw = [nct for nct, _ in idx.search(note, top_k=k, use_keywords=False)]
        kw = [nct for nct, _ in idx.search(note, top_k=k, use_keywords=True)]
        raw_vals.append(recall_at_k(raw, eligible, k))
        kw_vals.append(recall_at_k(kw, eligible, k))
        n += 1
    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    return {"n": n, "raw_recall": mean(raw_vals), "keyword_recall": mean(kw_vals)}


def main() -> None:
    report = {c: _recall(c, 100) for c in ("trec_2021", "trec_2022", "sigir")}
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
