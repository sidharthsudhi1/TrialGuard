"""C3: Llama-3.1-8B keywords versus the cached 70B lists.

Non-inferior: paired delta (8B minus 70B) CI lower bound >= -0.01 at recall@100
and @200 on both TREC cohorts, and p50 8B keyword latency <= 2s.
Index: TG_INDEX_EXCLUSION=0, the cached build.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from trialguard.config import settings
from trialguard.eval.cohorts import load_patients
from trialguard.eval.end_to_end import _gold_by_patient
from trialguard.eval.file_index import get_index
from trialguard.eval.retrieval_metrics import recall_at_k
from trialguard.eval.significance import paired_bootstrap_ci
from trialguard.retrieval.query_transform import generate_keywords

OUT = Path("data/reports/c3_keyword_model.json")
SMALL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
BIG = settings.deepinfra_model


def _patients(cohort: str) -> list[dict]:
    gold = _gold_by_patient(cohort)
    idx = get_index(cohort)
    corpus = set(idx.corpus_ids())
    rows = []
    for patient in load_patients(cohort):
        eligible = {
            nct for nct, lab in gold.get(patient["patient_id"], {}).items()
            if lab == "eligible" and nct in corpus
        }
        if eligible:
            rows.append({**patient, "eligible": eligible})
    return rows


def _arm(cohort: str, model: str, time_it: bool) -> dict:
    settings.deepinfra_model = model
    idx = get_index(cohort)
    rec100, rec200, times = [], [], []
    fallbacks = 0
    for patient in _patients(cohort):
        note = patient["description"]
        t0 = time.perf_counter()
        keywords = generate_keywords(note)
        if keywords == [note]:
            fallbacks += 1
        if time_it:
            times.append((time.perf_counter() - t0) * 1000)
        hits = [nct for nct, _ in idx.search(note, top_k=200, use_keywords=True)]
        rec100.append(recall_at_k(hits, patient["eligible"], 100))
        rec200.append(recall_at_k(hits, patient["eligible"], 200))
    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    times_sorted = sorted(times)
    p50 = times_sorted[len(times_sorted) // 2] if times_sorted else None
    return {
        "n": len(rec100),
        "recall@100": mean(rec100),
        "recall@200": mean(rec200),
        "keyword_p50_ms": None if p50 is None else round(p50, 1),
        "raw_note_fallbacks": fallbacks,
        "_r100": rec100,
        "_r200": rec200,
    }


def main() -> None:
    os.environ["TG_INDEX_EXCLUSION"] = "0"
    report = {"small": SMALL, "big": BIG, "cohorts": {}}
    adopt = True
    for cohort in ("trec_2021", "trec_2022"):
        big = _arm(cohort, BIG, time_it=False)
        small = _arm(cohort, SMALL, time_it=True)
        deltas = {}
        for depth, key in ((100, "_r100"), (200, "_r200")):
            ci = paired_bootstrap_ci(big[key], small[key], seed=0)
            lo = ci["ci"][0]
            deltas[f"recall@{depth}"] = {
                "delta": ci["point"],
                "ci": ci["ci"],
                "noninferior": lo is not None and lo >= -0.01,
            }
            adopt = adopt and deltas[f"recall@{depth}"]["noninferior"]
        fast = small["keyword_p50_ms"] is not None and small["keyword_p50_ms"] <= 2000
        adopt = adopt and fast
        for arm in (big, small):
            arm.pop("_r100")
            arm.pop("_r200")
        report["cohorts"][cohort] = {"70b": big, "8b": small, "delta_8b_minus_70b": deltas, "p50_under_2s": fast}
    report["adopt"] = adopt
    settings.deepinfra_model = BIG
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
