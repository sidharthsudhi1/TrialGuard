"""D1 (Phase 40): build the name-refusal jobs and score model entities.

    python scripts/d1_score.py jobs    # cohort notes + three planted sets
    python scripts/d1_score.py score   # recall and false positives per model
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from trialguard.agent import sanitize
from trialguard.agent.sanitize import _rule_names
from trialguard.eval.cohorts import load_patients
from trialguard.eval.planted_names import build

JOBS = Path("data/reports/d1_jobs.json")
ENTITIES = Path("data/reports/d1_entities.json")
OUT = Path("data/reports/d1_scores.json")
COHORTS = ("sigir", "trec_2021", "trec_2022", "trec_2023")
NAMED_COHORT_NOTES = {"sigir-201515", "trec-202148"}
# Person labels per model family; everything else (dates, ages, hospitals) is
# either already a regex class or not an identifier the contract refuses.
PERSON = {"PATIENT", "STAFF", "HCW", "PER", "PERSON", "NAME", "DOCTOR", "USERNAME"}
SETS = {0: "tuning", 1: "holdout_1", 2: "holdout_2"}


def wilson(x: int, n: int, z: float = 1.96) -> list[float]:
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def jobs() -> None:
    out = [
        {"id": f"cohort|{p['patient_id']}", "note": p["description"]}
        for c in COHORTS for p in load_patients(c)
    ]
    for h, label in SETS.items():
        for i, x in enumerate(build(seed=40, holdout=h)):
            out.append({"id": f"{label}|{i}", "note": x["note"], "name": x["name"]})
    JOBS.write_text(json.dumps({"jobs": out}))
    print(f"wrote {JOBS}: {len(out)} jobs")


def _flags(ents: list[dict], threshold: float) -> bool:
    return any(e["label"].upper() in PERSON and e["score"] >= threshold for e in ents)


def score() -> None:
    jobs_by_id = {j["id"]: j for j in json.loads(JOBS.read_text())["jobs"]}
    results = json.loads(ENTITIES.read_text())
    report = {}
    for model, r in results.items():
        ents = r["entities"]
        rows = {}
        for threshold in (0.5, 0.7, 0.9):
            for with_rules in (False, True):
                def hit(jid: str, ents=ents, threshold=threshold, with_rules=with_rules) -> bool:
                    flagged = _flags(ents[jid], threshold)
                    return flagged or (with_rules and _rule_names(jobs_by_id[jid]["note"]))

                fp = sorted(
                    jid.split("|")[1] for jid in jobs_by_id
                    if jid.startswith("cohort|") and jid.split("|")[1] not in NAMED_COHORT_NOTES
                    and hit(jid)
                )
                named = sum(hit(f"cohort|{n}") for n in NAMED_COHORT_NOTES)
                recall = {}
                for label in SETS.values():
                    ids = [j for j in jobs_by_id if j.startswith(f"{label}|")]
                    k = sum(hit(j) for j in ids)
                    recall[label] = {"hit": k, "n": len(ids), "recall": round(k / len(ids), 4),
                                     "ci95": wilson(k, len(ids))}
                rows[f"t{threshold}{'+rules' if with_rules else ''}"] = {
                    "false_positive_notes": fp, "named_cohort_flagged": named, "recall": recall,
                }
        report[model] = {k: r[k] for k in ("load_s", "median_ms", "p95_ms",
                                            "median_ms_4000_chars")} | {"arms": rows}
        print(model, {k: r[k] for k in ("median_ms", "median_ms_4000_chars")})
        for arm, row in rows.items():
            rec = " ".join(f"{s}={v['recall']}" for s, v in row["recall"].items())
            print(f"  {arm:<10} fp={len(row['false_positive_notes'])} named={row['named_cohort_flagged']} {rec}")
    report["served"] = _served(jobs_by_id, results[sanitize.DEID_MODEL]["entities"])
    print("served", json.dumps(report["served"]))
    OUT.write_text(json.dumps(report, indent=1))


def _served(jobs_by_id: dict, ents: dict) -> dict:
    """The adopted detector exactly: sanitize.detect_names on the recorded entities."""
    by_note = {jobs_by_id[j]["note"]: ents[j] for j in jobs_by_id}

    def replay(text: str) -> list[dict]:
        return [{"entity_group": e["label"], **e} for e in by_note[text]]

    sanitize._deid = lambda: replay
    fp = sorted(
        j.split("|")[1] for j in jobs_by_id
        if j.startswith("cohort|") and j.split("|")[1] not in NAMED_COHORT_NOTES
        and sanitize.detect_names(jobs_by_id[j]["note"])
    )
    named = sum(sanitize.detect_names(jobs_by_id[f"cohort|{n}"]["note"]) for n in NAMED_COHORT_NOTES)
    recall = {}
    for label in SETS.values():
        ids = [j for j in jobs_by_id if j.startswith(f"{label}|")]
        k = sum(sanitize.detect_names(jobs_by_id[j]["note"]) for j in ids)
        recall[label] = {"hit": k, "n": len(ids), "recall": round(k / len(ids), 4),
                         "ci95": wilson(k, len(ids))}
    return {"model": sanitize.DEID_MODEL, "threshold": sanitize.DEID_THRESHOLD,
            "false_positive_notes": fp, "named_cohort_flagged": named, "recall": recall}


if __name__ == "__main__":
    step = argparse.ArgumentParser()
    step.add_argument("step", choices=("jobs", "score"))
    {"jobs": jobs, "score": score}[step.parse_args().step]()
