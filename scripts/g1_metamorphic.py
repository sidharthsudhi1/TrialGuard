"""G1: one directional and one invariance edit on each human WS-5b item.

Scope was fixed on 2026-11-10 before any mutated note was assessed. Directional
success is a flipped verdict or cannot_determine. Invariance success is the
same verdict. This does not change a served default.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trialguard.eval.cohorts import load_patients
from trialguard.eval.metamorphic import directional, invariance
from trialguard.eval.nli_probe import load_gold
from trialguard.verify.grounding import normalize

OUT = Path("data/reports/g1_metamorphic.json")
GOLD = Path("data/reports/e7_combined_gold.jsonl")


def _criterion_verdict(state: dict, criterion: str) -> str | None:
    want = normalize(criterion)
    for assessment in state.get("assessments") or []:
        if normalize(str(assessment.get("criterion", ""))) == want:
            return assessment.get("verdict")
    return None


def _one(job: dict) -> dict:
    from trialguard.agent.graph import assess
    from trialguard.agent.schema import build_typed_criteria
    from trialguard.eval.file_index import _load_sigir_trials, _load_trec_trials
    from trialguard.ingestion.normalise import normalise_trial

    cohort, nct = job["cohort"], job["nct_id"]
    raw = _load_sigir_trials() if cohort == "sigir" else _load_trec_trials(cohort, {nct})
    trial = next((normalise_trial(t) for t in raw if t["nct_id"] == nct), None)
    if trial is None:
        return {**job, "error": "trial_missing"}
    criteria, truncated = build_typed_criteria(trial)
    try:
        state = assess(
            job["note"],
            nct,
            criteria,
            trial.get("eligibility_raw", ""),
            max_retries=2,
            criteria_truncated=truncated,
        )
    except Exception as exc:
        return {**job, "error": type(exc).__name__}
    got = _criterion_verdict(state, job["criterion"])
    return {**job, "new_verdict": got, "error": None}


def main() -> None:
    notes = {
        (c, p["patient_id"]): p["description"]
        for c in ("sigir", "trec_2021", "trec_2022")
        for p in load_patients(c)
    }
    items = [i for i in load_gold(GOLD) if i["label_source"] == "human"]
    jobs = []
    skipped = {"directional": 0, "invariance": 0}
    for item in items:
        note = notes.get((item["cohort"], item["patient_id"]), "")
        for kind, edit in (("directional", directional), ("invariance", invariance)):
            edited = edit(note, item.get("quote") or "")
            if not edited:
                skipped[kind] += 1
                continue
            jobs.append(
                {
                    "kind": kind,
                    "cohort": item["cohort"],
                    "patient_id": item["patient_id"],
                    "nct_id": item["nct_id"],
                    "criterion": item["criterion"],
                    "old_verdict": item["verdict"],
                    "note": edited,
                }
            )
    with ThreadPoolExecutor(max_workers=6) as pool:
        done = list(pool.map(_one, jobs))
    summary = {}
    for kind in ("directional", "invariance"):
        rows = [r for r in done if r["kind"] == kind and not r.get("error")]
        if kind == "directional":
            ok = [
                r for r in rows
                if r.get("new_verdict") in ("cannot_determine", "unverifiable")
                or (r.get("new_verdict") and r["new_verdict"] != r["old_verdict"])
            ]
        else:
            ok = [r for r in rows if r.get("new_verdict") == r["old_verdict"]]
        summary[kind] = {
            "attempted": sum(1 for r in done if r["kind"] == kind),
            "scored": len(rows),
            "success": len(ok),
            "rate": round(len(ok) / len(rows), 4) if rows else None,
            "skipped_unlocatable": skipped[kind],
            "errors": sum(1 for r in done if r["kind"] == kind and r.get("error")),
        }
    report = {"n_items": len(items), "summary": summary}
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
