"""V3: target-trial recovery on NIH-TrialBench, local only.

The corpus is the per-case candidate list published with the benchmark, not
ctgov_live. Notes stay off the public API: several summaries contain calendar
dates, which detect_phi refuses. License: NLM public domain, checked 2026-11-10.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("data/cache/trialbench/data/nih_trialbench")
OUT = Path("data/reports/v3_trialbench_recovery.json")


def _ids(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        return [str(v) for v in json.loads(value.replace("'", '"'))]
    return []


def _cases() -> list[dict]:
    rows = []
    with (ROOT / "benchmark_cases.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                case = json.loads(line)
                case["candidate_trial_ids"] = _ids(case.get("candidate_trial_ids"))
                rows.append(case)
    return rows


def _trial(nct: str) -> dict | None:
    path = ROOT / "trials" / f"{nct}.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return {
        "nct_id": raw.get("nctid") or nct,
        "title": raw.get("title") or "",
        "eligibility_raw": raw.get("raw_criteria") or "",
        "min_age": raw.get("minimum_age") or "",
        "max_age": raw.get("maximum_age") or "",
        "sex": raw.get("sex") or "",
    }


def _recovery(ranked: list[str], target: str, ks: tuple[int, ...]) -> dict[str, int]:
    return {f"hit@{k}": int(target in ranked[:k]) for k in ks}


def main() -> None:
    from trialguard.eval.file_index import FileIndex

    cases = _cases()
    needed: set[str] = set()
    for case in cases:
        needed.add(case["target_trial_id"])
        needed.update(case.get("candidate_trial_ids") or [])
    trials = []
    for nct in sorted(needed):
        trial = _trial(nct)
        if trial:
            trials.append(trial)
    idx = FileIndex("trialbench")
    idx.build(trials)
    depth = len(idx.corpus_ids())
    ks = (1, 5, 10, 50)
    arms = {"raw": [], "keyword": []}
    missing_target = 0
    for case in cases:
        target = case["target_trial_id"]
        cands = set(case.get("candidate_trial_ids") or [])
        cands.add(target)
        if target not in idx.corpus_ids():
            missing_target += 1
            continue
        note = case["patient_summary"]
        for arm, use_keywords in (("raw", False), ("keyword", True)):
            hits = idx.search(
                note,
                top_k=depth,
                dense_pool=depth,
                bm25_pool=depth,
                use_keywords=use_keywords,
            )
            ranked = [nct for nct, _ in hits if nct in cands]
            arms[arm].append(_recovery(ranked, target, ks))
    report = {"n_cases": len(cases), "n_trials_loaded": len(trials), "missing_target": missing_target}
    for arm, rows in arms.items():
        report[arm] = {
            "n": len(rows),
            **{
                k: round(sum(r[k] for r in rows) / len(rows), 4) if rows else None
                for k in (f"hit@{x}" for x in ks)
            },
        }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
