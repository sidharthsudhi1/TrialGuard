"""R4: 8B analyst on the deep-fusion tail. 70B only if the drop bar holds.

Isolated ledger, hard stop $3. Resumes from data/reports/r4_calls.jsonl.

    python scripts/r4_cascade.py
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from trialguard.eval.cascade import (
    DROP_LIMIT,
    GAIN_FLOOR_USD,
    drop_kills,
    drop_rate,
    tail_depth,
)

REPORTS = Path("data/reports")
PLAN = REPORTS / "r4_plan.json"
CALLS = REPORTS / "r4_calls.jsonl"
OUT = REPORTS / "r4_cascade.json"
LEDGER = Path("data/cache/ledgers/r4.json")
SMALL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
BIG = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
POOLS = (
    ("trec_2021", Path("data/cache/ltr/trec_2021_medcpt_excl.json")),
    ("trec_2022", Path("data/cache/ltr/trec_2022_medcpt_noexcl.json")),
)


def _patients() -> list[dict]:
    from trialguard.eval.cohorts import load_patients
    from trialguard.eval.end_to_end import _gold_by_patient

    rows = []
    for cohort, path in POOLS:
        notes = {p["patient_id"]: p["description"] for p in load_patients(cohort)}
        gold = _gold_by_patient(cohort)
        for raw in json.loads(path.read_text()):
            rows.append(
                {
                    "cohort": cohort,
                    "patient_id": raw["patient_id"],
                    "note": notes[raw["patient_id"]],
                    "candidates": raw["candidates"],
                    "gold": gold.get(raw["patient_id"], {}),
                }
            )
    return rows


def _trials(patients: list[dict], depth: int) -> dict[str, dict]:
    from trialguard.eval.file_index import _load_trec_trials
    from trialguard.ingestion.normalise import normalise_trial

    by_cohort: dict[str, set[str]] = {}
    for patient in patients:
        by_cohort.setdefault(patient["cohort"], set()).update(
            patient["candidates"][100 : 100 + depth]
        )
    found: dict[str, dict] = {}
    for cohort, keep in by_cohort.items():
        for trial in _load_trec_trials(cohort, keep):
            found.setdefault(cohort, {})[trial["nct_id"]] = normalise_trial(trial)
    return found


def _load_calls(model: str) -> dict[tuple[str, str], dict]:
    found: dict[tuple[str, str], dict] = {}
    if not CALLS.exists():
        return found
    for line in CALLS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("model") == model:
            found[(row["patient_id"], row["nct_id"])] = row
    return found


def _append(row: dict) -> None:
    CALLS.parent.mkdir(parents=True, exist_ok=True)
    with CALLS.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def _assess(note: str, nct: str, trial: dict) -> dict:
    from trialguard.agent.graph import assess
    from trialguard.agent.schema import build_typed_criteria

    criteria, truncated = build_typed_criteria(trial)
    if not criteria:
        return {"tier": None, "verdict": None, "error": "no criteria"}
    state = assess(
        note,
        nct,
        criteria,
        trial.get("eligibility_raw", ""),
        max_retries=2,
        criteria_truncated=truncated,
    )
    return {
        "tier": state.get("trial_tier"),
        "verdict": state.get("trial_verdict"),
        "error": None,
    }


def _run(work: list[dict], trials: dict[str, dict], model: str) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from trialguard.agent.ratelimit import BudgetExhausted
    from trialguard.llm.cost import active_ledger

    ledger = active_ledger()
    stop = threading.Event()
    lock = threading.Lock()

    def one(item: dict) -> dict:
        base = {
            "cohort": item["cohort"],
            "patient_id": item["patient_id"],
            "nct_id": item["nct_id"],
            "rank": item["rank"],
            "gold": item["gold"],
            "model": model,
        }
        if stop.is_set() or ledger.exhausted():
            stop.set()
            return {**base, "tier": None, "verdict": None, "error": "budget"}
        trial = trials.get(item["cohort"], {}).get(item["nct_id"])
        if trial is None:
            return {**base, "tier": None, "verdict": None, "error": "missing trial"}
        last = "error"
        for _ in range(3):
            try:
                return {**base, **_assess(item["note"], item["nct_id"], trial)}
            except BudgetExhausted:
                stop.set()
                return {**base, "tier": None, "verdict": None, "error": "budget"}
            except Exception as exc:
                last = type(exc).__name__
                time.sleep(2)
        return {**base, "tier": None, "verdict": None, "error": last}

    workers = max(1, int(os.environ.get("TG_EVAL_WORKERS", "8")))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, item) for item in work]
        for future in as_completed(futures):
            row = future.result()
            with lock:
                _append(row)
            done += 1
            if done % 25 == 0 or done == len(work):
                print(
                    f"{model} {done}/{len(work)} ${ledger.spent_usd():.4f}",
                    flush=True,
                )


def _tail_work(patients: list[dict], depth: int, have: dict, model: str) -> list[dict]:
    work = []
    for patient in patients:
        for offset, nct in enumerate(patient["candidates"][100 : 100 + depth]):
            if have.get((patient["patient_id"], nct), {}).get("tier"):
                continue
            label = patient["gold"].get(nct, "")
            work.append(
                {
                    "cohort": patient["cohort"],
                    "patient_id": patient["patient_id"],
                    "note": patient["note"],
                    "nct_id": nct,
                    "rank": 101 + offset,
                    "gold": label,
                }
            )
    print(f"{model} queue {len(work)}", flush=True)
    return work


def _complete(patients: list[dict], depth: int, have: dict) -> bool:
    for patient in patients:
        for nct in patient["candidates"][100 : 100 + depth]:
            if not have.get((patient["patient_id"], nct), {}).get("tier"):
                return False
    return True


def _arm_count(patients: list[dict], depth: int, tiers: dict, call_usd: float, remaining: float) -> int:
    """Largest per-arm count whose union still fits the remaining dollars."""
    if remaining < GAIN_FLOOR_USD or call_usd <= 0:
        return 0
    spare = remaining - 0.05
    best = 0
    lo, hi = 1, depth
    while lo <= hi:
        mid = (lo + hi) // 2
        calls = 0
        for patient in patients:
            tail = patient["candidates"][100 : 100 + depth]
            naive = tail[:mid]
            kept = [
                nct
                for nct in tail
                if tiers.get((patient["patient_id"], nct), {}).get("tier") != "excluded"
            ][:mid]
            calls += len(set(naive) | set(kept))
        if calls * call_usd * 1.3 <= spare:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _hits(ncts: list[str], tiers: dict, pid: str, gold: dict) -> int:
    eligible = {nct for nct, label in gold.items() if label == "eligible"}
    return sum(
        1
        for nct in ncts
        if nct in eligible and tiers.get((pid, nct), {}).get("tier") not in (None, "excluded")
    )


def _calibrate(patient: dict, trials: dict, ledger) -> float:
    nct = patient["candidates"][100]
    before = ledger.spent_usd()
    _run(
        [
            {
                "cohort": patient["cohort"],
                "patient_id": patient["patient_id"],
                "note": patient["note"],
                "nct_id": nct,
                "rank": 101,
                "gold": patient["gold"].get(nct, ""),
            }
        ],
        trials,
        SMALL,
    )
    cost = ledger.spent_usd() - before
    if cost <= 0:
        raise SystemExit("calibration call was cached and the plan file is missing")
    return cost


def main() -> None:
    from trialguard.config import settings
    from trialguard.eval.significance import paired_bootstrap_ci, wilcoxon_paired
    from trialguard.eval.spend import isolated_ledger

    os.environ["TG_PROMPT_VERSION"] = "v4"
    os.environ.setdefault("TG_EVAL_WORKERS", "8")
    os.environ["TG_LLM_TIMEOUT"] = "300"
    settings.llm_provider = "deepinfra"
    ledger = isolated_ledger(LEDGER, 3.0)
    patients = _patients()
    if PLAN.exists():
        plan = json.loads(PLAN.read_text())
        depth = int(plan["depth"])
    else:
        settings.deepinfra_model = SMALL
        probe_trials = _trials(patients[:1], 1)
        cost = _calibrate(patients[0], probe_trials, ledger)
        depth = tail_depth(cost, len(patients))
        plan = {
            "call_usd": round(cost, 6),
            "depth": depth,
            "n_patients": len(patients),
            "small": SMALL,
            "drop_limit": DROP_LIMIT,
        }
        PLAN.write_text(json.dumps(plan, indent=2))
        print(f"plan depth={depth} call=${cost:.6f}", flush=True)

    settings.deepinfra_model = SMALL
    trials = _trials(patients, depth)
    have = _load_calls(SMALL)
    work = _tail_work(patients, depth, have, SMALL)
    if work:
        _run(work, trials, SMALL)
    have = _load_calls(SMALL)
    rows = [row for row in have.values() if row.get("tier")]
    stats = drop_rate(rows)
    complete = _complete(patients, depth, have)
    report: dict = {
        "plan": plan,
        "complete": complete,
        "drop": stats,
        "spent_usd": round(ledger.spent_usd(), 6),
    }
    if not complete or drop_kills(stats["rate"]) or ledger.remaining_usd() < GAIN_FLOOR_USD:
        report["decision"] = (
            "drop" if complete and drop_kills(stats["rate"])
            else "no_budget" if complete and ledger.remaining_usd() < GAIN_FLOOR_USD
            else "incomplete"
        )
        OUT.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    settings.deepinfra_model = BIG
    have_big = _load_calls(BIG)
    if "seventy_call_usd" not in plan:
        sample = patients[0]
        nct = sample["candidates"][100]
        before = ledger.spent_usd()
        if not have_big.get((sample["patient_id"], nct), {}).get("tier"):
            _run(
                [
                    {
                        "cohort": sample["cohort"],
                        "patient_id": sample["patient_id"],
                        "note": sample["note"],
                        "nct_id": nct,
                        "rank": 101,
                        "gold": sample["gold"].get(nct, ""),
                    }
                ],
                trials,
                BIG,
            )
        cost = ledger.spent_usd() - before
        if cost <= 0:
            cost = plan["call_usd"]
        plan["seventy_call_usd"] = round(cost, 6)
        PLAN.write_text(json.dumps(plan, indent=2))
    else:
        cost = plan["seventy_call_usd"]

    count = _arm_count(patients, depth, have, cost, ledger.remaining_usd())
    plan["arm_count"] = count
    PLAN.write_text(json.dumps(plan, indent=2))
    if count <= 0:
        report["decision"] = "no_budget"
        report["arm_count"] = 0
        OUT.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    wanted: list[dict] = []
    arms: dict[str, dict[str, list[str]]] = {}
    seen: set[tuple[str, str]] = set()
    for patient in patients:
        tail = patient["candidates"][100 : 100 + depth]
        naive = tail[:count]
        kept = [
            nct
            for nct in tail
            if have.get((patient["patient_id"], nct), {}).get("tier") != "excluded"
        ][:count]
        arms[patient["patient_id"]] = {"naive": naive, "cascade": kept}
        for nct in set(naive) | set(kept):
            key = (patient["patient_id"], nct)
            if key in seen or have_big.get(key, {}).get("tier"):
                seen.add(key)
                continue
            seen.add(key)
            rank = 101 + tail.index(nct)
            wanted.append(
                {
                    "cohort": patient["cohort"],
                    "patient_id": patient["patient_id"],
                    "note": patient["note"],
                    "nct_id": nct,
                    "rank": rank,
                    "gold": patient["gold"].get(nct, ""),
                }
            )
    if wanted:
        _run(wanted, trials, BIG)
    have_big = _load_calls(BIG)
    naive_hits, cascade_hits = [], []
    finished = True
    for patient in patients:
        pid = patient["patient_id"]
        for name in ("naive", "cascade"):
            for nct in arms[pid][name]:
                if not have_big.get((pid, nct), {}).get("tier"):
                    finished = False
        naive_hits.append(_hits(arms[pid]["naive"], have_big, pid, patient["gold"]))
        cascade_hits.append(_hits(arms[pid]["cascade"], have_big, pid, patient["gold"]))
    report["arm_count"] = count
    report["seventy_complete"] = finished
    if not finished:
        report["decision"] = "incomplete"
    else:
        ci = paired_bootstrap_ci(naive_hits, cascade_hits)
        lo = ci["ci"][0]
        gain = ci["point"] > 0 and lo > 0
        report["gain"] = {
            "point": ci["point"],
            "ci": ci["ci"],
            "wilcoxon": wilcoxon_paired(naive_hits, cascade_hits),
            "naive_hits": naive_hits,
            "cascade_hits": cascade_hits,
        }
        report["decision"] = "gain" if gain else "no_gain"
    report["spent_usd"] = round(ledger.spent_usd(), 6)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "gain"}, indent=2))


if __name__ == "__main__":
    main()
