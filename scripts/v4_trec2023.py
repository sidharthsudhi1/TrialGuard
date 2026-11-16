"""V4: TREC 2023 keywords, retrieval scoring, and the capped analyst pass.

Retrieval itself runs on Modal (scripts/modal_retrieve_trec2023.py). This file
does the two DeepInfra pieces and the local scoring.

    python scripts/v4_trec2023.py keywords
    python scripts/v4_trec2023.py score
    python scripts/v4_trec2023.py assess
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPORTS = Path("data/reports")
KEYWORDS = REPORTS / "v4_keywords.json"
RANKINGS = REPORTS / "v4_rankings.json"
TRIALS = REPORTS / "v4_top_trials.json"
LEDGERS = Path("data/cache/ledgers")
BIG = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEPTHS = (100, 200, 500)


def _isolate(name: str, cap: float):
    from trialguard.config import settings
    from trialguard.eval.spend import isolated_ledger

    settings.llm_provider = "deepinfra"
    settings.deepinfra_model = BIG
    return isolated_ledger(LEDGERS / name, cap)


def keywords() -> None:
    """One 70B keyword list per questionnaire. Hard stop $0.20. Refuses PHI."""
    from trialguard.agent.sanitize import detect_phi
    from trialguard.eval.cohorts import load_patients
    from trialguard.retrieval.query_transform import generate_keywords

    os.environ["TG_PROMPT_VERSION"] = "v4"
    ledger = _isolate("v4_keywords.json", 0.20)
    before = ledger.spent_usd()
    rows = []
    for patient in load_patients("trec_2023"):
        flags = detect_phi(patient["description"])
        if flags:
            raise SystemExit(
                f"PHI classes on {patient['patient_id']}: {', '.join(flags)}. No LLM call."
            )
        words = generate_keywords(patient["description"])
        rows.append(
            {
                "patient_id": patient["patient_id"],
                "note": patient["description"],
                "keywords": words,
                "raw_fallback": words == [patient["description"]],
            }
        )
        print(patient["patient_id"], len(words), flush=True)
    KEYWORDS.parent.mkdir(parents=True, exist_ok=True)
    KEYWORDS.write_text(
        json.dumps(
            {
                "model": BIG,
                "run_usd": round(ledger.spent_usd() - before, 6),
                "patients": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {KEYWORDS}")


def _corpus_ids() -> set[str]:
    path = Path("data/indexes/trec_2023_medcpt_excl_ids.json")
    return set(json.loads(path.read_text()))


def score() -> None:
    """Recall of the Modal rankings against eligible qrels. No model calls."""
    from trialguard.eval.cohorts import load_labels
    from trialguard.eval.retrieval_metrics import recall_at_k
    from trialguard.eval.significance import paired_bootstrap_ci

    rankings = json.loads(RANKINGS.read_text())
    corpus = _corpus_ids()
    gold: dict[str, set[str]] = {}
    labelled = 0
    for row in load_labels("trec_2023"):
        if row["label"] != "eligible":
            continue
        labelled += 1
        if row["nct_id"] in corpus:
            gold.setdefault(row["patient_id"], set()).add(row["nct_id"])
    present = sum(len(v) for v in gold.values())
    arms: dict[str, dict[int, list[float]]] = {
        name: {k: [] for k in DEPTHS} for name in ("keyword", "raw")
    }
    patients = []
    for pid, eligible in gold.items():
        if pid not in rankings or not eligible:
            continue
        patients.append(pid)
        for name in ("keyword", "raw"):
            hits = rankings[pid][name]
            for k in DEPTHS:
                arms[name][k].append(recall_at_k(hits, eligible, k))

    def mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    deltas = {}
    for k in DEPTHS:
        ci = paired_bootstrap_ci(arms["raw"][k], arms["keyword"][k])
        deltas[str(k)] = {"point": ci["point"], "ci": ci["ci"]}
    report = {
        "patients": len(patients),
        "gold_eligible_in_corpus": present,
        "gold_eligible_labelled": labelled,
        "coverage": round(present / labelled, 4) if labelled else None,
        "keyword": {str(k): mean(arms["keyword"][k]) for k in DEPTHS},
        "raw": {str(k): mean(arms["raw"][k]) for k in DEPTHS},
        "keyword_minus_raw": deltas,
    }
    path = REPORTS / "v4_retrieval.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def assess() -> None:
    """Top-100 keyword arm, prompt v4, hard stop at $2 of new spend."""
    import trialguard.eval.end_to_end as e2e
    from trialguard.ingestion.normalise import normalise_trial

    os.environ["TG_PROMPT_VERSION"] = "v4"
    os.environ.setdefault("TG_EVAL_WORKERS", "8")
    os.environ["TG_LLM_TIMEOUT"] = "300"
    ledger = _isolate("v4_analyst.json", 2.0)
    before = ledger.spent_usd()
    rankings = json.loads(RANKINGS.read_text())
    raw_trials = json.loads(TRIALS.read_text())
    corpus = _corpus_ids()
    gold = e2e._gold_by_patient("trec_2023")
    notes = {row["patient_id"]: row["note"] for row in json.loads(KEYWORDS.read_text())["patients"]}
    rows = []
    for pid, lab in gold.items():
        eligible = {nct for nct, name in lab.items() if name == "eligible" and nct in corpus}
        if not eligible or pid not in rankings:
            continue
        rows.append(
            {
                "patient_id": pid,
                "note": notes[pid],
                "gold_eligible": sorted(eligible),
                "gold_labels": lab,
                "retrieved": rankings[pid]["keyword"][:100],
            }
        )

    def _load(_cohort: str, needed: set[str]) -> dict:
        out = {}
        for nct in needed:
            raw = raw_trials.get(nct)
            if raw is None:
                continue
            out[nct] = normalise_trial(
                {"nct_id": nct, "title": raw["title"], "eligibility_raw": raw["text"]}
            )
        return out

    e2e._load_corpus = _load
    counts = e2e.assess_retrieved(rows, "trec_2023")
    metrics = e2e.score(rows)
    report = {
        "cohort": "trec_2023",
        "top_k": 100,
        "model": BIG,
        "prompt_version": "v4",
        "metrics": metrics,
        "counts": counts,
        "run_usd": round(max(0.0, ledger.spent_usd() - before), 6),
    }
    path = REPORTS / "v4_e2e.json"
    path.write_text(json.dumps(report, indent=2))
    print(
        f"patients={metrics['patients']} completion={counts['completion']} "
        f"usd={report['run_usd']} surfaced={metrics['tier_surfaced']['recall']}"
    )


def main() -> None:
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "keywords":
        keywords()
    elif cmd == "score":
        score()
    elif cmd == "assess":
        assess()
    else:
        raise SystemExit("usage: v4_trec2023.py keywords|score|assess")


if __name__ == "__main__":
    main()
