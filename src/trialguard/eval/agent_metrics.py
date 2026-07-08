"""Agent faithfulness eval: single-pass baseline vs verified graph.

Measures what the "self-verifying" claim requires, all at $0:

- citation_precision: grounded decisive verdicts / decisive verdicts attempted.
  Deterministic (verify/grounding.py). 1 - this = unsupported-verdict rate,
  the hallucinated-citation proxy.
- unsupported_verdict_rate: decisive verdicts whose quote is NOT verbatim in source.
- abstention_rate: criteria answered cannot_determine/unverifiable (coverage inverse).
- trial_accuracy: trial roll-up vs qrels gold (eligible=2, excluded=1).

The two arms differ only in max_retries (0 = baseline, 2 = verified). Analyst
calls are cached by (patient, trial, prompt_version), so re-runs cost zero.
"""

from __future__ import annotations

import json
from pathlib import Path

REPORT_DIR = Path("data/reports")


def _build_subset(cohort: str, n_patients: int, per_class: int) -> list[dict]:
    """Pick patients with both eligible and excluded gold trials in-corpus.

    Returns list of {patient_id, note, trials:[{nct_id, gold, criteria, source_text}]}.
    """
    from trialguard.eval.cohorts import load_labels, load_patients
    from trialguard.eval.file_index import get_index
    from trialguard.ingestion.normalise import normalise_trial

    idx = get_index(cohort)
    texts = idx.trial_texts()  # nct_id -> title+criteria doc string
    corpus_ids = idx.corpus_ids()

    # Rebuild raw trials for criteria + source text.
    from trialguard.eval.file_index import _load_sigir_trials, _load_trec_trials
    raw = _load_sigir_trials() if cohort == "sigir" else _load_trec_trials(cohort)
    by_id = {t["nct_id"]: normalise_trial(t) for t in raw}

    patients = {p["patient_id"]: p for p in load_patients(cohort)}
    labels = load_labels(cohort)
    per_patient: dict[str, dict[str, str]] = {}
    for lbl in labels:
        per_patient.setdefault(lbl["patient_id"], {})[lbl["nct_id"]] = lbl["label"]

    subset = []
    for pid, lab in per_patient.items():
        if pid not in patients:
            continue
        elig = [n for n, l in lab.items() if l == "eligible" and n in corpus_ids and n in by_id]
        excl = [n for n, l in lab.items() if l == "excluded" and n in corpus_ids and n in by_id]
        if len(elig) < per_class or len(excl) < per_class:
            continue
        chosen = [(n, "eligible") for n in elig[:per_class]] + [(n, "excluded") for n in excl[:per_class]]
        trials = []
        for nct, gold in chosen:
            t = by_id[nct]
            criteria = t.get("inclusion_criteria", [])[:12]
            if not criteria:
                continue
            trials.append({
                "nct_id": nct,
                "gold": gold,
                "criteria": criteria,
                "source_text": t.get("eligibility_raw", "") or texts.get(nct, ""),
            })
        if trials:
            subset.append({"patient_id": pid, "note": patients[pid]["description"], "trials": trials})
        if len(subset) >= n_patients:
            break
    return subset


def _run_arm(subset: list[dict], max_retries: int, handler=None) -> dict:
    from trialguard.agent.graph import assess

    decisive = grounded = abstain = total_crit = 0
    trial_correct = trial_total = 0

    for p in subset:
        for tr in p["trials"]:
            state = assess(
                p["note"], tr["nct_id"], tr["criteria"], tr["source_text"],
                max_retries=max_retries, handler=handler,
            )
            for a in state["assessments"]:
                total_crit += 1
                v = a.get("verdict")
                # "decisive attempt" = analyst produced a met/not_met with a quote
                # that either grounded (still met/not_met) or was forced unverifiable.
                if v in ("met", "not_met"):
                    decisive += 1
                    grounded += 1  # grounded verdicts survive as met/not_met
                elif a.get("grounding_failure"):
                    decisive += 1  # attempted but failed grounding -> unverifiable
                if v in ("cannot_determine", "unverifiable"):
                    abstain += 1
            # trial-level accuracy vs qrels
            trial_total += 1
            if state["trial_verdict"] == tr["gold"]:
                trial_correct += 1

    cp = grounded / decisive if decisive else 0.0
    return {
        "max_retries": max_retries,
        "n_criteria": total_crit,
        "decisive_attempts": decisive,
        "grounded": grounded,
        "citation_precision": round(cp, 4),
        "unsupported_verdict_rate": round(1 - cp, 4),
        "abstention_rate": round(abstain / total_crit, 4) if total_crit else 0.0,
        "trial_accuracy": round(trial_correct / trial_total, 4) if trial_total else 0.0,
        "n_trials": trial_total,
    }


def run(cohort: str, n_patients: int, per_class: int) -> dict:
    from trialguard.tracing import get_langchain_handler
    handler = get_langchain_handler(session_id=f"agent-eval-{cohort}", tags=["agent-eval"])

    subset = _build_subset(cohort, n_patients, per_class)
    baseline = _run_arm(subset, max_retries=0, handler=handler)
    verified = _run_arm(subset, max_retries=2, handler=handler)
    return {"cohort": cohort, "n_patients": len(subset), "baseline": baseline, "verified": verified}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="sigir")
    parser.add_argument("--n-patients", type=int, default=5)
    parser.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args()

    out = run(args.cohort, args.n_patients, args.per_class)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "phase3_agent.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
