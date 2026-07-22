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
        elig = [n for n, gl in lab.items() if gl == "eligible" and n in corpus_ids and n in by_id]
        excl = [n for n, gl in lab.items() if gl == "excluded" and n in corpus_ids and n in by_id]
        if len(elig) < per_class or len(excl) < per_class:
            continue
        chosen = (
            [(n, "eligible") for n in elig[:per_class]]
            + [(n, "excluded") for n in excl[:per_class]]
        )
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
            subset.append(
                {"patient_id": pid, "note": patients[pid]["description"], "trials": trials}
            )
        if len(subset) >= n_patients:
            break
    return subset


def _run_arm(subset: list[dict], max_retries: int, handler=None) -> dict:
    from trialguard.agent.graph import assess

    decisive = grounded = abstain = total_crit = 0
    trial_correct = trial_total = 0
    total_retries = trials_with_retry = 0
    rate_limited = False
    per_trial: dict[str, dict[str, int]] = {}

    import os

    from trialguard.agent.analyst import CACHE_DIR as ACACHE
    from trialguard.agent.analyst import _cache_key
    cached_only = os.environ.get("TG_CACHED_ONLY") == "1"

    for p in subset:
        if rate_limited:
            break
        for tr in p["trials"]:
            if cached_only:
                cp_path = ACACHE / f"{_cache_key(p['note'], tr['nct_id'])}.json"
                if not cp_path.exists():
                    continue  # skip uncached trial; no fresh Groq call
            try:
                state = assess(
                    p["note"], tr["nct_id"], tr["criteria"], tr["source_text"],
                    max_retries=max_retries, handler=handler,
                )
            except Exception as e:
                # Groq free-tier daily token cap (TPD) is a hard wall. Stop and
                # report metrics over the trials that completed rather than crash.
                if "rate_limit" in str(e) or "429" in str(e):
                    rate_limited = True
                    break
                raise
            t_dec = t_uns = 0
            for a in state["assessments"]:
                total_crit += 1
                v = a.get("verdict")
                # "decisive attempt" = analyst produced a met/not_met with a quote
                # that either grounded (still met/not_met) or was forced unverifiable.
                if v in ("met", "not_met"):
                    decisive += 1
                    grounded += 1  # grounded verdicts survive as met/not_met
                    t_dec += 1
                elif a.get("grounding_failure"):
                    decisive += 1  # attempted but failed grounding -> unverifiable
                    t_dec += 1
                    t_uns += 1
                if v in ("cannot_determine", "unverifiable"):
                    abstain += 1
            per_trial[tr["nct_id"]] = {"decisive": t_dec, "unsupported": t_uns}
            # retry observability: how often the grounding back-edge fired, and how
            # deep. Native retry spans are in the trace; this is the aggregate.
            retries_used = state.get("retries", 0)
            total_retries += retries_used
            if retries_used:
                trials_with_retry += 1
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
        # coverage = criteria that end as a grounded, decisive verdict. Read jointly
        # with citation_precision: faithfulness bought with coverage is the tradeoff.
        "coverage": round(grounded / total_crit, 4) if total_crit else 0.0,
        "abstention_rate": round(abstain / total_crit, 4) if total_crit else 0.0,
        "trial_accuracy": round(trial_correct / trial_total, 4) if trial_total else 0.0,
        "n_trials": trial_total,
        "mean_retries": round(total_retries / trial_total, 4) if trial_total else 0.0,
        "trials_with_retry": trials_with_retry,
        "rate_limited": rate_limited,
        "per_trial": per_trial,
    }


def coverage_curve(subset: list[dict], min_tokens_values=(1, 2, 3, 4)) -> list[dict]:
    """Coverage vs citation-precision as the grounding strictness knob is swept.

    Deterministic over the cached analyst outputs (no retry, no fresh Groq calls):
    re-grounds each cached assessment at each min_tokens and recomputes the joint
    (coverage, precision). Shows the tradeoff the single operating point hides —
    and that the token guard (min_tokens=2) sits near the knee, not at an extreme.
    """
    from trialguard.agent.analyst import CACHE_DIR as ACACHE
    from trialguard.agent.analyst import _cache_key, analyze_trial
    from trialguard.verify.grounding import ground_assessments

    cached = []
    for p in subset:
        for tr in p["trials"]:
            if (ACACHE / f"{_cache_key(p['note'], tr['nct_id'])}.json").exists():
                raw = analyze_trial(p["note"], tr["nct_id"], tr["criteria"])
                cached.append((raw, p["note"] + "\n" + tr["source_text"]))

    curve = []
    for mt in min_tokens_values:
        decisive = grounded = total = 0
        for raw, source in cached:
            for a in ground_assessments(raw, source, min_tokens=mt):
                total += 1
                v = a.get("verdict")
                if v in ("met", "not_met"):
                    decisive += 1
                    grounded += 1
                elif a.get("grounding_failure"):
                    decisive += 1
        curve.append({
            "min_tokens": mt,
            "coverage": round(grounded / total, 4) if total else 0.0,
            "citation_precision": round(grounded / decisive, 4) if decisive else 0.0,
            "n_criteria": total,
        })
    return curve


def _observability(verified: dict) -> dict:
    """Run-level quality scores for the Langfuse dashboard (Phase 5 WS-2).

    Faithfulness = citation precision of the verified (thesis) arm. Kept in the
    report too, so the observability numbers survive even without a tracing backend.
    """
    return {
        "faithfulness": verified["citation_precision"],
        "unsupported_verdict_rate": verified["unsupported_verdict_rate"],
        "abstention_rate": verified["abstention_rate"],
        "coverage": verified["coverage"],
        "mean_retries": verified["mean_retries"],
    }


def run(cohort: str, n_patients: int, per_class: int) -> dict:
    from trialguard.eval.significance import matched_ab
    from trialguard.tracing import emit_scores, get_langchain_handler
    session_id = f"agent-eval-{cohort}"
    handler = get_langchain_handler(session_id=session_id, tags=["agent-eval"])

    subset = _build_subset(cohort, n_patients, per_class)
    baseline = _run_arm(subset, max_retries=0, handler=handler)
    verified = _run_arm(subset, max_retries=2, handler=handler)
    sig = matched_ab(baseline["per_trial"], verified["per_trial"])
    curve = coverage_curve(subset)
    # per_trial is bookkeeping for the matched test; drop it from the saved report.
    baseline.pop("per_trial", None)
    verified.pop("per_trial", None)
    scores = _observability(verified)
    emit_scores(scores, session_id=session_id)  # no-op without Langfuse creds
    return {
        "cohort": cohort,
        "n_patients": len(subset),
        "baseline": baseline,
        "verified": verified,
        "significance": sig,
        "coverage_curve": curve,
        "observability": scores,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="sigir")
    parser.add_argument("--n-patients", type=int, default=5)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--tag", default="phase4",
                        help="report filename prefix; keeps phase reports side by side")
    args = parser.parse_args()

    out = run(args.cohort, args.n_patients, args.per_class)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # Prefix + cohort so a run never clobbers another phase's or cohort's result.
    (REPORT_DIR / f"{args.tag}_agent_{args.cohort}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
