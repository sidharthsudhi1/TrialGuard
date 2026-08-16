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
    from trialguard.agent.schema import build_typed_criteria
    from trialguard.eval.cohorts import load_labels, load_patients
    from trialguard.eval.file_index import _load_sigir_trials, _load_trec_trials
    from trialguard.ingestion.normalise import normalise_trial

    raw = _load_sigir_trials() if cohort == "sigir" else _load_trec_trials(cohort)
    by_id = {t["nct_id"]: normalise_trial(t) for t in raw}
    corpus_ids = set(by_id)

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
            criteria, truncated = build_typed_criteria(t)
            if not criteria:
                continue
            trials.append({
                "nct_id": nct,
                "gold": gold,
                "criteria": criteria,
                "criteria_truncated": truncated,
                "source_text": t.get("eligibility_raw", ""),
            })
        if trials:
            subset.append(
                {"patient_id": pid, "note": patients[pid]["description"], "trials": trials}
            )
        if len(subset) >= n_patients:
            break
    return subset


def _is_transient(exc: BaseException) -> bool:
    """Timeouts and dropped connections are retryable; 429/budget are not."""
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(
        m in blob
        for m in ("timeout", "timed out", "apiconnectionerror", "connection reset")
    )


def _eval_workers() -> int:
    """Parallel analyst calls per arm. Default 1 — committed results stay
    byte-identical unless a run explicitly opts in. Higher values only help
    because the analyst call is network-bound; the cost ledger is lock-guarded."""
    import os

    return max(1, int(os.environ.get("TG_EVAL_WORKERS", "1")))


def _run_arm(subset: list[dict], max_retries: int, handler=None) -> dict:
    from trialguard.agent.graph import assess

    decisive = grounded = abstain = total_crit = 0
    trial_correct = trial_total = 0
    total_retries = trials_with_retry = 0
    rate_limited = False
    per_trial: dict[str, dict[str, int]] = {}
    # Same counters split by criterion kind. Exclusion criteria carry inverted
    # semantics and quote from a different span, so an aggregate rate can hide a
    # regression isolated to one kind.
    by_kind: dict[str, dict[str, int]] = {
        k: dict.fromkeys(("n_criteria", "decisive", "grounded", "abstain"), 0)
        for k in ("inclusion", "exclusion")
    }

    import os

    from trialguard.agent.analyst import CACHE_DIR as ACACHE
    from trialguard.agent.analyst import _cache_key
    cached_only = os.environ.get("TG_CACHED_ONLY") == "1"

    # Flatten to (note, trial) work items so an arm can run concurrently.
    work = []
    for p in subset:
        for tr in p["trials"]:
            if cached_only:
                cp_path = ACACHE / f"{_cache_key(p['note'], tr['nct_id'])}.json"
                if not cp_path.exists():
                    continue  # skip uncached trial; no fresh Groq call
            work.append((p["note"], tr))

    def _assess_one(item):
        import time

        note, tr = item
        last: Exception | None = None
        for attempt in range(3):
            try:
                return tr, assess(
                    note, tr["nct_id"], tr["criteria"], tr["source_text"],
                    max_retries=max_retries, handler=handler,
                    criteria_truncated=tr.get("criteria_truncated", False),
                )
            except Exception as e:
                last = e
                if not _is_transient(e) or attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise last  # pragma: no cover

    workers = _eval_workers()
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=workers)
        results_iter = pool.map(_assess_one, work)
    else:
        pool = None
        results_iter = map(_assess_one, work)

    try:
        for item in work:
            if rate_limited:
                break
            try:
                tr, state = next(results_iter)  # type: ignore[call-overload]
            except StopIteration:
                break
            except Exception as e:
                # Groq free-tier daily token cap (TPD) is a hard wall. Stop and
                # report metrics over the trials that completed rather than crash.
                # BudgetExhausted is the pre-emptive local gate; 429/rate_limit is
                # the server telling us the same thing.
                from trialguard.agent.ratelimit import BudgetExhausted
                if isinstance(e, BudgetExhausted) or "rate_limit" in str(e) or "429" in str(e):
                    rate_limited = True
                    break
                raise
            t_dec = t_uns = 0
            for a in state["assessments"]:
                total_crit += 1
                v = a.get("verdict")
                bk = by_kind.get(a.get("kind") or "inclusion", by_kind["inclusion"])
                bk["n_criteria"] += 1
                # "decisive attempt" = analyst produced a met/not_met with a quote
                # that either grounded (still met/not_met) or was forced unverifiable.
                if v in ("met", "not_met"):
                    decisive += 1
                    grounded += 1  # grounded verdicts survive as met/not_met
                    t_dec += 1
                    bk["decisive"] += 1
                    bk["grounded"] += 1
                elif a.get("grounding_failure"):
                    decisive += 1  # attempted but failed grounding -> unverifiable
                    t_dec += 1
                    t_uns += 1
                    bk["decisive"] += 1
                if v in ("cannot_determine", "unverifiable"):
                    abstain += 1
                    bk["abstain"] += 1
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
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    cp = grounded / decisive if decisive else 0.0
    kind_report = {}
    for kind, c in by_kind.items():
        k_cp = c["grounded"] / c["decisive"] if c["decisive"] else 0.0
        kind_report[kind] = {
            "n_criteria": c["n_criteria"],
            "decisive_attempts": c["decisive"],
            "grounded": c["grounded"],
            "citation_precision": round(k_cp, 4),
            "unsupported_verdict_rate": round(1 - k_cp, 4) if c["decisive"] else 0.0,
            "coverage": round(c["grounded"] / c["n_criteria"], 4) if c["n_criteria"] else 0.0,
            "abstention_rate": round(c["abstain"] / c["n_criteria"], 4) if c["n_criteria"] else 0.0,
        }
    return {
        "max_retries": max_retries,
        "by_kind": kind_report,
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
    from trialguard.agent.schema import attach_kinds, normalize_criteria
    from trialguard.verify.grounding import ground_assessments

    cached = []
    for p in subset:
        for tr in p["trials"]:
            if (ACACHE / f"{_cache_key(p['note'], tr['nct_id'])}.json").exists():
                raw = analyze_trial(p["note"], tr["nct_id"], tr["criteria"])
                # Kinds and the bare note are carried so the curve sweeps the same
                # verifier the graph ships (absence path included), not a variant.
                typed = normalize_criteria(tr["criteria"])
                cached.append(
                    (attach_kinds(raw, typed), p["note"] + "\n" + tr["source_text"], p["note"])
                )

    curve = []
    for mt in min_tokens_values:
        decisive = grounded = total = 0
        for raw, source, note in cached:
            for a in ground_assessments(raw, source, min_tokens=mt, patient_text=note):
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


def _observability(verified: dict, run_usd: float | None = None) -> dict:
    """Run-level quality scores for the Langfuse dashboard (Phase 5 WS-2).

    Faithfulness = citation precision of the verified (thesis) arm. Kept in the
    report too, so the observability numbers survive even without a tracing backend.

    `run_usd` (Phase 8 WS-5) puts spend next to quality on the same board. Without
    it the dashboard can say a run got more faithful but not what that cost, which
    is half of the tradeoff any tuning decision actually turns on.
    """
    scores = {
        "faithfulness": verified["citation_precision"],
        "unsupported_verdict_rate": verified["unsupported_verdict_rate"],
        "abstention_rate": verified["abstention_rate"],
        "coverage": verified["coverage"],
        "mean_retries": verified["mean_retries"],
    }
    if run_usd is not None:
        scores["run_usd"] = round(run_usd, 6)
    return scores


def run(cohort: str, n_patients: int, per_class: int) -> dict:
    from trialguard.eval.significance import matched_ab
    from trialguard.tracing import emit_scores, get_langchain_handler
    session_id = f"agent-eval-{cohort}"
    handler = get_langchain_handler(session_id=session_id, tags=["agent-eval"])

    from trialguard.llm.cost import active_ledger

    spend_before = active_ledger().spent_usd()

    subset = _build_subset(cohort, n_patients, per_class)
    baseline = _run_arm(subset, max_retries=0, handler=handler)
    verified = _run_arm(subset, max_retries=2, handler=handler)
    sig = matched_ab(baseline["per_trial"], verified["per_trial"])
    curve = coverage_curve(subset)
    # per_trial is bookkeeping for the matched test; drop it from the saved report.
    baseline.pop("per_trial", None)
    verified.pop("per_trial", None)
    # Ledger delta over the run. Cache hits cost nothing, so a fully cached rerun
    # honestly reports $0 rather than re-charging work already paid for.
    run_usd = max(0.0, active_ledger().spent_usd() - spend_before)
    scores = _observability(verified, run_usd=run_usd)
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
