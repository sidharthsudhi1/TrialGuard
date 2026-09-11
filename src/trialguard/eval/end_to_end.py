"""End-to-end evaluation: patient note in, grounded verdicts out.

Retrieval is measured against gold. Faithfulness is measured against gold. Neither
number says whether the system works, because a patient arrives with a note, not
with a candidate trial: the served path has to *surface* an eligible trial and
then reach a grounded verdict on it. This measures that composition.

The headline is `end_to_end_recall` — of every trial the cohort marks eligible,
the fraction this system both retrieved and called eligible. It is bounded above
by retrieval: a trial that is never retrieved cannot be assessed at any quality,
so `retrieval_recall` is reported beside it as the ceiling and
`agent_loss` as the gap the agent gives back.

Expect the headline to be well below either component metric. That is the point.
A system metric that flatters its parts is not worth having.

Cost: one analyst call per (patient, retrieved trial) that is not already cached.
`--cached-only` skips any (patient, trial) pair whose analyst response is not
already on disk, which is the right mode for iterating on retrieval: retrieval
changes which pairs get assessed, not what the analyst says about a pair it has
already seen. It does not make a run free — keyword extraction still costs one
call per note the keyword cache has never seen, at roughly $0.00004 each. The
run reports what it actually spent rather than claiming zero.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPORT_DIR = Path("data/reports")


def _load_corpus(cohort: str, needed: set[str]) -> dict[str, dict]:
    """Normalised trials for the retrieved ids only.

    TREC's full corpus is ~26k trials and does not fit twice in memory on a small
    machine, so only what retrieval actually surfaced is materialised.
    """
    from trialguard.eval.file_index import _load_sigir_trials, _load_trec_trials
    from trialguard.ingestion.normalise import normalise_trial

    raw = _load_sigir_trials() if cohort == "sigir" else _load_trec_trials(cohort, needed)
    return {t["nct_id"]: normalise_trial(t) for t in raw if t["nct_id"] in needed}


def _gold_by_patient(cohort: str) -> dict[str, dict[str, str]]:
    from trialguard.eval.cohorts import load_labels

    out: dict[str, dict[str, str]] = {}
    for lbl in load_labels(cohort):
        out.setdefault(lbl["patient_id"], {})[lbl["nct_id"]] = lbl["label"]
    return out


def retrieve_for_patients(
    cohort: str, n_patients: int, top_k: int, use_keywords: bool = True
) -> tuple[list[dict], dict[str, float]]:
    """Run the served retrieval shape over a cohort. Returns (rows, timing).

    Only patients with at least one in-corpus eligible trial are scored: a patient
    whose gold eligible set is empty makes recall undefined rather than zero, and
    averaging those in would quietly inflate the result.
    """
    from trialguard.eval.file_index import get_index

    gold = _gold_by_patient(cohort)
    idx = get_index(cohort)
    corpus_ids = set(idx._nct_ids)

    from trialguard.eval.cohorts import load_patients

    rows, t0 = [], time.perf_counter()
    for p in load_patients(cohort):
        pid = p["patient_id"]
        lab = gold.get(pid, {})
        eligible = {n for n, g in lab.items() if g == "eligible" and n in corpus_ids}
        if not eligible:
            continue
        hits = idx.search(p["description"], top_k=top_k, use_keywords=use_keywords)
        rows.append(
            {
                "patient_id": pid,
                "note": p["description"],
                "gold_eligible": sorted(eligible),
                "gold_labels": lab,
                "retrieved": [n for n, _ in hits],
            }
        )
        if len(rows) >= n_patients:
            break
    return rows, {"retrieval_s": round(time.perf_counter() - t0, 1)}


def _is_cached(note: str, nct_id: str) -> bool:
    """Whether the analyst's first attempt for this pair is already on disk.

    TG_CACHED_ONLY only guards the retry path inside the graph, so it does not by
    itself stop a first-attempt call. The check has to happen before assess() is
    entered or `--cached-only` would quietly spend money.
    """
    from trialguard.agent.analyst import CACHE_DIR, _cache_key

    return (CACHE_DIR / f"{_cache_key(note, nct_id)}.json").exists()


def _eval_workers() -> int:
    """Parallel assessments. Default 1, so a re-run of a committed number stays
    byte-identical unless it explicitly opts in. Mirrors the flag agent_metrics
    already uses; 10 is the measured DeepInfra ceiling before timeouts."""
    import os

    return max(1, int(os.environ.get("TG_EVAL_WORKERS", "1")))


def assess_retrieved(
    rows: list[dict], cohort: str, max_retries: int = 2, cached_only: bool = False
) -> dict:
    """Assess every retrieved trial for every patient. Mutates rows with verdicts."""
    from trialguard.agent.graph import assess
    from trialguard.agent.ratelimit import BudgetExhausted
    from trialguard.agent.schema import build_typed_criteria

    needed = {n for r in rows for n in r["retrieved"]}
    corpus = _load_corpus(cohort, needed)

    assessed = skipped = budget_stops = uncached = 0
    t0 = time.perf_counter()

    # Flatten first so the work can run concurrently. Every (patient, trial) pair
    # is independent -- the only shared state is the cost ledger, which serialises
    # its own writes -- so the assessments are identical either way and only the
    # wall clock moves.
    work: list[tuple[dict, str, dict, list[dict], bool]] = []
    for r in rows:
        r["verdicts"] = {}
        for nct in r["retrieved"]:
            trial = corpus.get(nct)
            if trial is None:
                skipped += 1
                continue
            criteria, truncated = build_typed_criteria(trial)
            if not criteria:
                skipped += 1
                continue
            if cached_only and not _is_cached(r["note"], nct):
                uncached += 1
                continue
            work.append((r, nct, trial, criteria, truncated))

    def _one(item):
        r, nct, trial, criteria, truncated = item
        return item, assess(
            r["note"],
            nct,
            criteria,
            trial.get("eligibility_raw", ""),
            max_retries=max_retries,
            criteria_truncated=truncated,
        )

    workers = _eval_workers()
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=workers)
        results = pool.map(_one, work)
    else:
        pool = None
        results = map(_one, work)

    try:
        for _ in work:
            try:
                item, state = next(results)  # type: ignore[call-overload]
            except StopIteration:
                break
            except BudgetExhausted:
                # Stop rather than silently scoring a partial run as if complete.
                # Which patient the exhausted call belonged to is not recoverable
                # once the work is pooled, so every patient still holding
                # unassessed trials is marked incomplete -- the honest reading,
                # since none of them was scored over its full retrieved set.
                budget_stops += 1
                assessed_ids = {n for row in rows for n in row["verdicts"]}
                for row in rows:
                    if set(row["retrieved"]) - assessed_ids:
                        row["incomplete"] = True
                break
            except Exception:  # noqa: BLE001 — one bad trial must not void the run
                skipped += 1
                continue
            r, nct, _trial, criteria, _truncated = item
            ass = state.get("assessments", [])
            r["verdicts"][nct] = {
                "trial_verdict": state.get("trial_verdict", "cannot_determine"),
                "trial_tier": state.get("trial_tier", "needs_review"),
                "n_unknown": state.get("n_unknown", 0),
                "n_criteria": len(ass),
                # What the analyst was asked for, against what came back. A prompt
                # that silently answers fewer criteria than it was given produces a
                # trial the roll-up can only call needs_review, and every rate below
                # would otherwise be computed over the shrunken denominator and look
                # unchanged. This is the direct measure of that.
                "n_criteria_asked": len(criteria),
                "n_unanswered": len(_unanswered(ass, criteria)),
                "n_unmatched": len(_unmatched(ass, criteria)),
                "n_grounded": sum(1 for a in ass if a.get("grounded")),
                "n_unverifiable": sum(1 for a in ass if a.get("verdict") == "unverifiable"),
                # Full distribution, not just the abstention rate. L4 improved the
                # faithfulness proxy 26% while making the system strictly worse, and
                # only the criterion-level split showed that the vanished grounding
                # failures had become abstentions rather than recoveries.
                "verdicts": _verdict_counts(ass),
                # WS-5a: which source each grounded quote actually came from. A
                # "note" span means the verdict rests on text the user supplied,
                # which is the class an attacker controls.
                "grounded_in": _provenance_counts(ass),
                # WS-5b: grounded decisive verdicts whose quote only restates the
                # criterion. A deterministic lower bound on non-entailment.
                "self_referential": sum(1 for a in ass if a.get("self_referential")),
                # Exclusion not_met grounded on a quote the absence check
                # disagrees with. Recorded, never enforced -- see grounding.py.
                "weak_absence": sum(1 for a in ass if a.get("weak_absence")),
            }
            assessed += 1
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    total = assessed + uncached
    return {
        "assessed": assessed,
        "skipped": skipped,
        "uncached_skipped": uncached,
        # Below 1.0 the run scored only the pairs that happened to be cached, which
        # is a biased subset — prior eval runs cached gold-selected pairs, not
        # retrieved ones. The headline is then not comparable to a full run.
        "cache_coverage": round(assessed / total, 4) if total else 0.0,
        "budget_stops": budget_stops,
        "assess_s": round(time.perf_counter() - t0, 1),
    }


_VERDICTS = ("met", "not_met", "cannot_determine", "unverifiable")


def _unanswered(assessments: list[dict], criteria: list[dict]) -> list[dict]:
    """Asked criteria with no assessment against them."""
    from trialguard.agent.schema import align_assessments

    slots, _ = align_assessments(assessments, criteria)
    return [c for c, answer in zip(criteria, slots) if answer is None]


def _unmatched(assessments: list[dict], criteria: list[dict]) -> list[dict]:
    """Assessments naming a criterion that was never asked for."""
    from trialguard.agent.schema import align_assessments

    _, leftover = align_assessments(assessments, criteria)
    return leftover


_PROVENANCE = ("trial", "note", "absence")


def _provenance_counts(assessments: list[dict]) -> dict[str, int]:
    counts = dict.fromkeys(_PROVENANCE, 0)
    for a in assessments:
        src = a.get("grounded_in")
        if a.get("grounded") and src in counts:
            counts[src] += 1
    return counts


def _verdict_counts(assessments: list[dict]) -> dict[str, int]:
    counts = dict.fromkeys(_VERDICTS, 0)
    for a in assessments:
        v = a.get("verdict")
        if v in counts:
            counts[v] += 1
    return counts


def score(rows: list[dict]) -> dict:
    """Compose retrieval and verdicts into the numbers that describe the system."""
    n_gold = n_retrieved = n_correct = 0
    said_eligible = said_eligible_right = 0
    verdict_counts: dict[str, int] = {}
    crit_total = crit_unver = 0
    crit_asked = crit_grounded = 0
    crit_unanswered = crit_unmatched = 0
    crit_verdicts = dict.fromkeys(_VERDICTS, 0)
    crit_provenance = dict.fromkeys(_PROVENANCE, 0)
    crit_self_ref = 0
    crit_weak_absence = 0

    for r in rows:
        gold_elig = set(r["gold_eligible"])
        got = set(r["retrieved"])
        verdicts = r.get("verdicts", {})
        n_gold += len(gold_elig)
        n_retrieved += len(gold_elig & got)
        for nct in gold_elig & got:
            if verdicts.get(nct, {}).get("trial_verdict") == "eligible":
                n_correct += 1
        for nct, v in verdicts.items():
            verdict_counts[v["trial_verdict"]] = verdict_counts.get(v["trial_verdict"], 0) + 1
            crit_total += v["n_criteria"]
            crit_unver += v["n_unverifiable"]
            crit_asked += v.get("n_criteria_asked", v["n_criteria"])
            crit_unanswered += v.get("n_unanswered", 0)
            crit_unmatched += v.get("n_unmatched", 0)
            crit_grounded += v.get("n_grounded", 0)
            for name, n in (v.get("verdicts") or {}).items():
                if name in crit_verdicts:
                    crit_verdicts[name] += n
            for name, n in (v.get("grounded_in") or {}).items():
                if name in crit_provenance:
                    crit_provenance[name] += n
            crit_self_ref += v.get("self_referential", 0)
            crit_weak_absence += v.get("weak_absence", 0)
            if v["trial_verdict"] == "eligible":
                said_eligible += 1
                if r["gold_labels"].get(nct) == "eligible":
                    said_eligible_right += 1

    def _rate(a: int, b: int) -> float:
        return round(a / b, 4) if b else 0.0

    # The tiered contract (A5) is scored as two operating points on one system,
    # never averaged: the "eligible" tier trades recall for precision and the
    # surfaced set does the reverse. A single F1 hides which one moved, and with
    # recall hard-capped by retrieval it would reward surfacing everything.
    surfaced_hit = surfaced_shown = surfaced_labelled = 0
    for r in rows:
        gold_elig = set(r["gold_eligible"])
        for nct, v in r.get("verdicts", {}).items():
            if v.get("trial_tier") == "excluded":
                continue
            surfaced_shown += 1
            label = r["gold_labels"].get(nct)
            if label:
                surfaced_labelled += 1
            if nct in gold_elig:
                surfaced_hit += 1

    retrieval_recall = _rate(n_retrieved, n_gold)
    end_to_end = _rate(n_correct, n_gold)
    return {
        "tier_surfaced": {
            "shown": surfaced_shown,
            "recall": _rate(surfaced_hit, n_gold),
            # Over labelled rows only: an unlabelled row is an unjudged pool gap,
            # not a false positive, and counting it as one understates every arm.
            "precision": _rate(surfaced_hit, surfaced_labelled),
        },
        "patients": len(rows),
        "gold_eligible_total": n_gold,
        "retrieval_recall": retrieval_recall,
        "end_to_end_recall": end_to_end,
        # What the agent gives back on trials retrieval already handed it. Keeping
        # the two apart is what makes a regression attributable to a stage.
        "agent_loss": round(retrieval_recall - end_to_end, 4),
        "eligible_precision": _rate(said_eligible_right, said_eligible),
        "trial_verdicts": verdict_counts,
        "criterion_unverifiable_rate": _rate(crit_unver, crit_total),
        "criterion_verdicts": crit_verdicts,
        "criterion_grounded": crit_grounded,
        "criterion_total": crit_total,
        # Criteria the analyst was handed but never answered. Non-zero means the
        # rates above describe a subset of what was asked.
        "criterion_asked": crit_asked,
        "criterion_unanswered": crit_unanswered,
        # Assessments naming something never asked. attach_kinds marks these
        # "unknown" and the roll-up treats them as unresolved, but they inflate
        # the total, which is why unanswered is no longer a subtraction: it
        # went negative on SIGIR and hid the real shortfall behind the excess.
        "criterion_unmatched": crit_unmatched,
        "criterion_grounded_rate": _rate(crit_grounded, crit_asked),
        "criterion_grounded_in": crit_provenance,
        # The share of grounded criteria whose only evidence is user-supplied
        # text. This is the size of the hole WS-5a is about, not an error rate.
        "note_only_grounded_rate": _rate(crit_provenance["note"], crit_grounded),
        "self_referential": crit_self_ref,
        # Lower bound, not an estimate: a quote citing the wrong patient fact is
        # equally unsupported and no string comparison can see it.
        "self_referential_rate": _rate(crit_self_ref, crit_grounded),
        "weak_absence": crit_weak_absence,
        "weak_absence_rate": _rate(crit_weak_absence, crit_grounded),
        "incomplete_patients": sum(1 for r in rows if r.get("incomplete")),
    }


def run(cohort: str, n_patients: int, top_k: int, cached_only: bool = False) -> dict:
    from trialguard.llm.cost import active_ledger
    from trialguard.llm.provider import active_model, active_provider

    if cached_only:
        os.environ["TG_CACHED_ONLY"] = "1"

    spend_before = active_ledger().spent_usd()
    rows, r_timing = retrieve_for_patients(cohort, n_patients, top_k)
    a_timing = assess_retrieved(rows, cohort, cached_only=cached_only)
    metrics = score(rows)
    run_usd = max(0.0, active_ledger().spent_usd() - spend_before)

    return {
        "cohort": cohort,
        "top_k": top_k,
        "provider": active_provider(),
        "model": active_model(),
        "prompt_version": os.environ.get("TG_PROMPT_VERSION", "v1"),
        "cached_only": cached_only,
        "metrics": metrics,
        "counts": a_timing,
        "timing": {**r_timing, **{"assess_s": a_timing["assess_s"]}},
        "run_usd": round(run_usd, 6),
    }


def main() -> None:
    import argparse

    from rich.console import Console
    from rich.table import Table

    ap = argparse.ArgumentParser(description="End-to-end eval: note -> retrieval -> verdicts")
    ap.add_argument("--cohort", default="sigir", choices=["sigir", "trec_2021", "trec_2022"])
    ap.add_argument("--n-patients", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--cached-only", action="store_true",
                    help="Skip pairs with no cached analyst response. Bounds the "
                         "expensive half; keyword extraction may still cost a few "
                         "cents on notes never seen before.")
    ap.add_argument("--out", default=None,
                    help="Report path (default: data/reports/e2e_<cohort>.json)")
    args = ap.parse_args()

    console = Console()
    console.print(
        f"[bold]End-to-end[/bold] {args.cohort} · {args.n_patients} patients · top-{args.top_k}"
        + (" · cached-only" if args.cached_only else "")
    )
    result = run(args.cohort, args.n_patients, args.top_k, args.cached_only)
    m = result["metrics"]

    t = Table(show_header=False, box=None)
    t.add_row("patients scored", str(m["patients"]))
    t.add_row("gold eligible trials", str(m["gold_eligible_total"]))
    t.add_row("retrieval recall (ceiling)", f"{m['retrieval_recall']:.4f}")
    t.add_row("[bold]end-to-end recall[/bold] (eligible tier)",
              f"[bold]{m['end_to_end_recall']:.4f}[/bold]")
    ts = m["tier_surfaced"]
    t.add_row("surfaced recall (eligible + review)", f"{ts['recall']:.4f}")
    t.add_row("surfaced precision", f"{ts['precision']:.4f}")
    t.add_row("trials surfaced", str(ts["shown"]))
    t.add_row("agent loss", f"{m['agent_loss']:.4f}")
    t.add_row("precision of 'eligible'", f"{m['eligible_precision']:.4f}")
    t.add_row("criterion unverifiable rate", f"{m['criterion_unverifiable_rate']:.4f}")
    t.add_row("trials assessed", str(result["counts"]["assessed"]))
    t.add_row("cache coverage", f"{result['counts']['cache_coverage']:.4f}")
    t.add_row("run cost", f"${result['run_usd']:.6f}")
    console.print(t)

    cov = result["counts"]["cache_coverage"]
    if result["cached_only"] and cov < 1.0:
        console.print(
            f"[yellow]cache coverage {cov:.2f}: {result['counts']['uncached_skipped']} "
            f"retrieved pairs were never assessed because they are not cached. The "
            f"headline describes a biased subset — run without --cached-only for a "
            f"comparable number.[/yellow]"
        )
    if m["incomplete_patients"]:
        console.print(
            f"[yellow]{m['incomplete_patients']} patient(s) incomplete — budget stopped "
            f"mid-run; treat the headline as a floor, not a result.[/yellow]"
        )

    out = Path(args.out) if args.out else REPORT_DIR / f"e2e_{args.cohort}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    console.print(f"\nReport: [cyan]{out}[/cyan]")


if __name__ == "__main__":
    main()
