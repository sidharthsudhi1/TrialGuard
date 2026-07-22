"""Phase 2 retrieval eval CLI.

Usage:
    python -m trialguard.scripts.eval_retrieval --cohort sigir
    python -m trialguard.scripts.eval_retrieval --cohort sigir --use-keywords
    python -m trialguard.scripts.eval_retrieval --cohort sigir --ablate
    python -m trialguard.scripts.eval_retrieval --cohort sigir --ablate --pool-size 100
    python -m trialguard.scripts.eval_retrieval --cohort sigir --use-rerank
    python -m trialguard.scripts.eval_retrieval --all-cohorts
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
REPORT_DIR = Path("data/reports")
K_LIST = [10, 20, 50, 100, 200]


def _retriever_fn(source: str, max_k: int, use_keywords: bool = False):
    from trialguard.eval.file_index import get_index

    idx = get_index(source)

    def _fn(patient_description: str, _source: str):
        t0 = time.perf_counter()
        results = idx.search(patient_description, top_k=max_k, use_keywords=use_keywords)
        total_ms = (time.perf_counter() - t0) * 1000
        return results, {"total_ms": round(total_ms, 1)}

    return _fn


def _reranking_retriever_fn(
    source: str,
    pool_size: int,
    trial_texts: dict[str, str],
    use_keywords: bool = True,
):
    """Retrieve wide (min(pool_size*4, 200)), rerank with cross-encoder, return pool_size."""
    from trialguard.eval.file_index import get_index
    from trialguard.retrieval.rerank import rerank as do_rerank

    idx = get_index(source)
    retrieve_depth = min(pool_size * 4, 200)

    def _fn(patient_description: str, _source: str):
        t0 = time.perf_counter()
        candidates = idx.search(patient_description, top_k=retrieve_depth, use_keywords=use_keywords)
        results = do_rerank(patient_description, candidates, trial_texts, top_k=pool_size)
        total_ms = (time.perf_counter() - t0) * 1000
        return results, {"total_ms": round(total_ms, 1)}

    return _fn


def run_rerank_experiment(cohorts: list[str], pool_size: int) -> dict[str, list[dict]]:
    """Run keyword baseline vs retrieve@wide→rerank→pool_size. Returns {baseline, rerank}."""
    from trialguard.eval.file_index import get_index
    from trialguard.eval.retrieval_metrics import compute_gold_coverage, evaluate_cohort_multi_k

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rerank_k_list = [10, 20, pool_size]
    baseline_results, rerank_results = [], []

    for cohort in cohorts:
        idx = get_index(cohort)
        cov = compute_gold_coverage(cohort, idx.corpus_ids())

        # Baseline: keyword, retrieve@pool_size, no rerank
        console.print(f"\n[bold]Rerank experiment: {cohort} — baseline (keyword, no rerank)...[/bold]")
        _print_coverage(cov, cohort)
        base = evaluate_cohort_multi_k(
            cohort=cohort,
            retriever_fn=_retriever_fn(cohort, pool_size, use_keywords=True),
            k_list=rerank_k_list,
            gold_coverage=cov["coverage"],
        )
        base["config"] = f"keyword@{pool_size} (no rerank)"
        base["coverage_info"] = cov
        baseline_results.append(base)

        recall_pool = base.get(f"recall@{pool_size}", 0.0)
        console.print(
            f"  recall@10: {base['recall@10']:.4f}  "
            f"recall@{pool_size}: {recall_pool:.4f}  "
            f"p50: {base['latency_p50_ms']}ms"
        )

        # Rerank: keyword retrieve@wide → cross-encoder → pool_size
        retrieve_depth = min(pool_size * 4, 200)
        console.print(
            f"\n[bold]Rerank experiment: {cohort} — rerank "
            f"(keyword@{retrieve_depth} → rerank → top-{pool_size})...[/bold]"
        )
        rr = evaluate_cohort_multi_k(
            cohort=cohort,
            retriever_fn=_reranking_retriever_fn(
                cohort, pool_size, idx.trial_texts(), use_keywords=True
            ),
            k_list=rerank_k_list,
            gold_coverage=cov["coverage"],
        )
        rr["config"] = f"keyword@{retrieve_depth}→rerank→{pool_size}"
        rr["coverage_info"] = cov
        rerank_results.append(rr)

        rr_pool = rr.get(f"recall@{pool_size}", 0.0)
        delta = rr_pool - recall_pool
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        console.print(
            f"  recall@10: {rr['recall@10']:.4f}  "
            f"recall@{pool_size}: {rr_pool:.4f} ({delta_str} vs baseline)  "
            f"p50: {rr['latency_p50_ms']}ms"
        )

    return {"baseline": baseline_results, "rerank": rerank_results}


def _print_rerank_table(baseline: list[dict], reranked: list[dict], pool_size: int) -> None:
    all_rows = [r for pair in zip(baseline, reranked) for r in pair]
    k_list = [10, 20, pool_size]

    table = Table(title=f"Rerank Pool-Compression Experiment (pool N={pool_size})", show_lines=True)
    table.add_column("Cohort")
    table.add_column("Config")
    for k in k_list:
        table.add_column(f"recall@{k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")

    for r in all_rows:
        recall_cells = []
        for k in k_list:
            val = r.get(f"recall@{k}", 0.0)
            style = "green" if val >= 0.65 else ("yellow" if val >= 0.53 else "white")
            recall_cells.append(f"[{style}]{val:.4f}[/{style}]")
        table.add_row(r["cohort"], r["config"], *recall_cells,
                      str(r["mrr"]), str(r["latency_p50_ms"]), str(r["latency_p95_ms"]))
    console.print(table)

    console.print()
    for base, rr in zip(baseline, reranked):
        base_pool = base.get(f"recall@{pool_size}", 0.0)
        rr_pool = rr.get(f"recall@{pool_size}", 0.0)
        delta = rr_pool - base_pool
        threshold = 0.65
        verdict = (
            f"[green]ADOPT rerank[/green] — recall@{pool_size} {rr_pool:.4f} ≥ {threshold}"
            if rr_pool >= threshold
            else f"[yellow]MARGINAL[/yellow] — recall@{pool_size} {rr_pool:.4f} < {threshold}, "
                 f"consider N=100 no-rerank (0.6919)"
        )
        console.print(
            f"[bold]{base['cohort']}[/bold]: Δrecall@{pool_size} = "
            f"{'+'if delta>=0 else ''}{delta:.4f}  →  {verdict}"
        )


def _write_rerank_report(
    baseline: list[dict], reranked: list[dict], pool_size: int, path: Path
) -> None:
    k_list = [10, 20, pool_size]
    all_rows = [r for pair in zip(baseline, reranked) for r in pair]
    lines = [
        "# Phase 2 Rerank Pool-Compression Experiment",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"\nHypothesis: recall@{pool_size}(retrieve@{min(pool_size*4,200)} → rerank → top-{pool_size})"
        f" > recall@{pool_size}(retrieve@{pool_size}) = baseline",
        "\nModel: cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU). Query: full patient note.",
        "\n## Results\n",
        "| Cohort | Config | " + " | ".join(f"recall@{k}" for k in k_list) + " | MRR | p50 ms | p95 ms |",
        "|---|---|" + "|".join("---" for _ in k_list) + "|---|---|---|",
    ]
    for r in all_rows:
        k_vals = " | ".join(f"{r.get(f'recall@{k}', 0.0):.4f}" for k in k_list)
        lines.append(
            f"| {r['cohort']} | {r['config']} | {k_vals} "
            f"| {r['mrr']:.4f} | {r['latency_p50_ms']} | {r['latency_p95_ms']} |"
        )

    lines.append("\n## Decision\n")
    for base, rr in zip(baseline, reranked):
        base_pool = base.get(f"recall@{pool_size}", 0.0)
        rr_pool = rr.get(f"recall@{pool_size}", 0.0)
        delta = rr_pool - base_pool
        threshold = 0.65
        if rr_pool >= threshold:
            decision = (
                f"**ADOPT** `retrieve@{min(pool_size*4,200)} → rerank → top-{pool_size}` "
                f"as Phase 4 default. "
                f"recall@{pool_size} = {rr_pool:.4f} ≥ {threshold} threshold. "
                f"Cheap {pool_size}-trial agent pool, high recall."
            )
        else:
            decision = (
                f"**FALL BACK** to N=100 no-rerank (recall@100 = 0.6919). "
                f"Rerank lift Δ={delta:+.4f} insufficient (recall@{pool_size} = {rr_pool:.4f} < {threshold}). "
                f"Accept 2× agent calls at N=100."
            )
        lines += [f"**{base['cohort']}**: {decision}", ""]

    lines += [
        "## Notes",
        "- Rerank query = full patient note (not keywords). Keywords cast wide; note judges precisely.",
        "- Cross-encoder scores cached per patient note hash — re-runs cost zero model calls.",
        "- General-domain reranker (ms-marco). If underperforming, clinical cross-encoder is next lever.",
        f"- Baseline recall@{pool_size} = {baseline[0].get(f'recall@{pool_size}', 0.0):.4f} "
        f"(keyword retrieve@{pool_size}, no rerank).",
    ]
    path.write_text("\n".join(lines))
    console.print(f"\nReport saved: {path}")


def _print_coverage(cov: dict, cohort: str) -> None:
    pct = cov["coverage"] * 100
    style = "green" if cov["coverage"] >= 0.95 else "yellow" if cov["coverage"] >= 0.70 else "red"
    console.print(
        f"  [{style}]Gold coverage: {pct:.1f}%[/{style}] "
        f"({cov['present_in_corpus']}/{cov['total_gold']} gold positives in {cohort} corpus, "
        f"missing: {cov['missing_count']})"
    )
    if cov["missing_sample"]:
        console.print(f"  Missing NCT IDs (first 10): {cov['missing_sample']}")


def run(cohorts: list[str], pool_size: int, use_keywords: bool) -> list[dict]:
    from trialguard.eval.file_index import get_index
    from trialguard.eval.retrieval_metrics import compute_gold_coverage, evaluate_cohort_multi_k

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    label = "keyword" if use_keywords else "raw-note"

    for cohort in cohorts:
        console.print(f"\n[bold]Evaluating {cohort} ({label})...[/bold]")
        idx = get_index(cohort)
        cov = compute_gold_coverage(cohort, idx.corpus_ids())
        _print_coverage(cov, cohort)

        results = evaluate_cohort_multi_k(
            cohort=cohort,
            retriever_fn=_retriever_fn(cohort, max(K_LIST), use_keywords),
            k_list=K_LIST,
            gold_coverage=cov["coverage"],
        )
        results["config"] = label
        results["coverage_info"] = cov
        all_results.append(results)

        pool_recall = results.get(f"recall@{pool_size}", 0.0)
        console.print(
            f"  recall@10: {results['recall@10']:.4f}  "
            f"recall@{pool_size}: {pool_recall:.4f}  "
            f"MRR: {results['mrr']:.4f}  "
            f"p50: {results['latency_p50_ms']}ms  n={results['n_patients']}"
        )

    return all_results


def _print_table(all_results: list[dict], pool_size: int) -> None:
    # Recall sweep table
    sweep = Table(title="Recall@N Sweep", show_lines=True)
    sweep.add_column("Cohort")
    sweep.add_column("Config")
    sweep.add_column("Coverage")
    for k in K_LIST:
        sweep.add_column(f"@{k}", justify="right")
    sweep.add_column("MRR", justify="right")
    sweep.add_column("p50 ms", justify="right")
    sweep.add_column("p95 ms", justify="right")

    for r in all_results:
        cov_pct = f"{r['coverage_info']['coverage'] * 100:.1f}%"
        recall_cells = []
        for k in K_LIST:
            val = r.get(f"recall@{k}", 0.0)
            style = "green" if val >= 0.90 else ("yellow" if val >= 0.50 else "white")
            recall_cells.append(f"[{style}]{val:.4f}[/{style}]")
        sweep.add_row(r["cohort"], r["config"], cov_pct, *recall_cells,
                      str(r["mrr"]), str(r["latency_p50_ms"]), str(r["latency_p95_ms"]))
    console.print(sweep)

    # Coverage-adjusted table
    adj = Table(title="Coverage-Adjusted Recall (recall / gold_coverage)", show_lines=True)
    adj.add_column("Cohort")
    adj.add_column("Config")
    adj.add_column("Gold coverage")
    for k in [50, 100, 200]:
        adj.add_column(f"@{k} raw", justify="right")
        adj.add_column(f"@{k} adj", justify="right")
    for r in all_results:
        cov = r["coverage_info"]["coverage"]
        cells = [r["cohort"], r["config"], f"{cov * 100:.1f}%"]
        for k in [50, 100, 200]:
            raw = r.get(f"recall@{k}", 0.0)
            adj_val = r.get(f"recall@{k}_adj")
            adj_str = f"{adj_val:.4f}" if adj_val is not None else "n/a"
            cells += [f"{raw:.4f}", adj_str]
        adj.add_row(*cells)
    console.print(adj)

    # Ceiling callout
    console.print()
    for r in all_results:
        pool_recall = r.get(f"recall@{pool_size}", 0.0)
        pool_recall_adj = r.get(f"recall@{pool_size}_adj")
        adj_str = f"{pool_recall_adj:.4f}" if pool_recall_adj is not None else "n/a"
        console.print(
            f"[bold]Agent best-case recall ceiling[/bold] "
            f"({r['cohort']} / {r['config']}, pool N={pool_size}): "
            f"[bold]{pool_recall:.4f}[/bold] raw  |  "
            f"[bold]{adj_str}[/bold] coverage-adjusted"
        )


def _write_report(all_results: list[dict], pool_size: int, path: Path, ablate: bool) -> None:
    config_note = "ablation: raw-note vs keyword-RRF" if ablate else (
        "keyword-RRF" if any(r.get("config") == "keyword" for r in all_results) else "raw-note"
    )
    lines = [
        "# Phase 2 Retrieval Metrics",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"\nRetriever: dense (BGE) + BM25 fused with RRF (k=60) — {config_note}",
    ]

    # Coverage section
    lines += ["\n## Gold Coverage\n",
               "| Cohort | Config | Total gold | In corpus | Missing | Coverage |",
               "|---|---|---|---|---|---|"]
    for r in all_results:
        cov = r["coverage_info"]
        lines.append(
            f"| {r['cohort']} | {r['config']} | {cov['total_gold']} "
            f"| {cov['present_in_corpus']} | {cov['missing_count']} "
            f"| {cov['coverage'] * 100:.1f}% |"
        )

    # Recall sweep
    k_headers = " | ".join(f"Recall@{k}" for k in K_LIST)
    lines += ["\n## Recall@N Sweep\n",
               f"| Cohort | Config | {k_headers} | MRR | p50 ms | p95 ms | n |",
               "|---|---|" + "|".join("---" for _ in K_LIST) + "|---|---|---|---|"]
    for r in all_results:
        k_vals = " | ".join(f"{r.get(f'recall@{k}', 0.0):.4f}" for k in K_LIST)
        lines.append(
            f"| {r['cohort']} | {r['config']} | {k_vals} "
            f"| {r['mrr']:.4f} | {r['latency_p50_ms']} | {r['latency_p95_ms']} "
            f"| {r['n_patients']} |"
        )

    # Coverage-adjusted
    lines += ["\n## Coverage-Adjusted Recall\n",
               "| Cohort | Config | Coverage | @50 raw | @50 adj | @100 raw | @100 adj | @200 raw | @200 adj |",
               "|---|---|---|---|---|---|---|---|---|"]
    for r in all_results:
        cov = r["coverage_info"]["coverage"]
        def _adj(k):
            v = r.get(f"recall@{k}_adj")
            return f"{v:.4f}" if v is not None else "n/a"
        lines.append(
            f"| {r['cohort']} | {r['config']} | {cov * 100:.1f}% "
            f"| {r.get('recall@50', 0.0):.4f} | {_adj(50)} "
            f"| {r.get('recall@100', 0.0):.4f} | {_adj(100)} "
            f"| {r.get('recall@200', 0.0):.4f} | {_adj(200)} |"
        )

    # Ceiling callout
    lines.append(f"\n## Agent Recall Ceiling (pool N={pool_size})\n")
    for r in all_results:
        raw = r.get(f"recall@{pool_size}", 0.0)
        adj_val = r.get(f"recall@{pool_size}_adj")
        adj_str = f"{adj_val:.4f}" if adj_val is not None else "n/a"
        lines.append(
            f"- **{r['cohort']} / {r['config']}**: "
            f"recall@{pool_size} = {raw:.4f} raw | {adj_str} coverage-adjusted"
        )

    lines += ["\n## Notes",
              "- Coverage < 1.0 means some gold trials were never loaded into the eval corpus.",
              "- Coverage-adjusted recall = raw recall / gold_coverage (ceiling achievable by retrieval alone).",
              f"- Pool size N={pool_size} is the assumed Phase 4 candidate pool passed to the agent.",
              "- Keyword latency cached; zero LLM calls after first run."]

    path.write_text("\n".join(lines))
    console.print(f"\nReport saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TrialGuard Phase 2 retrieval eval")
    parser.add_argument("--cohort", choices=["sigir", "trec_2021", "trec_2022"])
    parser.add_argument("--all-cohorts", action="store_true")
    parser.add_argument("--pool-size", type=int, default=50,
                        help="Candidate pool size passed to agent (default 50).")
    parser.add_argument("--use-rerank", action="store_true", default=False,
                        help="Run rerank pool-compression experiment "
                             "(keyword@wide → cross-encoder → pool-size). "
                             "Writes phase2_rerank.md.")

    kw_group = parser.add_mutually_exclusive_group()
    kw_group.add_argument("--use-keywords", action="store_true", default=False)
    kw_group.add_argument("--no-keywords", dest="use_keywords", action="store_false")
    kw_group.add_argument("--ablate", action="store_true", default=False,
                          help="Run raw-note and keyword configs side by side.")

    args = parser.parse_args()

    if args.all_cohorts:
        cohorts = ["sigir", "trec_2021", "trec_2022"]
    elif args.cohort:
        cohorts = [args.cohort]
    else:
        parser.error("Specify --cohort or --all-cohorts")

    if args.use_rerank:
        exp = run_rerank_experiment(cohorts, args.pool_size)
        _print_rerank_table(exp["baseline"], exp["rerank"], args.pool_size)
        _write_rerank_report(
            exp["baseline"], exp["rerank"], args.pool_size,
            REPORT_DIR / "phase2_rerank.md",
        )
    elif args.ablate:
        raw_results = run(cohorts, args.pool_size, use_keywords=False)
        kw_results = run(cohorts, args.pool_size, use_keywords=True)
        all_results = [r for pair in zip(raw_results, kw_results) for r in pair]
        _print_table(all_results, args.pool_size)
        _write_report(all_results, args.pool_size, REPORT_DIR / "phase2_retrieval.md", ablate=True)
    else:
        all_results = run(cohorts, args.pool_size, use_keywords=args.use_keywords)
        _print_table(all_results, args.pool_size)
        _write_report(all_results, args.pool_size, REPORT_DIR / "phase2_retrieval.md", ablate=False)


if __name__ == "__main__":
    main()
