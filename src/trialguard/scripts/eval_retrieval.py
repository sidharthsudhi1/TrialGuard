"""Phase 2 retrieval eval CLI.

Usage:
    python -m trialguard.scripts.eval_retrieval --cohort sigir
    python -m trialguard.scripts.eval_retrieval --cohort trec_2021 --top-k 20
    python -m trialguard.scripts.eval_retrieval --all-cohorts
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
REPORT_DIR = Path("data/reports")


def retriever_fn(source: str, top_k: int):
    from trialguard.eval.file_index import get_index
    import time

    idx = get_index(source)

    def _fn(patient_description: str, _source: str):
        t0 = time.perf_counter()
        results = idx.search(patient_description, top_k=top_k)
        total_ms = (time.perf_counter() - t0) * 1000
        latency = {"total_ms": round(total_ms, 1)}
        return results, latency

    return _fn


def run(cohorts: list[str], top_k: int) -> None:
    from trialguard.eval.retrieval_metrics import evaluate_cohort

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for cohort in cohorts:
        console.print(f"\n[bold]Evaluating {cohort}...[/bold]")
        results = evaluate_cohort(
            cohort=cohort,
            retriever_fn=retriever_fn(cohort, top_k),
            k=top_k,
        )
        all_results.append(results)
        console.print(
            f"  recall@{top_k}: {results['recall@k']:.4f}  "
            f"MRR: {results['mrr']:.4f}  "
            f"nDCG@{top_k}: {results[f'ndcg@{top_k}']:.4f}  "
            f"p50: {results['latency_p50_ms']}ms  "
            f"p95: {results['latency_p95_ms']}ms  "
            f"n={results['n_patients']}"
        )

    # Rich table
    table = Table(title="Phase 2 Retrieval Metrics", show_lines=True)
    table.add_column("Cohort")
    table.add_column(f"Recall@{top_k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column(f"nDCG@{top_k}", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("n", justify="right")

    target_hit = True
    for r in all_results:
        recall = r["recall@k"]
        style = "green" if recall >= 0.90 else "red"
        table.add_row(
            r["cohort"],
            f"[{style}]{recall:.4f}[/{style}]",
            str(r["mrr"]),
            str(r[f"ndcg@{top_k}"]),
            str(r["latency_p50_ms"]),
            str(r["latency_p95_ms"]),
            str(r["n_patients"]),
        )
        if recall < 0.90:
            target_hit = False

    console.print(table)

    if target_hit:
        console.print(f"[bold green]Target recall@{top_k} >= 0.90 hit on all cohorts.[/bold green]")
    else:
        console.print(
            f"[bold yellow]recall@{top_k} < 0.90 on some cohorts. "
            "Consider adding cross-encoder reranker.[/bold yellow]"
        )

    # Write markdown report
    _write_report(all_results, top_k, REPORT_DIR / "phase2_retrieval.md")


def _write_report(results: list[dict], k: int, path: Path) -> None:
    lines = [
        "# Phase 2 Retrieval Metrics",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"\nRetriever: dense (MiniLM) + BM25 fused with RRF (k=60)",
        f"\n## Results (top-{k})\n",
        f"| Cohort | Recall@{k} | MRR | nDCG@{k} | p50 ms | p95 ms | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['cohort']} | {r['recall@k']:.4f} | {r['mrr']:.4f} "
            f"| {r[f'ndcg@{k}']:.4f} | {r['latency_p50_ms']} | {r['latency_p95_ms']} "
            f"| {r['n_patients']} |"
        )
    lines += [
        "\n## Target",
        "- Recall@10 >= 0.90 (TrialGPT parity)",
        "\n## Notes",
        "- Source filter: retrieval scoped to eval corpus per cohort",
        "- BM25 corpus loaded in-memory once per process",
    ]
    path.write_text("\n".join(lines))
    console.print(f"\nReport saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TrialGuard Phase 2 retrieval eval")
    parser.add_argument("--cohort", choices=["sigir", "trec_2021", "trec_2022"])
    parser.add_argument("--all-cohorts", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.all_cohorts:
        cohorts = ["sigir", "trec_2021", "trec_2022"]
    elif args.cohort:
        cohorts = [args.cohort]
    else:
        parser.error("Specify --cohort or --all-cohorts")

    run(cohorts, args.top_k)


if __name__ == "__main__":
    main()
