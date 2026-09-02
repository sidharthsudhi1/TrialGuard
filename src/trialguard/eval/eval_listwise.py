"""H2 eval: does listwise reranking a deep pool beat plain retrieval at the same k?

The comparison is deliberately narrow. Both arms see the same patients, the same
index and the same fused ranking; the only difference is whether the top-k comes
off the head of that ranking or off an LLM screening pass over a deeper slice of
it. Retrieval-only, scored against gold, so it costs one screening call per
batch and no assessments.

CLI: python -m trialguard.eval.eval_listwise --cohort trec_2021 --n-patients 20
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPORT_DIR = Path("data/reports")


def _gold_by_patient(cohort: str) -> dict[str, set[str]]:
    from trialguard.eval.cohorts import load_labels

    out: dict[str, set[str]] = {}
    for lbl in load_labels(cohort):
        if lbl["label"] == "eligible":
            out.setdefault(lbl["patient_id"], set()).add(lbl["nct_id"])
    return out


def run(
    cohort: str, n_patients: int, pool: int, top_k: int, batch_size: int = 50
) -> dict:
    from trialguard.eval.cohorts import load_patients
    from trialguard.eval.file_index import get_index
    from trialguard.llm.cost import active_ledger
    from trialguard.llm.provider import active_model, active_provider
    from trialguard.retrieval.listwise import listwise_rerank

    idx = get_index(cohort)
    texts = idx.trial_texts()
    corpus = idx.corpus_ids()
    gold_all = _gold_by_patient(cohort)
    spend_before = active_ledger().spent_usd()

    rows = []
    t0 = time.perf_counter()
    for p in load_patients(cohort):
        gold = {g for g in gold_all.get(p["patient_id"], set()) if g in corpus}
        if not gold:
            continue

        # One deep retrieval per patient; both arms slice the same list, so any
        # difference is the reranker and not a re-run of retrieval.
        deep = idx.search(p["description"], top_k=pool, use_keywords=True)
        baseline = [n for n, _ in deep[:top_k]]
        reranked = [n for n, _ in listwise_rerank(
            p["description"], deep, texts, top_k=top_k, batch_size=batch_size
        )]

        rows.append({
            "patient_id": p["patient_id"],
            "n_gold": len(gold),
            "pool_hits": len(gold & {n for n, _ in deep}),
            "baseline_hits": len(gold & set(baseline)),
            "reranked_hits": len(gold & set(reranked)),
        })
        if len(rows) >= n_patients:
            break

    n_gold = sum(r["n_gold"] for r in rows)

    def _recall(key: str) -> float:
        return round(sum(r[key] for r in rows) / n_gold, 4) if n_gold else 0.0

    # Per-patient paired counts: the aggregate can move on one outlier patient
    # with many gold trials, which is how R3's SIGIR result overstated itself.
    better = sum(1 for r in rows if r["reranked_hits"] > r["baseline_hits"])
    worse = sum(1 for r in rows if r["reranked_hits"] < r["baseline_hits"])

    return {
        "cohort": cohort,
        "pool": pool,
        "top_k": top_k,
        "batch_size": batch_size,
        "provider": active_provider(),
        "model": active_model(),
        "patients": len(rows),
        "gold_total": n_gold,
        "pool_ceiling": _recall("pool_hits"),
        "baseline_recall": _recall("baseline_hits"),
        "reranked_recall": _recall("reranked_hits"),
        "patients_better": better,
        "patients_worse": worse,
        "wall_s": round(time.perf_counter() - t0, 1),
        "run_usd": round(max(0.0, active_ledger().spent_usd() - spend_before), 6),
        "rows": rows,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="H2 listwise rerank eval")
    ap.add_argument("--cohort", default="trec_2021",
                    choices=["sigir", "trec_2021", "trec_2022"])
    ap.add_argument("--n-patients", type=int, default=20)
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r = run(args.cohort, args.n_patients, args.pool, args.top_k, args.batch_size)
    print(f"{r['cohort']}  pool={r['pool']} -> top-{r['top_k']}  "
          f"{r['patients']} patients, {r['gold_total']} gold")
    print(f"  pool ceiling      {r['pool_ceiling']:.4f}")
    print(f"  baseline recall   {r['baseline_recall']:.4f}")
    print(f"  reranked recall   {r['reranked_recall']:.4f}")
    delta = r["reranked_recall"] - r["baseline_recall"]
    print(f"  delta             {delta:+.4f}  "
          f"({r['patients_better']} better / {r['patients_worse']} worse)")
    print(f"  wall {r['wall_s']}s  cost ${r['run_usd']:.4f}")

    out = Path(args.out) if args.out else REPORT_DIR / f"h2_listwise_{args.cohort}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, sort_keys=True))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
