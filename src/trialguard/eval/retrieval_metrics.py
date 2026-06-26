"""Retrieval evaluation metrics: recall@k, MRR, nDCG@k."""

from __future__ import annotations

import math


def recall_at_k(
    predictions: list[str],
    gold_positives: set[str],
    k: int,
) -> float:
    if not gold_positives:
        return 0.0
    hits = sum(1 for nct in predictions[:k] if nct in gold_positives)
    return hits / len(gold_positives)


def mrr(predictions: list[str], gold_positives: set[str]) -> float:
    for rank, nct in enumerate(predictions, start=1):
        if nct in gold_positives:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    predictions: list[str],
    gold_labels: dict[str, int],
    k: int,
) -> float:
    """nDCG with graded relevance: eligible=2, excluded=1, irrelevant=0."""
    label_map = {"eligible": 2, "excluded": 1, "irrelevant": 0}

    def dcg(ranking: list[str]) -> float:
        score = 0.0
        for i, nct in enumerate(ranking[:k], start=1):
            rel = label_map.get(gold_labels.get(nct, "irrelevant"), 0)
            score += rel / math.log2(i + 1)
        return score

    ideal = sorted(gold_labels.values(), key=lambda v: label_map.get(v, 0), reverse=True)
    ideal_dcg = sum(
        label_map.get(v, 0) / math.log2(i + 2)
        for i, v in enumerate(ideal[:k])
    )
    if ideal_dcg == 0:
        return 0.0
    return dcg(predictions) / ideal_dcg


def evaluate_cohort(
    cohort: str,
    retriever_fn,
    k: int = 10,
) -> dict:
    """Run retriever over all patients in cohort, return aggregated metrics.

    retriever_fn(patient_description, source) -> (list[(nct_id, score)], latency_dict)
    """
    from trialguard.eval.cohorts import load_labels, load_patients

    patients = load_patients(cohort)
    labels_list = load_labels(cohort)

    # Build per-patient label maps
    patient_labels: dict[str, dict[str, str]] = {}
    for lbl in labels_list:
        pid = lbl["patient_id"]
        if pid not in patient_labels:
            patient_labels[pid] = {}
        patient_labels[pid][lbl["nct_id"]] = lbl["label"]

    recalls, mrrs, ndcgs = [], [], []
    latencies = []

    for patient in patients:
        pid = patient["patient_id"]
        labels = patient_labels.get(pid, {})
        gold_positives = {nct for nct, lbl in labels.items() if lbl == "eligible"}

        if not gold_positives:
            continue

        results, latency = retriever_fn(patient["description"], cohort)
        predictions = [nct for nct, _ in results]

        latencies.append(latency["total_ms"])
        recalls.append(recall_at_k(predictions, gold_positives, k))
        mrrs.append(mrr(predictions, gold_positives))
        ndcgs.append(ndcg_at_k(predictions, labels, k))

    n = len(recalls)
    lat_sorted = sorted(latencies)

    return {
        "cohort": cohort,
        "n_patients": n,
        "recall@k": round(sum(recalls) / n, 4) if n else 0.0,
        "mrr": round(sum(mrrs) / n, 4) if n else 0.0,
        f"ndcg@{k}": round(sum(ndcgs) / n, 4) if n else 0.0,
        "latency_p50_ms": round(lat_sorted[n // 2], 1) if n else 0.0,
        "latency_p95_ms": round(lat_sorted[int(n * 0.95)], 1) if n else 0.0,
        "k": k,
    }
