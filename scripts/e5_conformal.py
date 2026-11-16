"""E5: false-exclusion of gold-eligible trials, calibrated across TREC cohorts.

alpha = 0.10. A threshold t excludes only when the disqualifying count is at
least t. Calibrate the smallest such t on one cohort and verify it on the other.
Adopt only if both directions hold. Cached assessments only.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/reports/e5_conformal.json")
ALPHA = 0.10


def _load(cohort: str) -> tuple[list[dict], list[dict]]:
    import os

    from trialguard.eval.end_to_end import assess_retrieved, retrieve_for_patients

    # V2 assessed the top 100 from 50-deep lists. 2022's cached index excludes
    # exclusion text; 2021's includes it. Matching that is what makes the
    # analyst cache hit.
    os.environ["TG_LIST_POOL"] = "50"
    os.environ["TG_PROMPT_VERSION"] = "v4"
    os.environ["TG_INDEX_EXCLUSION"] = "0" if cohort == "trec_2022" else "1"
    os.environ.setdefault("TG_EVAL_WORKERS", "8")
    patients, _ = retrieve_for_patients(cohort, n_patients=0, top_k=100)
    assess_retrieved(patients, cohort, cached_only=True)
    eligible = []
    for patient in patients:
        labels = patient["gold_labels"]
        for nct, verdict in patient["verdicts"].items():
            if labels.get(nct) != "eligible":
                continue
            eligible.append(
                {
                    "patient_id": patient["patient_id"],
                    "nct_id": nct,
                    "label": "eligible",
                    "n_disqualifying": int(verdict.get("n_disqualifying") or 0),
                }
            )
        for nct, verdict in patient["verdicts"].items():
            if labels.get(nct) != "excluded":
                continue
            eligible.append(
                {
                    "patient_id": patient["patient_id"],
                    "nct_id": nct,
                    "label": "excluded",
                    "n_disqualifying": int(verdict.get("n_disqualifying") or 0),
                }
            )
    return eligible, patients


def _split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    gold = [row for row in rows if row["label"] == "eligible"]
    blocked = [row for row in rows if row["label"] == "excluded"]
    return gold, blocked


def _fer(rows: list[dict], t: int) -> float:
    """Gold-eligible trials a threshold t would still exclude."""
    if not rows:
        return 1.0
    return sum(1 for row in rows if row["n_disqualifying"] >= t) / len(rows)


def _calibrate(rows: list[dict]) -> int | None:
    if not rows:
        return None
    cap = max((r["n_disqualifying"] for r in rows), default=1)
    for t in range(1, cap + 2):
        if _fer(rows, t) <= ALPHA:
            return t
    return None


def _ndcg_pairs(patients: list[dict]) -> tuple[list[float], list[float]]:
    """nDCG@10 of retrieval order versus n_unknown order, inside the non-excluded set."""
    from trialguard.eval.retrieval_metrics import ndcg_at_k

    base, alt = [], []
    for patient in patients:
        surfaced = []
        for nct in patient["retrieved"]:
            verdict = patient["verdicts"].get(nct)
            if not verdict or verdict.get("trial_tier") == "excluded":
                continue
            surfaced.append((nct, int(verdict.get("n_unknown") or 0)))
        if len(surfaced) < 2:
            continue
        labels = patient["gold_labels"]
        retrieval = [nct for nct, _ in surfaced]
        by_unknown = [nct for nct, _ in sorted(surfaced, key=lambda item: (item[1], retrieval.index(item[0])))]
        base.append(ndcg_at_k(retrieval, labels, 10))
        alt.append(ndcg_at_k(by_unknown, labels, 10))
    return base, alt


def main() -> None:
    # The saved end_to_end JSON does not keep per-trial disqualifiers. Re-read
    # them from the cached graph.
    data = {}
    ndcg = {}
    blocked: dict[str, list[dict]] = {}
    for cohort in ("trec_2021", "trec_2022"):
        eligible, patients = _load(cohort)
        data[cohort] = _split(eligible)[0]
        ndcg[cohort] = _ndcg_pairs(patients)
        blocked[cohort] = _split(eligible)[1]
    report = {"alpha": ALPHA, "cohorts": {}}
    for cohort, rows in data.items():
        report["cohorts"][cohort] = {
            "n_gold_eligible_assessed": len(rows),
            "fer_served": round(_fer(rows, 1), 4) if rows else None,
            "fer_t3": round(_fer(rows, 3), 4) if rows else None,
            "n_gold_excluded_assessed": len(blocked[cohort]),
            "gold_excluded_released_at_t3": round(
                sum(1 for row in blocked[cohort] if 0 < row["n_disqualifying"] < 3)
                / len(blocked[cohort]),
                4,
            ) if blocked[cohort] else None,
        }
    holds = []
    for cal, ver in (("trec_2021", "trec_2022"), ("trec_2022", "trec_2021")):
        t = _calibrate(data[cal])
        verified = None if t is None else _fer(data[ver], t) <= ALPHA
        holds.append(bool(t is not None and verified))
        report[f"{cal}_to_{ver}"] = {"t": t, "holds": verified}
    report["adopt"] = all(holds)
    from trialguard.eval.significance import compare_family

    family = {
        cohort: ndcg[cohort]
        for cohort in ndcg
        if ndcg[cohort][0]
    }
    report["r5_ndcg10"] = compare_family(family) if family else {}
    report["r5_adopt"] = all(
        row.get("significant_bh") and row.get("mean_delta", 0) > 0
        for row in report["r5_ndcg10"].values()
    ) if report["r5_ndcg10"] else False
    OUT.write_text(json.dumps({k: v for k, v in report.items()}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
