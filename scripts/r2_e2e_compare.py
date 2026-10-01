"""Does the cross-encoder's +0.05 recall@100 reach the user? End-to-end, paired.

Compares two end_to_end reports per cohort, both at top-100 on every patient:
deep fusion (served top_k>=100 path) vs the R2 cross-encoder fused with deep
fusion (`--order-file`). Paired per patient: surfaced recall (Wilcoxon + BH)
and lift with patient-bootstrap CIs. Light: reads two JSON reports per cohort.

    python scripts/r2_e2e_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

from trialguard.eval.significance import compare_family

REPORTS = Path("data/reports")


def _per(path: Path) -> dict[str, dict]:
    return {p["patient_id"]: p for p in json.loads(path.read_text())["metrics"]["per_patient"]}


def main() -> None:
    family, out = {}, {"cohorts": {}}
    for c in ("trec_2021", "trec_2022"):
        base_path = REPORTS / f"r2e2e_deep_{c}_top100.json"
        ce_path = REPORTS / f"r2e2e_ce_{c}_top100.json"
        base, ce = json.loads(base_path.read_text()), json.loads(ce_path.read_text())
        b, e = _per(base_path), _per(ce_path)
        ids = sorted(set(b) & set(e))
        family[f"{c}:surfaced_recall"] = (
            [b[i]["surfaced_hit"] / b[i]["gold"] for i in ids],
            [e[i]["surfaced_hit"] / e[i]["gold"] for i in ids],
        )
        family[f"{c}:retrieval_recall"] = (
            [b[i]["retrieved_gold"] / b[i]["gold"] for i in ids],
            [e[i]["retrieved_gold"] / e[i]["gold"] for i in ids],
        )
        out["cohorts"][c] = {"n": len(ids), "deep": _summary(base), "ce_rrf_deep": _summary(ce)}
    out["paired"] = compare_family(family)
    (REPORTS / "r2_e2e_compare.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


def _summary(r: dict) -> dict:
    m = r["metrics"]
    return {
        "completion": r["counts"].get("completion"),
        "retrieval_recall": m["retrieval_recall"],
        "surfaced_recall": m["tier_surfaced"]["recall"],
        "surfaced_recall_ci": m["ci95"]["surfaced_recall"],
        "surfaced_precision": m["tier_surfaced"]["precision"],
        "pool_base_rate": m["pool_base_rate"],
        "lift": m["surfaced_lift"],
        "lift_ci": m["ci95"]["surfaced_lift"],
        "unverifiable": m["criterion_unverifiable_rate"],
    }


if __name__ == "__main__":
    main()
