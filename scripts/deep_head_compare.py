"""How much of a progressive deep job's answer is already in its head?

A deep job assesses the top 100 in rank order and signals once the top 25 are
done. Top-25 is an exact prefix of top-100 when list depth is pinned, so the
head's result reruns cached-only at $0. Compares each top-25 report with the
top-100 report it is a prefix of: surfaced recall, the share of the top-100
answer the head already holds, lift, and the paired 25 -> 100 gain.

    python scripts/deep_head_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

from trialguard.eval.significance import compare_family

REPORTS = Path("data/reports")
CELLS = {
    # (head report, full report). TREC: served deep path (list pool 200).
    # SIGIR: V2's top-100 ran at list pool 50, so its head does too.
    "trec_2021:deep": ("deep_head_deep_trec_2021_top25.json", "r2e2e_deep_trec_2021_top100.json"),
    "trec_2021:ce": ("deep_head_ce_trec_2021_top25.json", "r2e2e_ce_trec_2021_top100.json"),
    "trec_2022:deep": ("deep_head_deep_trec_2022_top25.json", "r2e2e_deep_trec_2022_top100.json"),
    "trec_2022:ce": ("deep_head_ce_trec_2022_top25.json", "r2e2e_ce_trec_2022_top100.json"),
    "sigir:deep": ("deep_head_deep_sigir_top25.json", "v2_e2e_sigir_top100.json"),
}


def _per(r: dict) -> dict[str, dict]:
    return {p["patient_id"]: p for p in r["metrics"]["per_patient"]}


def _summary(r: dict) -> dict:
    m = r["metrics"]
    return {
        "completion": r["counts"].get("completion"),
        "cache_coverage": r["counts"].get("cache_coverage"),
        "surfaced_recall": m["tier_surfaced"]["recall"],
        "surfaced_recall_ci": m["ci95"]["surfaced_recall"],
        "surfaced_precision": m["tier_surfaced"]["precision"],
        "pool_base_rate": m["pool_base_rate"],
        "lift": m["surfaced_lift"],
        "lift_ci": m["ci95"]["surfaced_lift"],
    }


def main() -> None:
    family, out = {}, {"cells": {}}
    for cell, (head_f, full_f) in CELLS.items():
        head = json.loads((REPORTS / head_f).read_text())
        full = json.loads((REPORTS / full_f).read_text())
        h, f = _per(head), _per(full)
        ids = sorted(set(h) & set(f))
        family[cell] = (
            [h[i]["surfaced_hit"] / h[i]["gold"] for i in ids],
            [f[i]["surfaced_hit"] / f[i]["gold"] for i in ids],
        )
        hits_h = sum(h[i]["surfaced_hit"] for i in ids)
        hits_f = sum(f[i]["surfaced_hit"] for i in ids)
        # Patients for whom the head already shows at least one eligible trial,
        # out of those the full run shows one for: "does the user see something
        # useful at the banner?"
        some_f = [i for i in ids if f[i]["surfaced_hit"] > 0]
        out["cells"][cell] = {
            "n": len(ids),
            "head": _summary(head),
            "full": _summary(full),
            "head_share_of_surfaced_hits": round(hits_h / hits_f, 4) if hits_f else None,
            "patients_with_a_hit_in_head": sum(1 for i in some_f if h[i]["surfaced_hit"] > 0),
            "patients_with_a_hit_in_full": len(some_f),
        }
    out["paired_head_to_full"] = compare_family(family)
    (REPORTS / "deep_head_compare.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    for cell, v in out["cells"].items():
        p = out["paired_head_to_full"][cell]
        print(f"{cell}: n={v['n']} head {v['head']['surfaced_recall']:.3f} "
              f"full {v['full']['surfaced_recall']:.3f} share {v['head_share_of_surfaced_hits']} "
              f"hit-patients {v['patients_with_a_hit_in_head']}/{v['patients_with_a_hit_in_full']} "
              f"lift {v['head']['lift']:.2f}x/{v['full']['lift']:.2f}x "
              f"coverage {v['head']['cache_coverage']} gain {p['mean_delta']:+.3f} p_bh {p['p_bh']}")


if __name__ == "__main__":
    main()
