"""R1 follow-up: per-list pool depth in keyword RRF fusion.

R1's learned ranker added nothing over fusing deeper per-keyword lists (1000
instead of the served 50), and the deeper fusion alone beat the served order on
both TREC cohorts. This sweeps the per-list pool to find where the gain comes
from and where it stops. Same keywords, same index, same weights; only the pool
changes. $0: cached keywords, local embeddings.

    python scripts/r1_pool_sweep.py --cohort trec_2021
    TG_INDEX_EXCLUSION=0 python scripts/r1_pool_sweep.py --cohort trec_2022
    python scripts/r1_pool_sweep.py --cohort sigir
    python scripts/r1_pool_sweep.py --summarise
    python scripts/r1_pool_sweep.py --cohort sigir --pools 50 200 --depths 5 10 25 --tag head
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trialguard.eval.ltr import _fuse, rank_lists

POOLS = (50, 100, 200, 300, 500, 1000)
DEPTHS = (100, 200, 500)
OUT = Path("data/reports")


def sweep(cohort: str, pools=POOLS, depths=DEPTHS, tag: str = "") -> Path:
    from trialguard.eval.cohorts import load_patients
    from trialguard.eval.end_to_end import _gold_by_patient
    from trialguard.eval.file_index import get_index

    idx = get_index(cohort)
    ids = np.array(idx._nct_ids)
    corpus = set(idx._nct_ids)
    gold = _gold_by_patient(cohort)
    per: dict[str, list[float]] = {f"p{p}@{k}": [] for p in pools for k in depths}
    patients = []
    for pat in load_patients(cohort):
        elig = {n for n, g in gold.get(pat["patient_id"], {}).items()
                if g == "eligible" and n in corpus}
        if not elig:
            continue
        rl = rank_lists(idx, pat["description"])
        for p in pools:
            score = _fuse(rl["lists"], min(p, len(ids)), rl["weights"])
            order = ids[np.argsort(-score, kind="stable")]
            for k in depths:
                per[f"p{p}@{k}"].append(len(elig & set(order[:k])) / len(elig))
        patients.append(pat["patient_id"])
    path = OUT / f"r1_pool_sweep_{tag + '_' if tag else ''}{cohort}.json"
    path.write_text(json.dumps({"cohort": cohort, "pools": list(pools), "depths": list(depths),
                                "patients": patients, "per_patient": per}))
    return path


def summarise(tag: str = "") -> dict:
    from trialguard.eval.significance import compare_family

    out = {}
    pattern = f"r1_pool_sweep_{tag}_*.json" if tag else "r1_pool_sweep_trec_*.json"
    paths = sorted(OUT.glob(pattern)) + ([] if tag else sorted(OUT.glob("r1_pool_sweep_sigir.json")))
    for path in paths:
        d = json.loads(path.read_text())
        per = d["per_patient"]
        c = d["cohort"]
        pools, depths = d.get("pools", POOLS), d.get("depths", DEPTHS)
        means = {m: round(float(np.mean(v)), 4) for m, v in per.items()}
        fam = {f"p{p}@{k}": (per[f"p50@{k}"], per[f"p{p}@{k}"])
               for p in pools if p != 50 for k in depths}
        out[c] = {"n": len(d["patients"]), "mean": means, "vs_p50": compare_family(fam)}
        print(f"== {c} (n={len(d['patients'])})")
        for k in depths:
            row = "  ".join(f"p{p}={means[f'p{p}@{k}']:.4f}" for p in pools)
            print(f"  @{k}: {row}")
        for name, v in out[c]["vs_p50"].items():
            print(f"  {name} vs p50: {v['mean_delta']:+.4f} {v['delta_ci']} "
                  f"{v['better']}/{v['worse']} p_bh {v['p_bh']}")
    name = f"r1_pool_sweep_{tag}_summary.json" if tag else "r1_pool_sweep_summary.json"
    (OUT / name).write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["trec_2021", "trec_2022", "sigir"])
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--pools", type=int, nargs="+", default=list(POOLS))
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.cohort:
        print(sweep(args.cohort, tuple(args.pools), tuple(args.depths), args.tag))
    if args.summarise:
        summarise(args.tag)


if __name__ == "__main__":
    main()
