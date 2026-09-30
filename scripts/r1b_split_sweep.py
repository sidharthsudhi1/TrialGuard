"""R1b: split the list-depth gain into its dense and lexical halves.

Production dense search is the same exact matrix product as eval, but production
lexical search is Postgres FTS (ts_rank_cd), not rank-bm25. So the share of the
gain that carries to the served path for certain is the dense-depth share. $0.

    python scripts/r1b_split_sweep.py trec_2021
    TG_INDEX_EXCLUSION=0 python scripts/r1b_split_sweep.py trec_2022
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from trialguard.eval.cohorts import load_patients
from trialguard.eval.end_to_end import _gold_by_patient
from trialguard.eval.file_index import get_index
from trialguard.eval.ltr import RRF_K, rank_lists
from trialguard.eval.significance import compare_family

COMBOS = {"d50l50": (50, 50), "d200l50": (200, 50), "d50l200": (50, 200), "d200l200": (200, 200)}
DEPTHS = (100, 200)


def main(cohort: str) -> None:
    idx = get_index(cohort)
    ids = np.array(idx._nct_ids)
    corpus = set(idx._nct_ids)
    gold = _gold_by_patient(cohort)
    per: dict[str, list[float]] = {f"{c}@{k}": [] for c in COMBOS for k in DEPTHS}
    for p in load_patients(cohort):
        elig = {n for n, g in gold.get(p["patient_id"], {}).items()
                if g == "eligible" and n in corpus}
        if not elig:
            continue
        rl = rank_lists(idx, p["description"])
        for c, (dp, lp) in COMBOS.items():
            score = np.zeros(len(ids))
            for i, (r, w) in enumerate(zip(rl["lists"], rl["weights"], strict=True)):
                mask = r <= (dp if i % 2 == 0 else lp)  # lists alternate dense, lexical
                score[mask] += w / (RRF_K + r[mask])
            order = ids[np.argsort(-score, kind="stable")]
            for k in DEPTHS:
                per[f"{c}@{k}"].append(len(elig & set(order[:k])) / len(elig))
    fam = {f"{c}@{k}": (per[f"d50l50@{k}"], per[f"{c}@{k}"])
           for c in COMBOS if c != "d50l50" for k in DEPTHS}
    out = {
        "cohort": cohort,
        "mean": {m: round(float(np.mean(v)), 4) for m, v in per.items()},
        "vs_d50l50": compare_family(fam),
    }
    path = Path(f"data/reports/r1b_split_{cohort}.json")
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out["mean"]))
    for name, v in out["vs_d50l50"].items():
        print(f"  {name}: {v['mean_delta']:+.4f} {v['delta_ci']} p_bh {v['p_bh']}")


if __name__ == "__main__":
    main(sys.argv[1])
