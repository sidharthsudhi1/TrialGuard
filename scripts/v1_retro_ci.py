"""V1 retro: intervals on the published end-to-end headlines, from committed runs.

Reads the per-patient counts written by `end_to_end` (the v1_retro_* reports,
regenerated cached-only at $0 with every post-H1 flag off so they reproduce the
published table) and reports:

- the patient-bootstrap 95% CI on each headline row,
- the paired Wilcoxon for the 10 -> 100 pool widening, BH-adjusted across cohorts,
- the E3 oncology / non-oncology lift split with its interval,
- the minimum detectable effect on surfaced recall at n=20 and at full cohort size.

    python scripts/v1_retro_ci.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from trialguard.eval.cohorts import load_patients
from trialguard.eval.significance import (
    bootstrap_ci,
    compare_family,
    min_detectable_effect,
    ratio_of_sums,
)

REPORTS = Path("data/reports")
FULL_N = {"trec_2021": 75, "trec_2022": 50}

# Same classifier as scripts/e3_specialty_split.py, so the split matches E3.
ONCOLOGY = re.compile(
    r"\b("
    r"cancer|carcinoma|tumou?r|neoplasm|malignan|metasta|"
    r"lymphoma|leukemia|leukaemia|myeloma|melanoma|sarcoma|glioma|glioblastoma|"
    r"astrocytoma|adenocarcinoma|chemotherap|oncolog"
    r")",
    re.I,
)


def _load(cohort: str, k: int) -> dict:
    return json.loads((REPORTS / f"v1_retro_{cohort.replace('_', '')}_top{k}.json").read_text())


def _per_patient_recall(pp: list[dict]) -> dict[str, float]:
    return {p["patient_id"]: p["surfaced_hit"] / p["gold"] for p in pp}


def _lift(pp: list[dict], base_den: str = "assessed_labelled") -> dict:
    """Surfaced precision over pool base rate. `base_den="assessed"` reproduces
    E3's mixed-denominator figure, kept only to show the size of that error."""

    def col(name):
        return [p[name] for p in pp]

    def stat(idx):
        base = ratio_of_sums(col("assessed_gold"), col(base_den))(idx)
        prec = ratio_of_sums(col("surfaced_hit"), col("surfaced_labelled"))(idx)
        return prec / base if base else float("nan")

    return bootstrap_ci(stat, len(pp))


def main() -> None:
    out: dict = {"headlines": {}, "widening": {}, "specialty": {}, "power": {}}

    for cohort in ("trec_2021", "trec_2022"):
        for k in (10, 100):
            m = _load(cohort, k)["metrics"]
            out["headlines"][f"{cohort}@{k}"] = {
                "n": m["patients"],
                "surfaced_recall": m["tier_surfaced"]["recall"],
                "surfaced_recall_ci": m["ci95"]["surfaced_recall"],
                "surfaced_precision": m["tier_surfaced"]["precision"],
                "pool_base_rate": m["pool_base_rate"],
                "lift": m["surfaced_lift"],
                "lift_ci": m["ci95"]["surfaced_lift"],
            }

    family = {}
    for cohort in ("trec_2021", "trec_2022"):
        r10 = _per_patient_recall(_load(cohort, 10)["metrics"]["per_patient"])
        r100 = _per_patient_recall(_load(cohort, 100)["metrics"]["per_patient"])
        ids = sorted(set(r10) & set(r100))
        family[cohort] = ([r10[i] for i in ids], [r100[i] for i in ids])
        deltas = np.array(family[cohort][1]) - np.array(family[cohort][0])
        sd = float(deltas.std(ddof=1))
        mde20 = min_detectable_effect(deltas)
        out["power"][cohort] = {
            "surfaced_recall_delta_sd": round(sd, 4),
            "mde_at_n20": mde20,
            "mde_at_full_n": round(mde20 * np.sqrt(len(ids) / FULL_N[cohort]), 4),
            "full_n": FULL_N[cohort],
        }
    out["widening"] = compare_family(family)

    # E3's cells: all 75 TREC 2021 patients at top-10, current defaults.
    notes = {p["patient_id"]: p["description"] for p in load_patients("trec_2021")}
    pp = json.loads((REPORTS / "v1_retro_trec2021_all75_top10.json").read_text())
    pp = pp["metrics"]["per_patient"]
    for group in ("all", "oncology", "non_oncology"):
        sub = [
            p for p in pp
            if group == "all"
            or ("oncology" if ONCOLOGY.search(notes[p["patient_id"]]) else "non_oncology") == group
        ]
        out["specialty"][group] = {
            "n": len(sub),
            "lift": _lift(sub),
            "lift_e3_mixed_denominator": _lift(sub, base_den="assessed"),
        }

    path = REPORTS / "v1_retro_ci.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
