"""E3: does the system work outside oncology, on the cohort it already has?

CLAUDE.md scoped production to oncology, and the standing end-to-end numbers were
read as oncology numbers because of it. They are not: 63 of TREC 2021's 75 topics
never mention a cancer term, and gold coverage is 1.000 for both groups, so the
headline lift was always computed over a mix that is 84% non-oncology without
saying so.

This adds no data. It partitions the same patients and scores each group with
`end_to_end.score` unchanged, so the groups sum to the published aggregate by
construction and any difference between them is a property of the system rather
than of a new measurement.

    python scripts/e3_specialty_split.py --cohort trec_2021 --top-k 100 --cached-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

# Matched against the patient note. Deliberately a recall-oriented list of disease
# terms rather than treatments: "radiation" and "port" appear in non-oncology notes
# often enough to misfile them, while these stems essentially do not.
ONCOLOGY = re.compile(
    r"\b("
    r"cancer|carcinoma|tumou?r|neoplasm|malignan|metasta|"
    r"lymphoma|leukemia|leukaemia|myeloma|melanoma|sarcoma|glioma|glioblastoma|"
    r"astrocytoma|adenocarcinoma|chemotherap|oncolog"
    r")",
    re.I,
)


def specialty(note: str) -> str:
    return "oncology" if ONCOLOGY.search(note) else "non_oncology"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="trec_2021")
    ap.add_argument("--n-patients", type=int, default=75)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="skip the agent; retrieval recall costs no LLM calls")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from trialguard.eval.end_to_end import (
        assess_retrieved,
        retrieve_for_patients,
        score,
    )

    t0 = time.perf_counter()
    rows, r_timing = retrieve_for_patients(args.cohort, args.n_patients, args.top_k)
    for r in rows:
        r["specialty"] = specialty(r["note"])

    groups = {
        "all": rows,
        "oncology": [r for r in rows if r["specialty"] == "oncology"],
        "non_oncology": [r for r in rows if r["specialty"] == "non_oncology"],
    }

    out: dict = {
        "cohort": args.cohort,
        "top_k": args.top_k,
        "retrieval_only": args.retrieval_only,
        "cached_only": args.cached_only,
        "classifier": "regex over patient note, disease stems only",
        "timing": r_timing,
        "groups": {},
    }

    # Retrieval recall is free and does not need the agent, so it is always
    # reported; the pool ceiling is what bounds everything downstream.
    for name, sub in groups.items():
        if not sub:
            continue
        ceil = [
            len(set(r["retrieved"]) & set(r["gold_eligible"])) / len(r["gold_eligible"])
            for r in sub
        ]
        out["groups"][name] = {
            "patients": len(sub),
            "gold_eligible_total": sum(len(r["gold_eligible"]) for r in sub),
            "retrieval_ceiling": round(sum(ceil) / len(ceil), 4),
        }

    if not args.retrieval_only:
        if args.cached_only:
            os.environ["TG_CACHED_ONLY"] = "1"
        # Assess once over every row, then score the partitions. Assessing per
        # group would re-run shared work and let the two halves drift apart.
        a_timing = assess_retrieved(rows, args.cohort, cached_only=args.cached_only)
        out["counts"] = a_timing
        for name, sub in groups.items():
            if sub:
                out["groups"][name]["metrics"] = score(sub)

    out["timing"]["total_s"] = round(time.perf_counter() - t0, 1)
    dest = Path(args.out or f"data/reports/e3_specialty_{args.cohort}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["groups"], indent=2)[:2000])
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
