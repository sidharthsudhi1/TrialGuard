"""E7: draw and merge the adjudication set the entailment gap needs.

AD-15 put the gap at 36% on 50 hand-adjudicated items, and AD-23 named the
blocker for doing anything about it: several hundred adjudicated items, ideally
two raters, before any NLI threshold is defensible. WS-5b's 50 items were drawn
ad hoc and only the output survived, so there is nothing to extend. This is that
sampler, plus the merge step two raters imply.

It deliberately does not label anything. An LLM judge is ruled out for
enforcement by AD-3 on correlated-error grounds, and AD-15 already considered and
rejected using one offline to size the problem, because the number's whole
purpose is to be trustworthy where the automated path is not. The worksheet ships
with `entails` null and a human fills it in.

    # draw, $0 -- cached analyst calls only, uncached pairs are skipped
    python scripts/e7_entailment_sample.py --draw 300 --cohorts sigir,trec_2021

    # after two raters fill in `entails` independently
    python scripts/e7_entailment_sample.py --merge rater_a.jsonl rater_b.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

REPORT_DIR = Path("data/reports")
EXISTING = REPORT_DIR / "ws5b_entailment_sample.json"
DECISIVE = ("met", "not_met")


def _already_adjudicated() -> set[tuple[str, str, str]]:
    """WS-5b's 50 items, so an expansion never re-asks a settled question."""
    if not EXISTING.exists():
        return set()
    data = json.loads(EXISTING.read_text())
    return {
        (i["patient_id"], i["nct_id"], i["criterion"])
        for c in data["cohorts"].values()
        for i in c["items"]
    }


def draw(cohorts: list[str], n: int, top_k: int, seed: int) -> list[dict]:
    from trialguard.agent.graph import assess
    from trialguard.agent.schema import build_typed_criteria
    from trialguard.eval.end_to_end import _is_cached, _load_corpus, retrieve_for_patients

    os.environ["TG_CACHED_ONLY"] = "1"
    seen = _already_adjudicated()
    pool: list[dict] = []

    for cohort in cohorts:
        rows, _ = retrieve_for_patients(cohort, n_patients=10_000, top_k=top_k)
        corpus = _load_corpus(cohort, {n for r in rows for n in r["retrieved"]})
        for r in rows:
            for nct in r["retrieved"]:
                trial = corpus.get(nct)
                if trial is None or not _is_cached(r["note"], nct):
                    continue
                criteria, truncated = build_typed_criteria(trial)
                if not criteria:
                    continue
                state = assess(
                    r["note"], nct, criteria, trial.get("eligibility_raw", ""),
                    max_retries=0, criteria_truncated=truncated,
                )
                for a in state.get("assessments", []):
                    key = (r["patient_id"], nct, a.get("criterion", ""))
                    if not a.get("grounded") or a.get("verdict") not in DECISIVE:
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    pool.append({
                        "cohort": cohort,
                        "patient_id": r["patient_id"],
                        "nct_id": nct,
                        "kind": a.get("kind", "inclusion"),
                        "criterion": a.get("criterion", ""),
                        "verdict": a["verdict"],
                        "quote": a.get("quote", ""),
                        "grounded_by": a.get("grounded_by", ""),
                        "grounded_in": a.get("grounded_in", ""),
                        "entails": None,
                        "adjudication": "",
                    })

    # Stratify by (cohort, kind) so exclusion criteria are not crowded out by the
    # far more numerous inclusions -- AD-19 put the sharpest slice of this gap on
    # exclusion `not_met`, and a proportional draw would under-sample it.
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict]] = {}
    for item in pool:
        strata.setdefault((item["cohort"], item["kind"]), []).append(item)
    for items in strata.values():
        rng.shuffle(items)

    out: list[dict] = []
    keys = sorted(strata)
    while len(out) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop())
    return out



CSV_FIELDS = ["row", "cohort", "patient_id", "nct_id", "kind", "verdict",
              "criterion", "quote", "grounded_by", "entails", "adjudication"]


def to_csv(src: Path, dest: Path) -> int:
    """JSONL -> CSV, because nobody should hand-edit 300 lines of JSON.

    `row` is carried so a spreadsheet that reorders or sorts can still be mapped
    back; the merge keys on (patient_id, nct_id, criterion) regardless.
    """
    items = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for i, item in enumerate(items):
            row = {k: item.get(k, "") for k in CSV_FIELDS if k != "row"}
            row["row"] = i
            row["entails"] = "" if item.get("entails") is None else str(item["entails"]).lower()
            w.writerow(row)
    return len(items)


def from_csv(src: Path, dest: Path) -> dict:
    """CSV -> JSONL, refusing anything it cannot read as a judgment.

    A blank `entails` is an unrated row and stays null. Anything that is neither
    blank nor a recognised boolean is an error rather than a silent false: a
    typo'd label that reads as "not entailing" would bias the rate in exactly the
    direction the whole exercise is trying to measure.
    """
    true_, false_ = {"true", "t", "yes", "y", "1"}, {"false", "f", "no", "n", "0"}
    out, bad, rated = [], [], 0
    with src.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            raw = (row.get("entails") or "").strip().lower()
            if raw == "":
                entails = None
            elif raw in true_:
                entails = True
                rated += 1
            elif raw in false_:
                entails = False
                rated += 1
            else:
                bad.append({"line": i, "value": row.get("entails")})
                continue
            out.append({
                "cohort": row["cohort"], "patient_id": row["patient_id"],
                "nct_id": row["nct_id"], "kind": row["kind"], "verdict": row["verdict"],
                "criterion": row["criterion"], "quote": row["quote"],
                "grounded_by": row.get("grounded_by", ""),
                "entails": entails, "adjudication": row.get("adjudication", ""),
            })
    if bad:
        raise SystemExit(f"unreadable `entails` values, nothing written: {bad}")
    dest.write_text("\n".join(json.dumps(i) for i in out) + "\n")
    return {"rows": len(out), "rated": rated, "unrated": len(out) - rated}


def _kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def merge(path_a: Path, path_b: Path, out: Path) -> dict:
    def load(p: Path) -> dict[tuple, dict]:
        items = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return {(i["patient_id"], i["nct_id"], i["criterion"]): i for i in items}

    ra, rb = load(path_a), load(path_b)
    shared = sorted(set(ra) & set(rb))
    rated = [k for k in shared if ra[k]["entails"] is not None and rb[k]["entails"] is not None]
    if not rated:
        raise SystemExit("no overlapping items carry a judgment from both raters")

    va = [bool(ra[k]["entails"]) for k in rated]
    vb = [bool(rb[k]["entails"]) for k in rated]
    disagreements = [k for k, x, y in zip(rated, va, vb, strict=True) if x != y]

    agreed = [
        {**ra[k], "entails": va[i], "adjudication_b": rb[k].get("adjudication", "")}
        for i, k in enumerate(rated) if k not in set(disagreements)
    ]
    out.write_text("\n".join(json.dumps(i) for i in agreed) + "\n")

    non_ent = sum(1 for i in agreed if not i["entails"])
    return {
        "rated_by_both": len(rated),
        "agreed": len(agreed),
        "disagreed": len(disagreements),
        "cohen_kappa": round(_kappa(va, vb), 4),
        "raw_agreement": round(sum(x == y for x, y in zip(va, vb, strict=True)) / len(rated), 4),
        "non_entailing_rate_on_agreed": round(non_ent / len(agreed), 4) if agreed else None,
        "disagreements": [list(k) for k in disagreements],
        "note": "Disagreements are excluded from the merged set, not resolved. "
                "They need a third pass; a rate computed over agreed items only is "
                "biased toward the easy cases and must be reported as such.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, help="number of items to draw for adjudication")
    ap.add_argument("--cohorts", default="sigir,trec_2021")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--merge", nargs=2, metavar=("RATER_A", "RATER_B"))
    ap.add_argument("--to-csv", metavar="JSONL", help="worksheet -> CSV for rating")
    ap.add_argument("--from-csv", metavar="CSV", help="rated CSV -> JSONL")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.to_csv:
        dest = Path(args.out or Path(args.to_csv).with_suffix(".csv"))
        n = to_csv(Path(args.to_csv), dest)
        print(json.dumps({"rows": n, "csv": str(dest)}, indent=2))
        return

    if args.from_csv:
        dest = Path(args.out or Path(args.from_csv).with_suffix(".jsonl"))
        print(json.dumps({**from_csv(Path(args.from_csv), dest),
                          "jsonl": str(dest)}, indent=2))
        return

    if args.merge:
        dest = Path(args.out or "data/reports/e7_merged_gold.jsonl")
        summary = merge(Path(args.merge[0]), Path(args.merge[1]), dest)
        print(json.dumps(summary, indent=2))
        Path(str(dest).replace(".jsonl", "_summary.json")).write_text(json.dumps(summary, indent=2))
        return

    if not args.draw:
        ap.error("pass --draw N or --merge A B")

    items = draw(args.cohorts.split(","), args.draw, args.top_k, args.seed)
    dest = Path(args.out or "data/reports/e7_adjudication_worksheet.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(json.dumps(i) for i in items) + "\n")

    by = {}
    for i in items:
        by[(i["cohort"], i["kind"])] = by.get((i["cohort"], i["kind"]), 0) + 1
    print(json.dumps({
        "drawn": len(items),
        "requested": args.draw,
        "by_cohort_kind": {f"{c}/{k}": v for (c, k), v in sorted(by.items())},
        "worksheet": str(dest),
        "instructions": "Fill `entails` (true/false) and `adjudication` (one line of "
                        "reasoning). Question: does the quote establish this verdict? "
                        "Not: is the verdict correct. Two raters work independently on "
                        "separate copies, then --merge.",
    }, indent=2))


if __name__ == "__main__":
    main()
