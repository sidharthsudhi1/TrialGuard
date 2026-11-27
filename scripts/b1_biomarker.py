"""B1 (Phase 39): biomarker contradiction and observational share, measured only.

    python scripts/b1_biomarker.py prepare       # jobs for Modal, $0
    modal run scripts/modal_b1_retrieve.py       # rankings for marked notes
    python scripts/b1_biomarker.py score         # contradiction rates, $0
    python scripts/b1_biomarker.py study-types   # CT.gov lookups + one read-only Neon query

Oncology patients are E3's: the disease-stem regex over the note, restricted to
patients with at least one gold-eligible trial. Only notes that state a
biomarker status need a ranking; an unmarked note scores 0 contradictions by
construction. The metamorphic set flips every stated status in the note and in
its cached keywords (no keyword LLM call), then re-retrieves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from e3_specialty_split import specialty  # noqa: E402

from trialguard.eval.biomarker_contradiction import (  # noqa: E402
    flip_keywords,
    flip_text,
    note_statuses,
    observational_share,
    score,
)
from trialguard.eval.cohorts import load_labels, load_patients  # noqa: E402
from trialguard.eval.significance import (  # noqa: E402
    bootstrap_ci,
    paired_bootstrap_ci,
    ratio_of_sums,
)

COHORTS = ("sigir", "trec_2021", "trec_2022", "trec_2023")
K = 25
JOBS = Path("data/reports/b1_jobs.json")
RANKINGS = Path("data/reports/b1_rankings.json")
TRIALS = Path("data/reports/b1_top_trials.json")
V4_RANKINGS = Path("data/reports/v4_rankings.json")
V4_TRIALS = Path("data/reports/v4_top_trials.json")
V4_KEYWORDS = Path("data/reports/v4_keywords.json")
CONTRADICTION = Path("data/reports/b1_contradiction.json")
STUDY_TYPES = Path("data/reports/b1_study_types.json")
AUDIT = Path("data/reports/b1_audit.json")
TYPE_CACHE = Path("data/cache/study_type.json")


def _eligible(cohort: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for lab in load_labels(cohort):
        if lab["label"] == "eligible":
            out.setdefault(lab["patient_id"], set()).add(lab["nct_id"])
    return out


def oncology_patients(cohort: str) -> list[dict]:
    gold = _eligible(cohort)
    return [
        p for p in load_patients(cohort)
        if p["patient_id"] in gold and specialty(p["description"]) == "oncology"
    ]


def _cached_keywords(cohort: str, note: str, v4: dict[str, list[str]]) -> list[str]:
    if cohort == "trec_2023":
        return v4[note]
    from trialguard.retrieval.query_transform import CACHE_DIR, _note_hash

    path = CACHE_DIR / f"{_note_hash(note)}.json"
    if not path.exists():
        raise FileNotFoundError(f"no cached keywords for a {cohort} note; refusing an LLM call")
    return json.loads(path.read_text())


def prepare() -> None:
    v4 = {p["note"]: p["keywords"] for p in json.loads(V4_KEYWORDS.read_text())["patients"]}
    jobs = []
    for cohort in COHORTS:
        for p in oncology_patients(cohort):
            note = p["description"]
            if not note_statuses(note):
                continue
            keywords = _cached_keywords(cohort, note, v4)
            base = {"cohort": cohort, "patient_id": p["patient_id"]}
            if cohort != "trec_2023":
                jobs.append({**base, "variant": "original", "note": note, "keywords": keywords})
            flipped = flip_text(note)
            kept, dropped = flip_keywords(keywords, flipped)
            jobs.append({
                **base,
                "variant": "flipped",
                "note": flipped,
                "keywords": kept,
                "dropped_keywords": dropped,
            })
    JOBS.write_text(json.dumps({"k": K, "jobs": jobs}, indent=1))
    print(f"wrote {JOBS}: {len(jobs)} jobs")


def _rows(variant: str) -> tuple[dict[str, list[dict]], dict[str, str]]:
    ranked = json.loads(RANKINGS.read_text())
    trials = {n: t["text"] for n, t in json.loads(TRIALS.read_text()).items()}
    v4_rank = json.loads(V4_RANKINGS.read_text())
    for n, t in json.loads(V4_TRIALS.read_text()).items():
        trials.setdefault(n, t["text"])
    by_cohort: dict[str, list[dict]] = {}
    for cohort in COHORTS:
        rows = []
        for p in oncology_patients(cohort):
            note = p["description"]
            key = f"{cohort}|{p['patient_id']}|{variant}"
            if variant == "flipped":
                note = flip_text(note)
            if not note_statuses(note):
                if variant == "flipped":
                    continue  # nothing stated, nothing to flip
                retrieved = None
            elif key in ranked:
                retrieved = ranked[key]
            elif cohort == "trec_2023" and variant == "original":
                retrieved = v4_rank[p["patient_id"]]["keyword"]
            else:
                raise KeyError(f"no ranking for {key}; run modal_b1_retrieve.py")
            rows.append({"patient_id": p["patient_id"], "note": note, "retrieved": retrieved})
        by_cohort[cohort] = rows
    return by_cohort, trials


def _audited(block: dict, variant: str) -> dict:
    """The pooled rate after removing pairs the hand audit found false."""
    false = {
        (f["patient_id"], f["nct_id"])
        for f in json.loads(AUDIT.read_text())["false_positives"] if f["variant"] == variant
    }
    pp = block["per_patient"]
    kept = [[n for n in p["contradicted_ids"] if (p["patient_id"], n) not in false] for p in pp]
    num, den = [len(k) for k in kept], [p["surfaced"] for p in pp]
    marked = [i for i, p in enumerate(pp) if p["stated"]]
    flagged = sum(p["contradicted"] for p in pp)
    return {
        "flagged": flagged,
        "confirmed": sum(num),
        "precision": round(sum(num) / flagged, 4) if flagged else None,
        "rate": bootstrap_ci(ratio_of_sums(num, den), len(pp)),
        "rate_given_marker": bootstrap_ci(
            ratio_of_sums([num[i] for i in marked], [den[i] for i in marked]), len(marked)
        ),
    }


def _paired(original: dict, flipped: dict) -> dict:
    """Audited per-patient rate, flipped minus original, over the marked notes."""
    false = {
        (f["variant"], f["patient_id"], f["nct_id"])
        for f in json.loads(AUDIT.read_text())["false_positives"]
    }

    def rates(block, variant):
        return {
            p["patient_id"]: sum((variant, p["patient_id"], n) not in false
                                 for n in p["contradicted_ids"]) / p["surfaced"]
            for p in block["per_patient"] if p["stated"]
        }

    a, b = rates(original, "original"), rates(flipped, "flipped")
    ids = sorted(a.keys() & b.keys())
    return {
        "patients": ids,
        "original": [round(a[i], 4) for i in ids],
        "flipped": [round(b[i], 4) for i in ids],
        "diff": paired_bootstrap_ci([a[i] for i in ids], [b[i] for i in ids]),
    }


def score_all() -> None:
    out: dict = {"k": K, "classifier": "E3 disease-stem regex, gold-eligible patients only"}
    for variant in ("original", "flipped"):
        by_cohort, trials = _rows(variant)
        block = {c: score(rows, trials, K) for c, rows in by_cohort.items()}
        block["pooled"] = score([r for c in COHORTS for r in by_cohort[c]], trials, K)
        block["audited"] = _audited(block["pooled"], variant)
        out[variant] = block
    out["paired_flip"] = _paired(out["original"]["pooled"], out["flipped"]["pooled"])
    # What a biomarker-blind retriever would score: the original ranking, unchanged,
    # against the flipped statuses. Detector-flagged, not hand-audited.
    by_cohort, trials = _rows("original")
    blind = [
        {**r, "note": flip_text(r["note"])}
        for c in COHORTS for r in by_cohort[c] if r["retrieved"] is not None
    ]
    out["blind_baseline"] = score(blind, trials, K)
    CONTRADICTION.write_text(json.dumps(out, indent=1))
    for variant in ("original", "flipped"):
        p = out[variant]["pooled"]
        print(variant, p["patients"], p["surfaced"], p["contradicted"], p["rate"],
              p["rate_given_marker"])
        print("  audited", out[variant]["audited"])
    print("paired", out["paired_flip"])
    print("blind", out["blind_baseline"]["contradicted"], out["blind_baseline"]["rate"])


def study_types() -> None:
    from trialguard.ingestion.ctgov import lookup_study_types

    cache = json.loads(TYPE_CACHE.read_text()) if TYPE_CACHE.exists() else {}

    def types_for(ids: set[str]) -> dict[str, str | None]:
        todo = sorted(ids - cache.keys())
        if todo:
            print(f"looking up {len(todo)} ids on CT.gov", flush=True)
            cache.update(lookup_study_types(todo))
            TYPE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            TYPE_CACHE.write_text(json.dumps(cache))
        return {n: cache.get(n) for n in ids}

    out: dict = {"gold_eligible": {}}
    for cohort in COHORTS:
        ids = set().union(*_eligible(cohort).values())
        out["gold_eligible"][cohort] = observational_share(ids, types_for(ids))
        print(cohort, out["gold_eligible"][cohort], flush=True)

    from trialguard.db.schema import get_conn

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT nct_id FROM trials WHERE source = 'ctgov_live' AND expired_at IS NULL"
        )
        served = {r[0] for r in cur.fetchall()}
    out["served_corpus"] = observational_share(served, types_for(served))
    print("served", out["served_corpus"])
    STUDY_TYPES.write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prepare", "score", "study-types"))
    step = ap.parse_args().step
    {"prepare": prepare, "score": score_all, "study-types": study_types}[step]()


if __name__ == "__main__":
    main()
