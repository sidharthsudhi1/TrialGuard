"""E1-E3 against the WS-5b gold. The badge bar is the one written before this run.

Sound citations damaged is the first number. A badge ships only when a check
flags at most 2 sound human citations and at least 80% of its flags are unsound.
"""

from __future__ import annotations

import json
from pathlib import Path

from trialguard.eval.nli_probe import load_gold
from trialguard.verify.entailment import (
    context_conflict,
    numeric_conflict,
    uncovered_conjunct,
)

REPORT = Path("data/reports/e1e3_entailment.json")
GOLD = Path("data/reports/e7_combined_gold.jsonl")


def _flags(item: dict) -> dict[str, str | None]:
    note = item.get("patient_text") or ""
    return {
        "e1": context_conflict(item["criterion"], item.get("quote") or "", item["verdict"]),
        "e2": numeric_conflict(item["criterion"], item["verdict"], note) if note else None,
        "e3": uncovered_conjunct(item["criterion"], item.get("quote") or "", item["verdict"]),
    }


def _score(items: list[dict], key: str) -> dict:
    flagged = [it for it in items if it["flags"][key]]
    damaged = [it for it in flagged if it["entails"]]
    caught = [it for it in flagged if not it["entails"]]
    unsound = [it for it in items if not it["entails"]]
    precision = (len(caught) / len(flagged)) if flagged else None
    return {
        "n": len(items),
        "flagged": len(flagged),
        "sound_citations_damaged": len(damaged),
        "unsound_caught": len(caught),
        "unsound": len(unsound),
        "precision": None if precision is None else round(precision, 4),
        "recall": round(len(caught) / len(unsound), 4) if unsound else None,
        "badge": bool(flagged) and len(damaged) <= 2 and precision is not None and precision >= 0.8,
    }


def main() -> None:
    items = load_gold(GOLD)
    # The 50 human items do not carry the note. E2 needs it, so attach the
    # cohort note when the quote can be found there. E1 and E3 read the quote.
    from trialguard.eval.cohorts import load_patients

    notes: dict[tuple[str, str], str] = {
        (c, str(p["patient_id"])): str(p["description"])
        for c in ("sigir", "trec_2021", "trec_2022")
        for p in load_patients(c)
    }
    scored = []
    for item in items:
        note = notes.get((str(item.get("cohort") or ""), str(item.get("patient_id") or "")), "")
        row = {**item, "patient_text": note}
        row["flags"] = _flags(row)
        scored.append(row)
    human = [s for s in scored if s["label_source"] == "human"]
    model = [s for s in scored if s["label_source"] != "human"]
    report = {
        "human": {k: _score(human, k) for k in ("e1", "e2", "e3")},
        "model_labelled_not_used": {k: _score(model, k) for k in ("e1", "e2", "e3")},
        "n_human": len(human),
        "n_model": len(model),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
