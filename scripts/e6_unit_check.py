"""E6: unit-aware numeric check, scored on the human items, the analyst cache and G2b.

    python scripts/e6_unit_check.py

Writes data/reports/e6_unit_check.json and e6_unit_audit.jsonl (flags for the precision read).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from trialguard.verify.unit_check import criterion_clauses, numeric_conflict

REPORTS = Path("data/reports")
CACHE = Path("data/cache/analyst")
OUT = REPORTS / "e6_unit_check.json"
AUDIT = REPORTS / "e6_unit_audit.jsonl"
SEED = 20261118
SAMPLE = 60


def _gold_items() -> dict:
    rows = [json.loads(line) for line in (REPORTS / "e7_combined_gold.jsonl").open()]
    out = {}
    for source in ("human", "model"):
        items = [r for r in rows if r["label_source"] == source]
        flagged = [r for r in items if numeric_conflict(r["criterion"], r["verdict"], r["quote"])]
        out[source] = {
            "n": len(items),
            "in_scope": sum(1 for r in items if criterion_clauses(r["criterion"])),
            "flags": len(flagged),
            "sound_flagged": sum(1 for r in flagged if r["entails"]),
            "unsound_flagged": sum(1 for r in flagged if not r["entails"]),
            "unsound_total": sum(1 for r in items if not r["entails"]),
            "flagged": [
                {k: r[k] for k in ("criterion", "verdict", "quote", "entails")} for r in flagged
            ],
        }
    return out


def _g2b() -> tuple[dict, set[str], set[str]]:
    gold = json.loads((REPORTS / "g2b_gold.json").read_text())["rows"]
    calls = {}
    for line in (REPORTS / "g2b_calls.jsonl").open():
        row = json.loads(line)
        calls[row["id"]] = row
    caught = false_flags = wrong = right = 0
    for g in gold:
        c = calls[g["id"]]
        if c["verdict"] not in ("met", "not_met"):
            continue
        flag = numeric_conflict(g["criterion"], c["verdict"], c["quote"])
        if (c["verdict"] == "met") == g["holds"]:
            right += 1
            false_flags += bool(flag)
        else:
            wrong += 1
            caught += bool(flag)
    stats = {"wrong": wrong, "caught": caught, "right": right, "false_flags": false_flags}
    return stats, {g["criterion"] for g in gold}, {g["note"] for g in gold}


def _cache_entries(g2b_criteria: set[str], g2b_notes: set[str]) -> tuple[list[dict], int]:
    seen: set[tuple[str, str, str]] = set()
    entries: list[dict] = []
    files = 0
    for path in sorted(CACHE.glob("*.json")):
        files += 1
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        rows = data if isinstance(data, list) else data.get("assessments", [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            verdict, quote = row.get("verdict"), row.get("quote") or ""
            criterion = row.get("criterion") or ""
            if verdict not in ("met", "not_met") or not quote:
                continue
            if criterion in g2b_criteria and any(quote in n for n in g2b_notes):
                continue
            key = (criterion, verdict, quote)
            if key in seen:
                continue
            seen.add(key)
            entries.append({"criterion": criterion, "verdict": verdict, "quote": quote})
    return entries, files


def main() -> None:
    g2b, g2b_criteria, g2b_notes = _g2b()
    entries, files = _cache_entries(g2b_criteria, g2b_notes)
    in_scope = [e for e in entries if criterion_clauses(e["criterion"])]
    flags = []
    for e in in_scope:
        reason = numeric_conflict(e["criterion"], e["verdict"], e["quote"])
        if reason:
            flags.append({**e, "reason": reason})
    audit = flags if len(flags) <= SAMPLE else random.Random(SEED).sample(flags, SAMPLE)
    with AUDIT.open("w") as sink:
        for i, f in enumerate(audit):
            sink.write(json.dumps({"audit_id": i, **f}) + "\n")
    report = {
        "seed": SEED,
        "gold": _gold_items(),
        "cache": {
            "files": files,
            "decisive_unique": len(entries),
            "in_scope": len(in_scope),
            "flags": len(flags),
            "audited": len(audit),
        },
        "g2b_dev": g2b,
    }
    OUT.write_text(json.dumps(report, indent=2))
    printable = {
        **report,
        "gold": {
            k: {x: y for x, y in v.items() if x != "flagged"} for k, v in report["gold"].items()
        },
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
