"""G2b: the served analyst graph on strict synthetic gold. Isolated ledger, hard stop $1.

    python scripts/g2_analyst.py

Resumes from data/reports/g2b_calls.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPORTS = Path("data/reports")
GOLD = REPORTS / "g2b_gold.json"
CALLS = REPORTS / "g2b_calls.jsonl"
OUT = REPORTS / "g2b_results.json"
LEDGER = Path("data/cache/g2b_ledger.json")


def _verdict(state: dict, criterion: str) -> tuple[str, str]:
    for a in state.get("assessments", []):
        if a.get("criterion", "").strip() == criterion.strip():
            return a.get("verdict", "cannot_determine"), a.get("quote", "")
    rows = state.get("assessments", [])
    if len(rows) == 1:
        return rows[0].get("verdict", "cannot_determine"), rows[0].get("quote", "")
    return "cannot_determine", ""


def main() -> None:
    os.environ["TG_PROMPT_VERSION"] = "v4"
    os.environ["TG_ANALYST_DELAY"] = "0"
    os.environ["TG_LLM_TIMEOUT"] = "300"
    from trialguard.agent.graph import assess
    from trialguard.config import settings
    from trialguard.eval.spend import isolated_ledger
    from trialguard.eval.synthetic_gold import score_rows

    settings.llm_provider = "deepinfra"
    ledger = isolated_ledger(LEDGER, 1.0)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    gold = json.loads(GOLD.read_text())["rows"][:limit]
    done = {}
    if CALLS.exists():
        for line in CALLS.read_text().splitlines():
            row = json.loads(line)
            done[row["id"]] = row

    def one(row: dict) -> dict:
        try:
            state = assess(
                row["note"],
                row["nct_id"],
                [{"text": row["criterion"], "kind": "inclusion"}],
                row["criterion"],
                max_retries=2,
            )
        except Exception as exc:
            return {"id": row["id"], "error": f"{type(exc).__name__}: {exc}"[:300]}
        verdict, quote = _verdict(state, row["criterion"])
        return {
            "id": row["id"],
            "verdict": verdict,
            "quote": quote,
            "retries": state.get("retries", 0),
        }

    todo = [r for r in gold if r["id"] not in done or "error" in done[r["id"]]]
    with ThreadPoolExecutor(max_workers=8) as pool, CALLS.open("a") as sink:
        for result in pool.map(one, todo):
            done[result["id"]] = result
            sink.write(json.dumps(result) + "\n")
            sink.flush()
            print(result["id"], result.get("verdict", result.get("error")), flush=True)

    errors = [r for r in done.values() if "error" in r]
    report = {
        "n_gold": len(gold),
        "errors": len(errors),
        "spent_usd": round(ledger.spent_usd(), 6),
        "prompt_version": "v4",
        "model": settings.deepinfra_model,
        **score_rows(gold, done),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "by_attr"}, indent=2))


if __name__ == "__main__":
    main()
