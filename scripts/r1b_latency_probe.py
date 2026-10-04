"""R1b: what a 200-deep per-keyword lexical list costs on the served path.

Production lexical search is Postgres FTS. This runs the served query under
EXPLAIN ANALYZE at LIMIT 50 and LIMIT 200 against the production corpus inside a
READ ONLY transaction, so it measures server-side execution time without
network noise and cannot write. It also records how often a keyword's match set
even reaches the limit: below it, a deeper LIMIT changes nothing.

Keywords come from the committed keyword cache, so this spends nothing.

    python scripts/r1b_latency_probe.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from trialguard.retrieval.bm25 import SQL

POOLS = (50, 200)
N_KEYWORDS = 60
REPS = 3


def _keywords(n: int) -> list[str]:
    kws: list[str] = []
    for p in sorted(Path("data/cache/keywords").glob("*.json")):
        data = json.loads(p.read_text())
        kws += data if isinstance(data, list) else data.get("keywords", [])
    random.Random(0).shuffle(kws)
    return kws[:n]


def fts(keywords: list[str]) -> dict:
    from trialguard.db.schema import get_conn

    sql = SQL.format(source_clause="AND source = %(source)s")
    ms: dict[int, list[float]] = {p: [] for p in POOLS}
    rows: dict[int, list[int]] = {p: [] for p in POOLS}
    rng = random.Random(1)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")

        def run(kw: str, p: int) -> dict:
            cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql,
                        {"q": kw, "top_k": p, "source": "ctgov_live"})
            return cur.fetchone()[0][0]

        # Warm pass first: whichever limit ran first would otherwise pay the cold
        # buffer cost for both, and the first draft of this probe did exactly that.
        for kw in keywords:
            run(kw, max(POOLS))
        for kw in keywords:
            reps: dict[int, list[float]] = {p: [] for p in POOLS}
            got: dict[int, int] = {}
            for _ in range(REPS):
                for p in rng.sample(POOLS, len(POOLS)):
                    plan = run(kw, p)
                    reps[p].append(plan["Execution Time"])
                    got[p] = plan["Plan"].get("Actual Rows", 0)
            for p in POOLS:
                ms[p].append(float(np.median(reps[p])))
                rows[p].append(got[p])
        conn.rollback()
    return {
        f"limit_{p}": {
            "p50_ms": round(float(np.percentile(ms[p], 50)), 2),
            "p95_ms": round(float(np.percentile(ms[p], 95)), 2),
            "mean_rows": round(float(np.mean(rows[p])), 1),
            "share_reaching_limit": round(float(np.mean([r >= p for r in rows[p]])), 3),
        }
        for p in POOLS
    }


def main() -> None:
    kws = _keywords(N_KEYWORDS)
    result = {"n_keywords": len(kws), "fts": fts(kws)}
    path = Path("data/reports/r1b_latency_probe.json")
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
