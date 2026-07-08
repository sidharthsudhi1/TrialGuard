"""One retrieval experiment per process. Env-driven config, JSON out.

Env:
  TG_EMBED_BACKEND = bge | medcpt
  TG_INDEX_EXCLUSION = 0 | 1
Args: <cohort> <use_keywords:0|1>
Prints one JSON line with recall sweep + config tag.
"""

import json
import os
import sys

from trialguard.eval.file_index import get_index
from trialguard.eval.retrieval_metrics import compute_gold_coverage, evaluate_cohort_multi_k
from trialguard.ingestion.embed import embed_tag

K_LIST = [10, 20, 50, 100, 200]


def main() -> None:
    cohort = sys.argv[1]
    use_keywords = sys.argv[2] == "1"

    idx = get_index(cohort)
    cov = compute_gold_coverage(cohort, idx.corpus_ids())

    def _fn(desc, _src):
        import time
        t0 = time.perf_counter()
        results = idx.search(desc, top_k=max(K_LIST), use_keywords=use_keywords)
        return results, {"total_ms": round((time.perf_counter() - t0) * 1000, 1)}

    res = evaluate_cohort_multi_k(cohort, _fn, K_LIST, gold_coverage=cov["coverage"])
    out = {
        "cohort": cohort,
        "tag": embed_tag(),
        "keywords": use_keywords,
        "coverage": cov["coverage"],
        "n": res["n_patients"],
        "mrr": res["mrr"],
        "p50_ms": res["latency_p50_ms"],
        **{f"r@{k}": res[f"recall@{k}"] for k in K_LIST},
        **{f"r@{k}_adj": res[f"recall@{k}_adj"] for k in K_LIST},
    }
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
