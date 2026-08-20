# Phase 2 Retrieval Metrics

Generated: 2026-08-19 20:42

Retriever: dense (BGE) + BM25 fused with RRF (k=60) — keyword-RRF

## Gold Coverage

| Cohort | Config | Total gold | In corpus | Missing | Coverage |
|---|---|---|---|---|---|
| trec_2021 | keyword | 5124 | 5123 | 1 | 100.0% |

## Recall@N Sweep

| Cohort | Config | Recall@10 | Recall@20 | Recall@50 | Recall@100 | Recall@200 | MRR | p50 ms | p95 ms | n |
|---|---|---|---|---|---|---|---|---|---|---|
| trec_2021 | keyword | 0.0798 | 0.1455 | 0.2859 | 0.4227 | 0.5453 | 0.6257 | 6061.5 | 15087.5 | 75 |

## Coverage-Adjusted Recall

| Cohort | Config | Coverage | @50 raw | @50 adj | @100 raw | @100 adj | @200 raw | @200 adj |
|---|---|---|---|---|---|---|---|---|
| trec_2021 | keyword | 100.0% | 0.2859 | 0.2860 | 0.4227 | 0.4228 | 0.5453 | 0.5454 |

## Agent Recall Ceiling (pool N=50)

- **trec_2021 / keyword**: recall@50 = 0.2859 raw | 0.2860 coverage-adjusted

## Notes
- Coverage < 1.0 means some gold trials were never loaded into the eval corpus.
- Coverage-adjusted recall = raw recall / gold_coverage (ceiling achievable by retrieval alone).
- Pool size N=50 is the assumed Phase 4 candidate pool passed to the agent.
- Keyword latency cached; zero LLM calls after first run.