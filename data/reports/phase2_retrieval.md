# Phase 2 Retrieval Metrics

Generated: 2026-06-26 23:42

Retriever: dense (BGE) + BM25 fused with RRF (k=60) — ablation: raw-note vs keyword-RRF

## Gold Coverage

| Cohort | Config | Total gold | In corpus | Missing | Coverage |
|---|---|---|---|---|---|
| sigir | raw-note | 411 | 396 | 15 | 96.4% |
| sigir | keyword | 411 | 396 | 15 | 96.4% |

## Recall@N Sweep

| Cohort | Config | Recall@10 | Recall@20 | Recall@50 | Recall@100 | Recall@200 | MRR | p50 ms | p95 ms | n |
|---|---|---|---|---|---|---|---|---|---|---|
| sigir | raw-note | 0.0898 | 0.1307 | 0.2749 | 0.4001 | 0.4001 | 0.1911 | 74.0 | 151.2 | 53 |
| sigir | keyword | 0.1345 | 0.3032 | 0.5335 | 0.6919 | 0.7840 | 0.2843 | 224.0 | 250.7 | 53 |

## Coverage-Adjusted Recall

| Cohort | Config | Coverage | @50 raw | @50 adj | @100 raw | @100 adj | @200 raw | @200 adj |
|---|---|---|---|---|---|---|---|---|
| sigir | raw-note | 96.4% | 0.2749 | 0.2853 | 0.4001 | 0.4153 | 0.4001 | 0.4153 |
| sigir | keyword | 96.4% | 0.5335 | 0.5537 | 0.6919 | 0.7181 | 0.7840 | 0.8137 |

## Agent Recall Ceiling (pool N=50)

- **sigir / raw-note**: recall@50 = 0.2749 raw | 0.2853 coverage-adjusted
- **sigir / keyword**: recall@50 = 0.5335 raw | 0.5537 coverage-adjusted

## Notes
- Coverage < 1.0 means some gold trials were never loaded into the eval corpus.
- Coverage-adjusted recall = raw recall / gold_coverage (ceiling achievable by retrieval alone).
- Pool size N=50 is the assumed Phase 4 candidate pool passed to the agent.
- Keyword latency cached; zero LLM calls after first run.