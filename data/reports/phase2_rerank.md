# Phase 2 Rerank Pool-Compression Experiment

Generated: 2026-06-27 00:39

Hypothesis: recall@50(retrieve@200 → rerank → top-50) > recall@50(retrieve@50) = baseline

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU). Query: full patient note.

## Results

| Cohort | Config | recall@10 | recall@20 | recall@50 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| sigir | keyword@50 (no rerank) | 0.1345 | 0.3032 | 0.5335 | 0.2835 | 233.1 | 261.4 |
| sigir | keyword@200→rerank→50 | 0.1347 | 0.2356 | 0.3605 | 0.2330 | 244.1 | 40988.0 |

## Decision

**sigir**: **FALL BACK** to N=100 no-rerank (recall@100 = 0.6919). Rerank lift Δ=-0.1730 insufficient (recall@50 = 0.3605 < 0.65). Accept 2× agent calls at N=100.

## Notes
- Rerank query = full patient note (not keywords). Keywords cast wide; note judges precisely.
- Cross-encoder scores cached per patient note hash — re-runs cost zero model calls.
- General-domain reranker (ms-marco). If underperforming, clinical cross-encoder is next lever.
- Baseline recall@50 = 0.5335 (keyword retrieve@50, no rerank).