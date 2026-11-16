# S1: one GEMM for the dense fan-out

Run 2026-11-10 on `trec_2021_medcpt_excl_embeddings.npy`, 26149 x 768, 12 query vectors, top-500. Adopt bar, written before the run: top-500 id sets equal, p95 at concurrency 1 no worse than the per-query loop, throughput at least 2x at concurrency 8. Raw: `s1_gemm.json`.

| | p95 ms |
|---|---|
| c=1, one product per query | 56.14 |
| c=1, one GEMM | 18.04 |
| c=1, GEMM with BLAS threads capped at 1 | 16.76 |
| c=8, one product per query | 244.15 |
| c=8, one GEMM | 96.28 |

Top-500 id sets match on all 12 queries. Speedup at c=8 is 2.54x. The c=1 GEMM is faster, not slower. Float16 storage does not keep the top-500 set, so the matrix stays float32.

## Decision

DONE-ADOPTED for the single product (`retrieval/gemm.py`, used by the file index and by `vector_cache.search_queries` on the keyword path). Float16 is not adopted. The BLAS cap stays at 0: the uncapped GEMM is the arm that cleared the bar.
