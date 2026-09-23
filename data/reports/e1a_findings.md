# E1a — where the in-process dense matrix stops being the right answer

Measured 2026-09-22 on a `c6i.2xlarge` (8 vCPU, 15 GB) with pgvector 0.8.6 in
Docker. 200 queries, k=100, three corpus sizes. Raw arms:
[`e1a_ann_scale.json`](e1a_ann_scale.json).

AD-6 (Phase 9) moved dense search out of pgvector into an 80 MB in-process matrix,
because at 26k rows the planner declined ivfflat and seq-scanned anyway. That
argument is a function of corpus size. This measures the size at which it expires.

## The answer

**The crossover is near 50k rows, and production sits just under it at 26,037.**

| n | matrix p50 | HNSW ef=200 p50 | ratio |
|---|---|---|---|
| 100,000 | 5.84 ms | 3.09 ms | 1.9x |
| 500,000 | 29.32 ms | 2.49 ms | 11.8x |
| 1,000,000 | 58.07 ms | 2.51 ms | **23.1x** |

The matrix is exactly linear in corpus size — 0.058 ms per thousand rows across
the whole range — because a brute-force scan has no other shape. HNSW is flat:
2.5-3.1 ms at every size, which is what a graph index is for. Extrapolating the
matrix line down, the two meet at roughly **50k rows**.

So AD-6 was right when it was written and stays right today, with about 2x of
headroom. It is not a permanent property of the system, and the date it expires
is a corpus-size threshold rather than a guess.

## Full results

Recall is against exact top-100 on the same corpus.

| n | backend | p50 ms | p95 ms | recall | build s |
|---|---|---|---|---|---|
| 100k | matrix (exact) | 5.84 | 5.93 | 1.0000 | 0.1 |
| 100k | ivfflat probes=10 | 4.34 | 6.37 | 0.8677 | 15.9 |
| 100k | ivfflat probes=40 | 13.94 | 17.99 | 0.9688 | 15.9 |
| 100k | ivfflat probes=100 | 32.25 | 34.63 | 0.9969 | 15.9 |
| 100k | hnsw ef=100 | 2.06 | 2.48 | 0.9589 | 53.3 |
| 100k | hnsw ef=200 | 3.09 | 3.80 | 0.9847 | 53.3 |
| 100k | hnsw ef=400 | 5.14 | 6.22 | 0.9947 | 53.3 |
| 500k | matrix (exact) | 29.32 | 29.61 | 1.0000 | 0.3 |
| 500k | ivfflat probes=40 | 36.85 | 48.79 | 0.9825 | 87.2 |
| 500k | ivfflat probes=100 | 87.03 | 103.85 | 0.9981 | 87.2 |
| 500k | hnsw ef=100 | 1.91 | 2.58 | 0.9721 | 136.0 |
| 500k | hnsw ef=200 | 2.49 | 3.25 | 0.9906 | 136.0 |
| 500k | hnsw ef=400 | 3.80 | 4.96 | 0.9922 | 136.0 |
| 1M | matrix (exact) | 58.07 | 58.55 | 1.0000 | 0.7 |
| 1M | ivfflat probes=40 | 70.15 | 97.44 | 0.9802 | 216.2 |
| 1M | ivfflat probes=100 | 170.73 | 207.09 | 0.9960 | 216.2 |
| 1M | hnsw ef=200 | 2.51 | 3.46 | 0.9891 | 447.4 |
| 1M | hnsw ef=400 | 3.87 | 5.24 | 0.9904 | 447.4 |

**HNSW dominates ivfflat at every size on both axes.** At 1M, ivfflat needs
`probes=100` and 170 ms to reach 0.996; HNSW reaches 0.989 in 2.51 ms. The Phase 7
decision to tune `probes` was the right move for the index that was there, but the
index itself is the wrong one above ~100k. HNSW costs roughly 2x the build time
(447 s vs 216 s at 1M), paid once per refresh.

**Storage, which the latency table hides.** pgvector's index is about the size of
the heap, so the database costs roughly twice the raw vectors:

| n | raw vectors | pg heap | pg index | total on disk |
|---|---|---|---|---|
| 100k | 0.307 GB | 0.419 | 0.412 | 0.83 GB |
| 500k | 1.536 GB | 2.097 | 2.052 | 4.15 GB |
| 1M | 3.072 GB | 4.193 | 4.102 | 8.30 GB |

The matrix holds 3.07 GB of RAM at 1M against 8.30 GB of disk for the indexed
table. Which is cheaper depends on whether RAM or storage is the scarce resource,
and on a suspend-happy Fly machine that reloads the matrix on every cold start, it
is not obviously RAM.

## What this does not claim

**The corpus is synthetic and that is a real limit.** Vectors are grown from the
real 26,149-vector TREC 2021 MedCPT index by resampling with replacement and
perturbing. `sigma` was calibrated rather than chosen: at 0.15 the synthetic
corpus reproduces the real corpus's mean top-1 neighbour cosine to within 0.003
(0.8573 vs 0.8548). This answers *where does the architecture break*. It does not
answer *what is production recall at 1M*, which needs real trials (E1c).

**Two rows are contaminated by cold cache and are excluded above.** The first arm
queried after each index build pays the page-cache warm-up. 20 warmup queries were
enough at 100k and 500k and not at 1M, where the index is 4.1 GB against 2 GB of
`shared_buffers`: `ivfflat probes=10` reported p95 306 ms and `hnsw ef=100`
reported p50 76 ms / p95 254 ms. Both are first-after-build. Their recall figures
(0.9381, 0.9677) are unaffected — only the timings are.

**`ef_search` must be at least k.** An earlier sweep at `ef_search=40` with k=100
reported HNSW recall of 0.36-0.40 across all three sizes. That measures the sweep,
not the index: a 40-wide search frontier cannot return 100 good neighbours. The
harness now refuses `ef_search < k` rather than reporting it.

**Single box, n=200 queries, one run.** p95 off 200 samples is indicative. Docker
Postgres at `shared_buffers=2GB` is not Neon's managed configuration.

## Recommendation

Nothing to change now — 26k is under the crossover and the matrix is exact, which
is worth something on its own. The trigger to revisit is corpus size, and it is
specific: **at ~50k rows HNSW matches the matrix on latency, and past ~100k the
matrix is losing badly enough that exactness stops paying for itself.** If E1c's
all-conditions corpus lands near 120k, that crossing has already happened and
`ctgov_live` should move to an HNSW index at `ef_search=200`.
