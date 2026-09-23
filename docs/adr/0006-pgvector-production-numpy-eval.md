# AD-6 — pgvector for production, numpy file index for eval

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

pgvector (production) + numpy file index (eval)

## Alternatives considered

Load 26k eval trials into Neon free tier (too small)

## Amended, Phase 5

Benchmark ([`phase5_vectorstore.md`](../../data/reports/phase5_vectorstore.md)) confirms the split: numpy brute is exact and sub-ms at eval scale, and the free-tier 512 MB ceiling (full corpus ≈ 1.5 GB of vectors) is the real driver. Amendment: production ivfflat at default `probes=1` loses ~64% recall vs exact; `dense_search` now sets `ivfflat.probes` (default 20) to recover it at near-flat latency

**Alternatives considered.** Silent default-probes recall loss; managed alternative (size ceiling, not engine, is binding)

## Amended, Phase 7

Neon upgraded off the free tier; production `ctgov_live` populated with 25,965 recruiting oncology trials (531 MB, over the 512 MB free ceiling). Lexical moved from in-memory BM25 to Postgres FTS (tsvector + GIN) to serve at corpus scale; `probes` retuned to 40 (20 recovered only ~62% of exact). See [`phase7_retrieval.md`](../../data/reports/phase7_retrieval.md)

**Alternatives considered.** Staying on the free tier (corpus no longer fits); keeping in-memory BM25 (rebuilt per process, doesn't scale)

## Amended again, Phase 9

pgvector stays the store of record, but dense search is served from an 80 MB in-process matrix. `EXPLAIN` showed the planner declining ivfflat at `probes=40` and scanning the table, so production had been running exact search at 320–413 ms per query; the same exact search takes 3.8–6.7 ms locally, with identical rankings. `eval/file_index.py` had always done this — the two paths differed by label, not by measurement

**Alternatives considered.** Keeping dense search in Postgres; tuning `probes` further, which the planner ignores
