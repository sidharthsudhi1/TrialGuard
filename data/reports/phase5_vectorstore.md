# Vector-store benchmark — pgvector vs local exact (Phase 5, WS-5)

Confirms AD-6 (pgvector for production, numpy FileIndex for eval) with measured
numbers, and surfaces one real production gap: the ivfflat index is not
recall-equivalent to the exact eval backend at its default setting.

**Setup.** 2,991 MedCPT (768-dim) SIGIR trial vectors from the cached eval index
(`data/indexes/sigir_medcpt_excl_embeddings.npy`), 100 self-queries, k=50, seed 0.
Local backends run offline at $0; the pgvector arm loads the vectors into an
isolated temp table on the configured Neon free tier and drops it afterward.
Reproduce: `python scripts/bench_vectorstore.py --source sigir --pgvector`.

## Latency and recall

| Backend | build | query p50 | query p95 | recall vs exact |
|---|---|---|---|---|
| numpy brute (FileIndex, exact) | — | 0.16 ms | 0.22 ms | 1.000 |
| sklearn brute (exact) | 0.002 s | 3.76 ms | 3.95 ms | 1.000 |
| pgvector ivfflat, lists=54, probes=1 | 88 s* | 24.8 ms | 49.8 ms | **0.376** |

\* build is network-bound (row-by-row insert to remote Neon), not compute; ignore
it as a compute figure.

At eval scale, exact numpy brute force is sub-millisecond and recall-perfect. This
is why the eval harness uses `FileIndex` (numpy) and needs no database — AD-6's
eval-side choice is confirmed. sklearn brute is exact too but ~20x slower for no
benefit here.

## The ivfflat recall gap (production-relevant finding)

pgvector's ivfflat with the **default probes=1** returns only ~0.38 of the exact
top-50. Because the eval recall numbers in the README were measured through the
exact `FileIndex`, production pgvector at the default setting would silently
underperform them. Probes trade recall back at almost no latency cost (latency here
is dominated by the ~23 ms Neon round trip):

| probes | recall vs exact | p50 | p95 |
|---|---|---|---|
| 1 | 0.357 | 22.8 ms | 36.5 ms |
| 5 | 0.745 | 23.2 ms | 36.6 ms |
| 10 | 0.877 | 23.4 ms | 34.6 ms |
| 25 | 0.978 | 24.1 ms | 31.2 ms |
| 54 (all lists) | 1.000 | 32.8 ms | 42.6 ms |

**Action taken:** `dense_search` now issues `SET LOCAL ivfflat.probes`
(`config.pgvector_probes`, default 20), so production dense retrieval recovers
near-exact recall instead of running blind at probes=1. Tune toward `sqrt(lists)`
or higher for the production corpus. End-to-end validation on a populated prod
table is quota/compute-paced (P1).

## Size ceiling — the real AD-6 driver

The decision axis is memory, not speed. Vectors-only footprint (768-dim float32),
against the Neon free tier's 512 MB:

| Corpus | n | vectors only |
|---|---|---|
| SIGIR eval | 2,991 | 9.2 MB |
| TREC 2021/2022 | 26,000 | 79.9 MB |
| scoped oncology prod | 50,000 | 153.6 MB |
| full ClinicalTrials.gov | 500,000 | 1,536 MB |

Vectors are only part of the row — trial metadata, the ivfflat index, and the BM25
side all share the 512 MB. The full registry (1.5 GB of vectors alone) cannot live
on the free tier, and 26k TREC is already tight once index + metadata are added.
This is exactly why eval does not use pgvector: the eval corpora are loaded into
numpy `FileIndex`, and pgvector is reserved for the scoped, production-sized
oncology corpus that fits.

## AD-6 verdict

**Confirmed, with one amendment.** pgvector for production + numpy FileIndex for
eval stands: the split is driven by the free-tier size ceiling, and exact numpy is
faster and recall-perfect at eval scale. Amendment: production pgvector must run
with tuned `ivfflat.probes` (now wired into `dense_search`); at the default it is
not recall-equivalent to the exact backend the eval numbers were measured on.
A managed alternative (Qdrant/Pinecone free tier) was not benchmarked live —
deferred (P1) — because the same size ceiling, not query engine, is the binding
constraint.
