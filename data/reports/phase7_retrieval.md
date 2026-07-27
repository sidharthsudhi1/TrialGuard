# Phase 7 — Production retrieval validation

Production corpus loaded into pgvector (`ctgov_live`): **25,965 recruiting oncology
trials** (RECRUITING 18,885 / NOT_YET_RECRUITING 6,331 / ENROLLING_BY_INVITATION
749), all with MedCPT embeddings and a generated `doc_tsv` FTS column over title +
inclusion + exclusion.

Two things had to be shown before trusting the production path: (1) the BM25 →
Postgres FTS migration did not regress ranking, and (2) ivfflat `probes` is tuned
for the real corpus size.

## WS-4a — FTS ranking vs gold (does ts_rank_cd regress vs rank-bm25?)

SIGIR loaded into pgvector under `source='sigir'` (the same 2,991 trials the
FileIndex eval uses, so the corpus is held constant), recall run through the
production `retrieve()` path (pgvector dense + FTS lexical + RRF, keyword queries).
n=53, gold coverage 0.9635.

| metric | FileIndex (rank-bm25) | Production (FTS + pgvector) |
|---|---|---|
| recall@10 | 0.1345 | **0.1775** |
| recall@100 (adj) | 0.72 | 0.7165 |
| recall@200 (adj) | 0.81 | 0.798 |
| MRR | — | 0.327 |

Recall@100/200 are within noise of the rank-bm25 baseline; recall@10 improves by
+0.043. This confirms the standing hypothesis that under short keyword queries the
lexical arm's contribution flips positive — FTS captures it. **No fallback
(ParadeDB / ts_rank weight tuning) needed.**

Note: the `source` filter selects ~10% of the table, so the planner seq-scans the
SIGIR subset and sorts exactly — ivfflat is bypassed for eval-cohort scopes, so the
dense arm here is **exact**, and these numbers are not confounded by ANN error.

## WS-4b — ivfflat probes bench (ctgov_live)

For the 90%-of-table `ctgov_live` scope the planner does use ivfflat, so `probes`
governs recall. No gold for live trials, so recall is measured **vs exact**
(probes=161 = all lists) over 30 real query vectors, top-100, warm latency.

| probes | recall vs exact | ms p50 (warm) |
|---|---|---|
| 1 | 0.11 | 70 |
| 5 | 0.31 | 58 |
| 10 | 0.45 | 57 |
| 20 (old default) | 0.62 | 58 |
| 40 | ~1.00 | 172 |
| 80 | 0.97 | 85 |

The old default `probes=20` recovered only ~62% of the exact top-100. The knee is
~40 (near-full recall, still sub-200ms). **Default raised 20 → 40** (`config.py`).
This affects only `ctgov_live`; eval-cohort scopes seq-scan exactly regardless.

## Status

WS-1..WS-4 complete. Production hybrid retrieval validated end-to-end on the live
26k corpus. Remaining: WS-5 (refresh / status expiry).
