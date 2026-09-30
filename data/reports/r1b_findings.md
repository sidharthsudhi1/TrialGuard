# R1b: 200-deep keyword lists, on the served path

Done 2026-09-30. $0. Follows `r1_findings.md`, which found that fusing
200-deep per-keyword lists instead of 50 lifts TREC recall@100 by ~0.02 and
recall@200 by ~0.07.

## What changed

`fusion.list_pool()` sets the depth of every per-keyword list fed to RRF. The
default is 200, and `TG_LIST_POOL=50` reproduces every ranking committed before
this. Both `retrieval/pipeline.retrieve()` (served) and `FileIndex.search`
(eval) read it when no explicit pool is passed. Explicit `dense_pool` /
`bm25_pool` arguments still win, so scripts that pin a depth are unchanged.
This follows the `TG_KEYWORD_DECAY` pattern.

## Does the gain survive the served lexical backend?

Production dense search is the same exact matrix product as eval. Production
lexical search is Postgres FTS (`ts_rank_cd` over `websearch_to_tsquery`), not
rank-bm25, and TREC is not loaded in Postgres, so the full effect cannot be
replayed on the served backend. The split tells us how much depends on it
(`r1b_split_{cohort}.json`, Wilcoxon + BH against 50/50):

| cohort | depth | dense 200, lexical 50 | dense 50, lexical 200 | both 200 |
|---|---|---|---|---|
| TREC 2021 | @100 | **+0.021** (p_bh <0.001) | +0.011 (p_bh 0.013) | +0.018 |
| TREC 2021 | @200 | **+0.059** (p_bh <0.001) | +0.045 | +0.073 |
| TREC 2022 | @100 | **+0.016** (p_bh 0.027) | +0.014 (p_bh 0.026) | +0.022 |
| TREC 2022 | @200 | **+0.055** (p_bh <0.001) | +0.041 | +0.067 |

The dense half alone accounts for most of the gain at both depths, on both
cohorts. That half runs on the identical backend in production, so it carries
over as measured. The lexical increment on top is ~0.01–0.015 @200 on rank-bm25;
whether FTS delivers it is unmeasured. FTS applies AND semantics to a
multi-word keyword, so many keyword queries may match fewer than 50 trials, in
which case `LIMIT 200` changes nothing and costs nothing.

## Cost

- **Dense (in-process matrix, 26,037 × 768, local):** full matmul + argpartition
  + sort takes p50 3.3 ms at top-50 and 2.8 ms at top-200, which is noise. The
  matmul dominates, and it doesn't depend on k.
- **Lexical (Postgres FTS), measured 2026-09-30** (`r1b_latency_probe.json`):
  60 cached keywords, the served query under `EXPLAIN ANALYZE` against the
  production corpus in a read-only transaction. There is one warm pass, then
  `LIMIT` 50 and 200 run in random order, 3 reps each, median per keyword.

  | | p50 | p95 | mean rows returned | keywords reaching the limit |
  |---|---|---|---|---|
  | LIMIT 50 | 0.43 ms | 20.5 ms | 27.5 | 47% |
  | LIMIT 200 | 0.46 ms | 20.1 ms | 79.6 | 28% |

  There's no difference. FTS ranks the whole GIN match set before the limit
  applies, and 72% of keywords match fewer than 200 trials anyway.
  **The first draft of this probe reported LIMIT 50 at p95 1,853 ms.** That was
  an ordering artefact: 50 always ran first and paid the cold-buffer read for
  both. The warm pass and randomized order remove it. That cold first touch is
  real, though, and it's independent of the limit, which is worth knowing for
  E6's cold path.
- **Fusion:** 24 lists × 200 entries instead of × 50 in pure Python, well under
  1 ms `[inferred]`.

## Side effects

- **Eval reproducibility:** every committed ranking and cached end-to-end number
  used pool 50. Rerunning them needs `TG_LIST_POOL=50` (added to the V1 flag
  list). New eval runs at 200 will retrieve different trials, so their analyst
  pairs are partly uncached.
- **Demo presets:** the two preset notes may surface different trials. Any new
  (note, trial) pair pays one analyst call on first click, then is cached. It's a
  one-time cost of cents.
- **Recall projection:** at the served top-100 the retrieval gain is +0.02, so
  surfaced recall is expected to rise ~+0.01 (H1's flat ~0.6 conversion)
  `[inferred]`. Confirming it takes a fresh end-to-end run at pool 200: ~1–2k
  new pairs per TREC cohort, ~$0.5–1 each.

## Standing

- **Adopted in code**, default 200. AD-35.
- **Not yet deployed.** Both backends cost nothing extra at 200 (dense is
  noise, FTS is identical warm). What's left is a Fly deploy and E6's
  warm-path harness against it, to confirm the p95 stays inside the 1500 ms SLO.
