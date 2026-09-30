# R1: a learned reranker adds nothing, but deeper fusion does

Run 2026-09-30. $0: cached keywords, local MedCPT, no LLM calls. Raw:
`r1_ltr.json`, `r1_pool_sweep_{trec_2021,trec_2022,sigir}.json`,
`r1_pool_sweep_summary.json`.

## What was tested

LightGBM LambdaMART (fixed hyperparameters, set before any test fold was scored)
over 27 cheap features:

- per-keyword dense and lexical ranks and scores
- the note-level dense score
- demographic gate margins
- criteria counts
- keyword overlap with inclusion vs exclusion text

The ranker reorders a fused top-500 candidate pool built from 1000-deep
per-keyword lists. It is trained on one TREC cohort and tested on the other, in
both directions, with graded labels (eligible 2, excluded 1) and with
eligible-only labels. Test topics are never seen in training.

The baseline is `FileIndex.search` itself: the served eval order, with per-list
pool 50. It reproduces V2's retrieval ceiling exactly (0.3388 / 0.392 pooled
recall@100).

## Result 1: the learned ranker is rejected

Adopt bar (plan §4): ≥ +0.05 recall@100 in both directions, BH p < 0.05.

| fold | vs served @100 | vs served @200 | **vs deep fusion @100** | vs deep @200 |
|---|---|---|---|---|
| graded 2021→2022 | +0.026 (p_bh 0.007) | +0.068 | −0.000 (ns) | +0.004 (ns) |
| graded 2022→2021 | −0.005 (ns) | +0.055 | **−0.026** (p_bh 0.001) | **−0.018** (p_bh 0.01) |
| eligible 2021→2022 | +0.008 (ns) | +0.056 | **−0.018** (p_bh 0.04) | −0.008 (ns) |
| eligible 2022→2021 | +0.002 (ns) | +0.061 | **−0.019** (p_bh 0.01) | −0.012 (p_bh 0.04) |

Every gain the ranker shows over the served order, deep fusion already has on
its own. Against deep fusion the ranker is never better, and on 5 of 8 cells it
is significantly worse. The features carry no ordering signal beyond what
fusing deeper lists already encodes. This is the same conclusion R4 and H2
reached for zero-shot rerankers, and it now covers a supervised one trained on
the task's own labels.

In every fold the top feature by gain was the deep fusion rank itself (714–965,
about 3–5x the next feature). The ranker largely learned to copy deep fusion.

## Result 2: per-list pool depth is the lever

Same keywords, index and `1/i` weights. Only the per-keyword list depth fed to
RRF changes. Per-patient mean recall, Wilcoxon + BH within cohort:

| cohort | depth | pool 50 (served) | 100 | **200** | 500 | 1000 |
|---|---|---|---|---|---|---|
| TREC 2021 | @100 | 0.439 | 0.457 | **0.458** | 0.457 | 0.460 |
| TREC 2021 | @200 | 0.561 | 0.617 | **0.634** | 0.635 | 0.634 |
| TREC 2021 | @500 | 0.692 | 0.755 | **0.806** | 0.829 | 0.832 |
| TREC 2022 | @100 | 0.499 | 0.518 | **0.520** | 0.527 | 0.525 |
| TREC 2022 | @200 | 0.607 | 0.657 | **0.674** | 0.670 | 0.671 |
| TREC 2022 | @500 | 0.698 | 0.754 | **0.786** | 0.803 | 0.807 |
| SIGIR | @100 | 0.768 | 0.770 | 0.745 | 0.755 | 0.747 |
| SIGIR | @200 | 0.844 | 0.862 | 0.856 | 0.835 | 0.844 |

**Pool 200 vs 50:**
- **TREC 2021:** +0.018 @100 (p_bh 0.0007), +0.073 @200, +0.113 @500.
- **TREC 2022:** +0.022 @100 (p_bh 0.02), +0.067 @200, +0.088 @500.
- **SIGIR:** no cell is significant. Its corpus is 2,991 trials and gold already sits near the head, so there is nothing deep to recover.

**Mechanism.** With 50-deep lists, a trial scores only if it reaches the top 50
of some keyword's list. Trials that rank moderately for many keywords get zero,
yet that consensus is exactly what an eligible trial looks like. Deeper lists
let it accumulate.

The structural plan (§5) had recorded the opposite: 1000-deep fusion *diluted*
recall@200 (0.5243 vs 0.5454). That was measured before keyword decay (AD
keyword_decay, 2026-09-04). Under uniform weights, deep lists from minor keywords
add noise. Under `1/i` they add mostly consensus from the important ones. Two
levers each measured alone have now interacted, which is the pattern R3 and the
decay study already showed once.

## What this means for the system

- **At the served top-100 the gain is small:** +0.02 recall@100 (4–5% relative).
  H1 showed surfaced recall scales with the pool's gold content at a flat ~0.6
  conversion, so this projects to roughly +0.01 surfaced recall at top-100
  `[inferred]`.
- **At depth it's large:** +0.07 @200 and +0.09–0.11 @500. Those are the pools
  any future deep-assessment or cascade (R4) would consume. Pool depth should
  be set before R4 runs.
- **Cost:** eval dense search is exact over an in-memory matrix, so depth is
  free there. In production, per-keyword Postgres FTS returns `LIMIT 200`
  instead of 50, and the matrix path takes a larger `argpartition`. Neither has
  been measured on the served path yet.

## Standing

- R1 (learned ranker): **DONE-REJECTED.** `eval/ltr.py` stays as the harness.
- New item R1b: **per-list pool 200 on the served path**, behind a flag whose
  `=50` value reproduces every committed ranking (the `TG_KEYWORD_DECAY`
  pattern). Needs a served-path latency measurement (E6 harness), an end-to-end
  check that surfaced recall moves as projected, and an ADR. This changes
  production retrieval, so it waits on a go-ahead.
