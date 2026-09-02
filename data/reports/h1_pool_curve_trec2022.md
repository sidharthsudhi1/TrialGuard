# H1 replication — TREC 2022

Run 2026-09-02. Same design as `h1_pool_curve_trec2021.md`: TREC 2022, 20
patients, prompt v4, DeepInfra, current verifier. One prewarm of all top-100
pairs (2,000 assessments, all fresh, $0.85, zero errors at 10 workers / 300s
timeout), then four k-points scored from the analyst cache at coverage 1.0, so
every row slices the same responses at a different depth. Raw:
`h1_pool_curve_trec2022.json`.

**Index caveat:** TREC 2022 has only the `medcpt_noexcl` index cached, so this
cohort retrieves against title+inclusion while 2021 retrieves against
title+inclusion+exclusion. That is the same variant every prior committed 2022
number uses, so the cohorts are each internally consistent and comparable to
their own history, but the absolute ceilings are not comparable across the two
cohorts. The claim under test — does conversion survive depth — is a
within-cohort trend and is unaffected.

## Result

| k | retrieval ceiling | surfaced recall | conversion | surfaced precision | pool base rate | lift |
|---|---|---|---|---|---|---|
| 10 | 0.0674 | 0.0430 | 0.638 | 0.6979 | 0.5469 | 1.28x |
| 25 | 0.1386 | 0.0931 | 0.672 | 0.6561 | 0.4789 | 1.37x |
| 50 | 0.2413 | 0.1675 | 0.694 | 0.5918 | 0.4372 | 1.35x |
| 100 | 0.3780 | **0.2580** | 0.683 | 0.5332 | 0.3688 | **1.45x** |

Depth diagnostic on the same cohort (`gold_rank_depth_trec2022.json`, n=50):
recall 0.2371@50, 0.3988@100, 0.5918@200, **0.7875@500**, 0.8634@1000, with
5.4% of gold unranked and median gold rank 133 — the same shape as 2021
(0.7658@500, 2.4% unranked, median 176).

## Replication verdict: holds, and slightly stronger

The 2021 finding was that conversion and lift stay flat as the assessed pool is
widened 10x, so the agent behaves as a constant-lift filter rather than one
borrowing its judgment from pool richness. On 2022 both quantities not only
hold, they drift *up*: conversion 0.638 → 0.683 and lift 1.28x → 1.45x while
the pool dilutes from 54.7% to 36.9% gold. Neither cohort shows the decay the
kill criterion was written for.

| | TREC 2021 | TREC 2022 |
|---|---|---|
| surfaced recall, k=10 → k=100 | 0.0276 → 0.1770 | 0.0430 → 0.2580 |
| multiplier | **6.4x** | **6.0x** |
| conversion across depths | 0.575 – 0.623 (flat) | 0.638 – 0.694 (flat, rising) |
| lift across depths | 1.60x – 1.67x (flat) | 1.28x – 1.45x (flat, rising) |
| criterion unverifiable rate | 0.0395 → 0.0471 | 0.0132 → 0.0207 |

Two cohorts, independently prewarmed, agree on the direction and roughly on the
magnitude. This is the replication `pipeline_review.md` §05a demanded before
trusting any single-cohort conclusion — and unlike A1 (which died on TREC) and
R5/R3 (which inverted or shrank), H1 survives it.

Absolute levels differ as expected from cohort and index variant: 2022 starts
from a richer pool (54.7% vs 42.9% gold at k=10), which mechanically compresses
lift, and its unverifiable rate is roughly half 2021's.

## Reading

- The structural gap is confirmed closed on both cohorts of record: it was the
  k=10 hand-off, not the retrieval representation. Deep candidates exist
  (~0.78–0.79 recall@500 on both), and the agent converts them at a
  depth-independent rate.
- What remains is serving economics. At k=100 an assessment costs ~$0.05 and
  ~10 min/patient at 10-way concurrency. H2 (cheap LLM listwise rerank 500 → 50)
  is now the only open question in this plan, and it is a cost question.
- Still extrapolation, not measurement: k=500 would put surfaced recall near
  0.5 on both cohorts if conversion held, but 100 → 500 is outside the data on
  both.
- n=20 patients per cohort. Directions are solid across two independent runs;
  second decimals are not.
