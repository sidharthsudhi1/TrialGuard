# H1 — The agent is the reranker: surfaced recall vs assessed pool size

Run 2026-09-01. TREC 2021, 20 patients, prompt v4, DeepInfra, current verifier
(absence grounding). One prewarm of all top-100 pairs (1,998 assessments,
1,773 fresh, $0.99 including retries and one abandoned worker config), then
all four k-points scored from the analyst cache at coverage 1.0, so every row
below describes the same responses sliced at different depths. Retrieval is
the served keyword-RRF shape; top-k at each depth is a prefix of one fused
ranking. Raw: `h1_pool_curve_trec2021.json`.

Motivated by `docs/structural_recall_plan.md` §2/H1, which followed from the
depth diagnostic (`gold_rank_depth_trec2021.json`): gold is present at depth
(0.77 recall@500) but misordered, so the question was whether the agent's
discrimination survives being handed a deeper, poorer pool.

## Result

| k | retrieval ceiling | surfaced recall | conversion | surfaced precision | pool base rate | lift |
|---|---|---|---|---|---|---|
| 10 | 0.0480 | 0.0276 | 0.575 | 0.6885 | 0.4294 | 1.60x |
| 25 | 0.1066 | 0.0664 | 0.623 | 0.6645 | 0.3990 | 1.67x |
| 50 | 0.1836 | 0.1099 | 0.599 | 0.5860 | 0.3661 | 1.60x |
| 100 | 0.2980 | **0.1770** | 0.594 | 0.5546 | 0.3447 | **1.61x** |

Criterion unverifiable rate stays in band (0.0395 → 0.0471); trials surfaced
per patient grows 3.5 → 36.0, ranked within the needs_review tier by
`n_unknown` as A5 already specifies.

**The two quantities that could have killed this are both flat.**
Conversion (surfaced recall / ceiling) holds at ~0.6 from k=10 to k=100, and
lift over the pool's own base rate holds at 1.6x while the pool is diluted
from 43% to 34% gold. Precision falls 0.69 → 0.55 *exactly in proportion to
the dilution* — the agent behaves as a constant-lift filter, not as a filter
whose judgment was borrowed from pool richness. The kill criterion (lift
decaying toward 1.0) did not begin to move at 10x widening.

**Surfaced end-to-end recall moves 0.0276 → 0.1770, a 6.4x, with no change
to retrieval, prompt, or verifier.** The plan's projection (0.34 × 0.57 ≈
0.19 at k=100) was right to first order: this cohort slice's ceiling is
0.298 rather than the n=75 0.34, and 0.298 × 0.594 = 0.177 is what was
measured.

## Reading

- The structural gap of `pipeline_review.md` §05a is answered: it was never
  the retrieval representation. Deep candidates exist (0.77@500), the agent
  converts them at a depth-independent rate, and the binding constraint was
  the arbitrary k=10 hand-off.
- Extrapolation, not measurement: at k=500 the ceiling is ~0.77 and constant
  conversion would put surfaced recall near 0.45. Whether lift holds through
  another 5x dilution (base rate keeps falling with depth) is exactly the
  untested part; the trend through four points is flat but 100 → 500 is
  outside the data.
- Serving cost is the real limiter, not quality: ~$0.05 and ~10 min/patient
  at k=100 with 10-way concurrency. That is what H2 (cheap LLM listwise
  rerank 500 → 50) exists to buy down — it is now a serving-economics
  question, not a recall question.
- n=20 patients, one cohort. Directions solid, second decimals not; TREC 2022
  replication is cheap (its pairs are uncached, ~$1).

## Caveats

- `end_to_end_recall` (eligible tier only) also scales (0.0020 → 0.0217) but
  stays dominated by the completeness requirement; the surfaced tier is the
  contract A5 ships and the number quoted here.
- Base rates are over labelled rows only, consistent with how surfaced
  precision is scored (unjudged pool gaps are not false positives).
- The prewarm ran the graph with max_retries=2 as served; retry outcomes are
  baked into the cached responses the four slices share.
