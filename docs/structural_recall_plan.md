# The structural gap: where the missing recall actually is

Written 2026-09-01. Opens the question `pipeline_review.md` §05a closed with:
the cheap levers are exhausted, and what would lift TREC end-to-end recall
toward its ceiling has not been asked. This document asks it, starting from a
new measurement rather than a hypothesis.

Markers as in the pipeline review: **[measured]** is a number from this
repository; **[inferred]** is a hypothesis the plan exists to test.

## 0 — The diagnostic nobody had run  `[measured]`

Every prior retrieval number stopped at k=200. If the gold trials the system
misses are *unrepresentable* — ranked past 5,000 or absent from the fused
candidate space entirely — then the fix is a new embedder or query form. If
they are merely *misordered* — present at moderate depth — the fix is a
second-stage reordering, and the embedder is fine.

Deep-pool run, TREC 2021, n=75, keyword config, `dense_pool = bm25_pool =
1000` per keyword, cached MedCPT index, $0
(`data/reports/gold_rank_depth_trec2021.json`):

| fused depth | cumulative gold recall |
|---|---|
| 10 | 0.0463 |
| 100 | 0.3401 |
| 200 | 0.5243 |
| **500** | **0.7658** |
| **1000** | **0.8711** |
| 2000 | 0.9246 |
| 5000 | 0.9565 |
| unranked | **2.44%** |

Median gold rank **176**, p90 **1,084**. Per patient, the *best* gold trial
sits at median rank 2 (consistent with MRR 0.63) — the head is already right;
it is the body of each patient's ~74 eligible trials that smears across ranks
100–1,000.

**Reading.** The current representation finds 87% of gold within the top 4%
of a 26k corpus. This is not a representation failure. The structural gap is
an *ordering* problem over a ~500–1,000 candidate pool — and generating that
pool is nearly free (dense search is in-process at 3.8–6.7 ms; deep BM25 adds
little). The three levers §05a named now rank very differently:

- **a different retrieval model** — demoted. It would be solving invisibility,
  and only 2.4% of gold is invisible.
- **a different query representation** — demoted for the same reason; keywords
  already put the candidates in the pool.
- **a larger pool passed onward** — promoted. The whole question becomes: what
  is the cheapest precise reorderer of 500 candidates?

## 1 — What is already known about reordering  `[measured]`

Four reorderers have been tried on the ≤200 pool and all failed:

| lever | result |
|---|---|
| RRF constant sweep (R5) | cohorts disagree in sign |
| pool-depth sweep (R5) | no significant gain |
| keyword-rank weighting (R3) | +3.4% TREC, Fisher p=0.0998 |
| cross-encoder rerank, ms-marco then MedCPT head (R4) | loses outright; MedCPT −9.5% at best, 21 s/patient |

The pattern: reweighting *existing* rank signals does nothing, and the two
rerankers tested were (i) wrong-domain and (ii) a PubMed-search head applied
to eligibility prose — neither has ever read an eligibility criterion against
a patient. Nothing tried so far brings *new evidence* to the ordering. The
agent, which does read criteria, is measurably precise: it has never emitted
`eligible` on a gold-excluded trial, and surfaced precision on TREC is 0.69
against a 0.43 base rate (1.60x lift, `e1b_findings.md`).

## 2 — Ranked hypotheses

### H1 — The agent is the reranker: widen the assessed pool  `[MEASURED 2026-09-01 — CONFIRMED]`

> Run same-day: TREC 2021, n=20, k ∈ {10, 25, 50, 100}, $0.99, cache
> coverage 1.0 at every point. Surfaced recall 0.0276 → **0.1770 (6.4x)** at
> k=100. Conversion flat at ~0.6 and lift over pool base rate flat at 1.6x
> across the full 10x widening — the kill criterion (lift decaying toward
> 1.0) did not move. The projection below was right to first order
> (0.298 × 0.594 = 0.177 measured). The structural gap was the k=10
> hand-off, not the retrieval representation. Remaining question is serving
> economics (~$0.05, ~10 min/patient at k=100), which is H2's job.
> `data/reports/h1_pool_curve_trec2021.md`.

Stop treating top-10 as the hand-off. Retrieval delivers 0.34 recall at 100
and 0.77 at 500; the tiered contract (A5) already separates what the agent
assesses into eligible / needs_review / excluded with ranked review depth.
Composing measured numbers: surfaced conversion on TREC is ~0.57 of the
retrieval ceiling, so assessing top-100 projects surfaced end-to-end recall
of roughly 0.34 × 0.57 ≈ **0.19** against today's 0.0276 — a ~7x, using only
components that already exist. `[inferred]` — the conversion rate was measured
at top-10 pool richness and may fall with depth; that is exactly what the
experiment measures.

- **Test:** e2e harness, TREC 2021, n=20, top-k ∈ {10, 25, 50, 100}. Same
  harness that ran E1b. ~2,000 assessments ≈ **$1.50–2.50** DeepInfra,
  EC2 for the corpus (existing recipe).
- **Decide on:** surfaced recall and surfaced precision *as a curve in k*. If
  precision holds within ~0.1 of the top-10 figure while recall scales, the
  structural gap is closed by spending ~$0.07/patient of inference, and the
  "retrieval representation" question is answered: *it was never the
  representation.*
- **Kill:** surfaced precision collapses toward base rate as k grows — the
  agent's discrimination was a property of pool richness, not of the trials.

### H2 — LLM listwise rerank: 500 → 50 before the agent  `[inferred]`

If H1's cost or latency profile is unacceptable for serving (100 assessments
per patient is ~$0.07 and minutes of wall clock even parallelized), insert a
coarse LLM pass: batches of ~50 candidates (title + condition + one-line
eligibility gist), listwise "rank the plausible ones", fuse. This brings the
missing new evidence (a model actually reading the trial against the note) at
~1/100th the token cost of full assessment — ~$0.005/patient, seconds.
Unlike R4's cross-encoders this is the same family of judgment the agent
already makes, just shallower. Not faithfulness-relevant: ordering only,
verdicts stay grounded downstream.

- **Test:** retrieval-only harness vs gold, both cohorts. Success = recall@50
  from the reranked 500-pool beats today's recall@50 0.286 by enough to shift
  the H1 curve left (target ≥ 0.45, i.e. capture most of the 0.77@500).
- **Kill:** ≤ +0.05 recall@50, or per-patient latency > ~5 s at serving
  concurrency, or cost > $0.01/patient.

### H3 — Embedder upgrade A/B  `[inferred, demoted to hygiene]`

A modern instruction-tuned embedder (long-context, so truncation dies as a
side effect) could compress the 100–1,000 smear toward the head. Evidence is
against a large move: model-size swaps did nothing twice (AD-5), chunking did
nothing (R1), and only domain match (MedCPT) ever paid. Worth one cached-index
A/B on the FileIndex harness because it is $0 and local — not worth blocking
on. Run only after H1/H2 have an answer.

- **Kill:** fails to beat recall@100 0.4227 / recall@500 0.766 on TREC.

## 3 — Order

1. **H1** — one EC2 session, ~$2. It is the only experiment that directly
   measures the number the thesis cares about (surfaced end-to-end recall),
   and its k-curve is the denominator every other lever multiplies into.
2. **H2** — only if H1 says the agent's precision survives depth but the
   serving cost/latency at the required k is unacceptable.
3. **H3** — background hygiene, $0, never blocking.

## 4 — What this plan does not do

- It does not touch the agent's conversion rate (~0.57 of ceiling on TREC).
  That is the other factor of the product, is bounded above by honest
  abstention on notes that genuinely lack facts, and per `e1b_findings.md`
  its remaining loss shape (false exclusion via inference overreach) is a
  separate, smaller question.
- It does not re-run any fusion-stage tuning. Four levers said no.
- It does not revisit chunking, probes, or general-domain rerankers
  (`pipeline_review.md` §07).

## 5 — Caveats

- Depth curve is TREC 2021 only. SIGIR's median 6 gold/patient makes a depth
  curve mostly ceiling; TREC is the cohort of record. TREC 2022 replication
  is cheap if H1 moves.
- Deep-pool fusion at 1000/list scored recall@200 at 0.5243 vs 0.5454 under
  default 50/50 pools — RRF dilutes slightly at depth. Immaterial at the
  depths H1/H2 consume, but H2 should fuse from the deep pool, not re-run
  default pools.
- qrels label only the judged pool; the 2.4% unranked figure inherits that.
