# R2: a fine-tuned cross-encoder is the first reranker to beat fusion, and it misses its bar

Run 2026-10-01 on EC2 (c7i.8xlarge, CPU, bf16), cross-cohort, pre-registered in
`docs/weakpoints_fix_plan.md` §4 before any test data was scored. About $4.5 of
compute; no LLM calls. Raw: `r2_crossencoder.json`;
`data/cache/r2/scores_*.json` hold per-candidate scores.

## Setup (as pre-registered, plus one dated amendment)

- **Model:** `ncbi/MedCPT-Cross-Encoder`, fine-tuned for 1 epoch with a
  localized contrastive loss: 1 eligible positive and 7 negatives per group from
  the patient's own candidate pool, up to 3 of them "excluded" hard negatives.
  lr 2e-5, seed 0.
- **Input:** query = cached keywords; doc = title + inclusion + exclusion,
  512-token cap.
- **Candidates:** R1's fused top-500 per patient. Train on one TREC cohort, test
  on the other, both directions. Test topics are never seen in training.
- **Arms:** `ce` (cross-encoder score alone) and `ce_rrf_deep` (RRF of the
  cross-encoder rank and the deep-fusion rank).
- **Baseline:** deep fusion, the strongest known ordering and what `top_k>=100`
  now serves.
- **Amendment (before scoring):** the account has 0 GPU quota, so training ran
  on CPU in bf16 with attention dropout 0. Hidden dropout stayed at 0.1. This
  was for compute only.

## Result

Recall vs deep fusion, Wilcoxon + BH across the 8-cell family:

| arm | fold | @100 Δ | @200 Δ | better/worse @100 |
|---|---|---|---|---|
| ce | 2021→2022 | +0.040 [0.009, 0.070] | +0.046 | 31/11 |
| ce | 2022→2021 | **+0.053** [0.036, 0.070] | **+0.071** | 52/14 |
| ce_rrf_deep | 2021→2022 | +0.0495 [0.030, 0.070] | +0.043 | 38/5 |
| ce_rrf_deep | 2022→2021 | +0.0499 [0.039, 0.061] | +0.058 | 59/6 |

All 8 cells are significant (p_bh ≤ 0.008). Absolute recall@100: deep fusion
0.525 / 0.460, `ce_rrf_deep` 0.574 / 0.510.

## Against the pre-registered bar

**Adopt** required an arm to beat deep fusion by **≥ +0.05 recall@100 in both
directions** at BH p < 0.05.

- `ce` clears it on 2022→2021 (+0.053) and misses on 2021→2022 (+0.040).
- `ce_rrf_deep` misses both, by 0.0005 and 0.0001: +0.0495 and +0.0499.

**Neither arm is adopted.** The bar was written before the data, and the run
landing 0.0005 under it doesn't move it. Every arm is positive and significant
in both directions and below +0.05, which is exactly the condition the
pre-registration set for running **R3** (synthetic training pairs). R3 is now
READY.

## Why this one worked where every earlier reranker failed

| reranker | supervision | result vs fusion |
|---|---|---|
| ms-marco cross-encoder (R4, 2026-08-31) | none, web search | −33% recall@50 |
| MedCPT cross-encoder (R4) | none, PubMed search | −9.5% |
| LLM listwise screen (H2) | none | −0.01 / −0.04 recall@50 |
| LambdaMART over retrieval signals (R1) | TREC qrels | ≤ 0, worse on 5/8 cells |
| **MedCPT cross-encoder fine-tuned (R2)** | **TREC qrels, reads text** | **+0.04 to +0.05 @100, +0.04 to +0.07 @200** |

Supervision alone (R1) wasn't enough, and neither was reading the text alone
(R4, H2). The combination is what moves recall: a model that reads the trial
text and was trained on this task's own eligibility labels. That agrees with
the 2024–2026 systems that fine-tune rerankers (IELAB at TREC CT 2023,
TrialMatchAI, LLM-Match). It also fits the 2025 finding that rerankers
generalize poorly to novel queries (arXiv:2508.16757) without contradicting it:
here every test query is novel, and the gain held, because training and test
come from the same task.

`ce_rrf_deep` is the more consistent arm: 59/6 and 38/5 patients better/worse,
with tight CIs. `ce` alone has the larger best case and the weaker worst case.
That pattern argues for fusing the cross-encoder with retrieval rather than
replacing it.

## Serving cost (measured)

Scoring ran at about 60–70 pairs/s on 16 physical cores in bf16: roughly
**7–8 s per patient for a 500-candidate pool**, or about 1.5 s for 100. That
only fits the deep path, which already runs as a multi-minute job, never the
interactive top-5/25 search. Training cost about 80 min per direction on CPU.

## What next

- **R3** (pre-registered trigger met): add synthetic patient notes written for
  real CT.gov trials that appear in neither test cohort, retrain, and use the
  same bar. Kill if it adds less than +0.02 @100 over R2.
- If R3 clears the bar, wire the cross-encoder into the deep path only, behind
  a flag, and confirm the end-to-end effect with H1's harness at top-100.
- Either way, this is the first evidence that ordering is learnable here.
  `structural_recall_plan.md`'s conclusion that reordering doesn't respond was
  true of every unsupervised or text-blind lever, not of reordering itself.
