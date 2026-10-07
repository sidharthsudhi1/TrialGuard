# R2 end to end: the cross-encoder's recall gain reaches the user

Does the fine-tuned cross-encoder's +0.05 recall@100 (`r2_findings.md`) survive
the agent and the tiered roll-up? Yes. About 60-75% of the retrieval gain shows
up as surfaced recall, significant on both TREC cohorts.

Runs 2026-10-04 to 2026-10-07 on EC2 (m7i.large), DeepInfra
Llama-3.3-70B-Instruct-Turbo, prompt v4, served config, top-100, full cohorts,
completion 1.0 and 0 errors in every cell. Raw: `r2e2e_{deep,ce}_{cohort}_top100.json`,
`r2_e2e_compare.json`.

## Setup

- **Arms:** `deep` = deep fusion (what `top_k>=100` serves, list pool 200).
  `ce` = `ce_rrf_deep` from R2: RRF of the cross-encoder rank and the deep-fusion
  rank, cross-cohort scores (trained on the other TREC cohort, never on the test
  topics). Read via `end_to_end --order-file`.
- **Pairing:** same patients, same agent, same cache. Wilcoxon + BH over the
  4-test family (2 cohorts x {retrieval recall, surfaced recall}), patient
  bootstrap CIs.

## Result

| cohort | n | arm | retrieval recall | surfaced recall [95% CI] | surfaced precision | base rate | lift [95% CI] | unverifiable |
|---|---|---|---|---|---|---|---|---|
| TREC 2021 | 75 | deep | 0.357 | 0.218 [0.192, 0.247] | 0.600 | 0.388 | 1.55x [1.45, 1.65] | 3.28% |
| TREC 2021 | 75 | ce | 0.406 | 0.251 [0.225, 0.281] | 0.621 | 0.427 | 1.45x [1.38, 1.54] | 3.49% |
| TREC 2022 | 50 | deep | 0.412 | 0.285 [0.239, 0.338] | 0.534 | 0.413 | 1.29x [1.22, 1.38] | 2.05% |
| TREC 2022 | 50 | ce | 0.452 | 0.318 [0.269, 0.373] | 0.552 | 0.443 | 1.25x [1.18, 1.32] | 1.98% |

Paired deltas, ce minus deep:

| cohort | metric | Δ [95% CI] | better / worse / tie | p_bh |
|---|---|---|---|---|
| TREC 2021 | retrieval recall | +0.052 [0.041, 0.063] | 59 / 6 / 10 | <0.001 |
| TREC 2021 | surfaced recall | **+0.032** [0.025, 0.040] | 56 / 5 / 14 | <0.001 |
| TREC 2022 | retrieval recall | +0.054 [0.034, 0.073] | 37 / 4 / 9 | <0.001 |
| TREC 2022 | surfaced recall | **+0.040** [0.024, 0.054] | 35 / 5 / 10 | <0.001 |

## Reading

- **The gain carries through.** Surfaced recall rises +0.032 / +0.040, i.e.
  62% / 73% of the retrieval gain. Relative to deep fusion that is +15% / +14%.
  Few patients get worse (5 of 75, 5 of 50).
- **Lift falls, precision does not.** Surfaced precision rises (0.600 -> 0.621,
  0.534 -> 0.552). Lift drops (1.55x -> 1.45x, 1.29x -> 1.25x) only because
  the cross-encoder's pool is richer in gold (base rate +0.04 / +0.03), so the
  denominator moved. The agent separates the pool as well as before; the pool
  is better.
- **Faithfulness unchanged.** Criterion unverifiable 3.28% -> 3.49% and
  2.05% -> 1.98%. Grounded rate 0.593 / 0.591 and 0.601 / 0.610.
- **'Eligible' tier precision slightly lower** (0.531 -> 0.520, 0.588 -> 0.571).
  Not tested; small next to the recall gain.

## Cost and operations

- LLM ~$2.5 across three launches; EC2 ~$3.
- The 2026-10-01 run and part of the 10-05 run failed on DeepInfra
  `402 You need positive balance` (account empty). The harness now records
  `error_types` (02eacb2), which is how this was found.
- About 11h of work was lost on 10-04 when the session-bound poller died and the
  box hit its watchdog. Pollers now run detached from `~/tg_poll/`.

## Decision for the user

Wiring the cross-encoder into the deep (`top_k>=100`) path costs ~7-8 s/patient
on CPU for 500 candidates (R2) and buys +0.03-0.04 surfaced recall at no
faithfulness cost. R2 missed its retrieval bar by 0.0005; end to end the gain is
clear on both cohorts. Recommendation: adopt on the deep path, behind a flag,
with the headline lift quoted against the new (higher) base rate.
