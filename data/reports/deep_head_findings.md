# Deep head: what a progressive deep job shows at the 25-trial banner

A progressive deep job (plan item R6, PR #46; not the pipeline review's R6 in
`r6_closure.md`) assesses the top 100 in rank order and signals once the top 25
are done. How useful is that head on its own, and is the tail worth waiting for?

Run 2026-10-08 on EC2, cached-only, $0 LLM, completion and cache coverage 1.0
in every cell. Top-25 is an exact prefix of the top-100 run it is compared with:
TREC at list pool 200 (the served deep path) against `r2e2e_*_top100.json`,
SIGIR at list pool 50 against `v2_e2e_sigir_top100.json`. Raw:
`deep_head_*_top25.json`, `deep_head_compare.json`; script
`scripts/deep_head_compare.py`.

## Result

| cohort | ordering | n | head surfaced recall [95% CI] | full (top-100) | head share of hits | patients with a hit in head | head lift | full lift |
|---|---|---|---|---|---|---|---|---|
| TREC 2021 | deep | 75 | 0.074 [0.063, 0.087] | 0.218 | 34% | 74 / 75 | 1.51x | 1.55x |
| TREC 2021 | ce | 75 | 0.096 [0.082, 0.112] | 0.251 | 38% | 75 / 75 | 1.33x | 1.45x |
| TREC 2022 | deep | 50 | 0.100 [0.080, 0.125] | 0.285 | 35% | 49 / 50 | 1.21x | 1.29x |
| TREC 2022 | ce | 50 | 0.120 [0.098, 0.148] | 0.318 | 38% | 49 / 50 | 1.15x | 1.25x |
| SIGIR | deep | 52 | 0.291 [0.245, 0.335] | 0.502 | 58% | 41 / 45 | 1.48x | 1.47x |

"Patients with a hit" counts, among patients whose full run surfaces at least
one eligible trial, those whose head already does.

Tail gain, head -> full, paired (Wilcoxon + BH over the 5 cells): +0.166 to
+0.203 surfaced recall on TREC, +0.179 on SIGIR, all p_bh < 0.001. No patient
is worse with the tail (it only adds trials); 46-74 of each TREC cohort gain.

## Reading

- **The banner is useful.** At the head, 74/75, 49/50 and 41/45 patients
  already see at least one eligible trial. The progressive design's promise,
  something useful in the usual wait, holds.
- **The tail is most of the answer.** The head holds about a third of the
  top-100 surfaced hits on TREC (34-38%) and about 58% on SIGIR. Stopping at 25
  would give up roughly two thirds of what deep mode finds on TREC. Keep
  assessing to 100.
- **Head lift is about the same as full lift** (1.51x vs 1.55x, 1.21x vs 1.29x,
  1.48x vs 1.47x on the deep ordering), so the head is not a worse-quality
  sample; it is just smaller.
- **The cross-encoder helps the head most in relative terms.** Head surfaced
  recall +0.022 / +0.020 (+30% / +20%) over deep ordering, and its head share
  rises to 38%. If R6 ships, the cross-encoder (`retrieval_ce_model`) is what
  makes the banner richer; that is the case for running R2 `train-final` at
  deploy time.
- The ce head's lower lift is the base-rate effect seen in `r2e2e_findings.md`:
  its top 25 are richer in gold (base rate 0.53 vs 0.44 on TREC 2021).

## Decision

No change to R6 as built: head of 25, assess to 100. Deploying the
cross-encoder alongside it would lift the head by about a fifth to a third.
