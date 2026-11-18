# R2S: a cross-encoder fast enough to serve (registration)

Written 2026-11-18, before any variant below was timed or scored.

## Why

R2's fine-tuned cross-encoder raises surfaced recall by +0.032 / +0.040 at top-100
(`r2e2e_findings.md`). Its "7-8 s per patient" cost was measured on a 32-vCPU
c7i.8xlarge in bf16. On the serving machine (Fly performance-2x, 2 vCPU, 4 GB,
fp32) one deep search measured 438-476 s for 500 candidates at 512 tokens, and
batch 128 peaked at 3.28 GB in a process holding MedCPT. As measured, R2 cannot
be served. This item asks whether a cheaper variant keeps the gain.

Compute: Fly probe machines for speed (the serving hardware), Modal for training
and scoring. No EC2.

## Variants

All keep R2's fusion: RRF (k=60) of the cross-encoder rank and the deep-fusion
rank. Serving batch is 16 in every variant (memory, not quality: batch size does
not change scores).

| ID | Weights | Max tokens | Candidates reranked |
|---|---|---|---|
| F512 | fp32 | 512 | 500 (R2 as measured; reference) |
| Q512 | int8 dynamic (Linear layers) | 512 | 500 |
| Q256 | int8 dynamic | 256 | 500 |
| Q256-300 | int8 dynamic | 256 | 300 |

## Stage A: speed (Fly performance-2x, 4 GB, the serving image)

Docs: 500 real `ctgov_live` trials (deterministic sample, `ORDER BY md5(nct_id || 'r2s')`),
text built by `ce_rerank.doc_text`. Queries: 3 fixed keyword lists. F512 runs 1
query (reference only). One process per variant, MedCPT loaded first as in serving.

- **Pass:** median cross-encoder wall time per search **<= 30 s**, and probe-process
  peak RSS **<= 2.2 GB**.
- Variants that pass go to Stage B in order of least change: Q512, Q256, Q256-300.
- **If none passes, stop.** R2 stays off; the user chooses between option 3
  (rerank the tail after the head is shown) and closing.

## Stage B: retrieval recall (Modal)

Retrain both cross-cohort folds with the Modal L4 recipe (`scripts/modal_train_ce.py`,
fp16, already checked against CPU R2 within 0.005) and save the fold models. Score
the Stage-A-passing variant on each held-out cohort.

- **Pass, both directions (2021->2022, 2022->2021):** variant minus deep fusion
  recall@100 **>= +0.04** with BH-adjusted p < 0.05, **and** variant minus the fp32
  GPU fold **>= -0.01** (point estimate).
- First variant in the Stage A order that passes goes to Stage C. None passes: stop.

## Stage C: end to end (Modal)

Served config, top-100, full TREC 2021 (n=75) and 2022 (n=50), same harness as
`r2e2e_findings.md` (`end_to_end --order-file`), variant vs deep.

- **Adopt if:** surfaced recall delta > 0 with BH p < 0.05 on both cohorts, and
  criterion unverifiable rate rises by no more than 0.5 points on either.
- Adoption = flag on (`RETRIEVAL_CE_MODEL`) plus a deploy, which waits for the user.
