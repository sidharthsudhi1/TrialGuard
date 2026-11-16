# AD-40 — A small model does not screen the deep tail

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

Llama-3.1-8B is not used to screen ranks past 100 before the 70B analyst sees them. Run with the full analyst prompt and grounding on deep-fusion ranks 101 to 136, it tiered 287 of 761 gold-eligible trials `excluded` (0.377; TREC 2021 0.393, TREC 2022 0.356, 125 patients, $0.674). The bar, written before the run, was 0.05. The 70B phase did not run, so there is no surfaced-recall number for the cascade. The served path stays a 70B assessment of the retrieval prefix. [`r4_cascade_findings.md`](../../data/reports/r4_cascade_findings.md).

## Alternatives considered

- **Promote what the 8B does not exclude.** That is the cascade. It would discard more than a third of the gold in the slice before the 70B reads it.
- **A one-line glance instead of the full prompt.** Already negative on both TREC cohorts (`h2_findings.md`).
