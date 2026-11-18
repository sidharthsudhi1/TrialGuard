# AD-44 — The R2 cross-encoder is not served on the current machine

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

The fine-tuned cross-encoder from R2 stays off in production (`RETRIEVAL_CE_MODEL` empty). Its end-to-end gain is real: +0.032 / +0.040 surfaced recall at top-100 on TREC 2021 / 2022 with faithfulness flat ([`r2e2e_findings.md`](../../data/reports/r2e2e_findings.md)). Its serving cost was not. The "7-8 s per patient" in R2 was measured on a 16-core bf16 EC2 box. On the Fly performance-2x API machine, 500 candidates at 512 tokens take 438-476 s, and batch 128 peaks at 3.3 GB next to MedCPT on a 4 GB machine. The cheapest registered variant (int8, 256 tokens, 300 candidates) still took a median of 95 s against a 30 s bar written before the run ([`r2s_findings.md`](../../data/reports/r2s_findings.md)).

## Alternatives considered

- **Serve it anyway.** It adds 1.5-8 minutes before the first deep result and saturates both vCPUs.
- **Retest recall on cheaper variants.** Pointless once the fastest one fails the speed bar; Stage B and C were not run.
- **Rerank the tail after the head is shown, use a bigger or GPU machine, or distil a smaller model.** All plausible. Each changes what was measured or what production costs, so each would be its own registered item.

## Consequences

The serving-cost line in `config.py` is corrected. Any future cost figure for a model on the serving path is measured on the serving machine, not the box it was evaluated on.
