# AD-9 — Notebook GPUs for batch jobs only

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

Kaggle/Colab for batch jobs only

## Alternatives considered

Always-on GPU, local only

## Unexercised

Notebook GPU never needed through Phase 3 — MedCPT (110M) embeds on local CPU/MPS, and the eval bottleneck was Groq token quota (disk cache), not GPU hours. `notebooks/` stays empty; revisit only if Phase 4/5 batch work exceeds local compute

**Alternatives considered.** —
