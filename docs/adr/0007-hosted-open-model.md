# AD-7 — A hosted open model for inference

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

Groq free-tier hosted open model

## Alternatives considered

Local quantised LLM, paid frontier API

## Amended, Phase 8

Metered host (DeepInfra) for the same open model. The free tier's throughput ceiling, not its price, blocked two Phase 4 measurements for weeks; the whole outstanding set then cost $0.0958. The host serves only an FP8 build, so a parity gate ran *before* adoption — quantization changes numerical precision on the model producing verbatim quotes, and that failure is silent (a paraphrased quote just becomes an honest *unverifiable*). Citation precision was unchanged on a matched baseline arm. See [`phase8_provider_parity.md`](../../data/reports/phase8_provider_parity.md)

**Alternatives considered.** Staying free (throughput-blocked); a paid frontier API (breaks comparability with every committed number); Together/Fireworks at full precision (~9x the price, held as the fallback had parity failed)
