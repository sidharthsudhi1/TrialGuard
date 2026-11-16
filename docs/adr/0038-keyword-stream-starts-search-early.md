# AD-38 — Keyword extraction streams, and the cold path is still the LLM

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

On a cache miss, lexical search for a keyword starts when that keyword's JSON string closes, and the dense scores are one GEMM after the list is complete. The parsed list is what gets fused. If the stream and the parse disagree, the early searches are dropped. On three cold notes the first keyword arrived before the call ended (5.1 s / 9.9 s and 5.6 s / 14.1 s on the two ordinary calls; one call stalled at 61 s / 67 s). `TG_KEYWORD_STREAM=0` is the blocking path. The 1.5 second time-to-first-results target is not met: 5 seconds is still the model. [`c1_findings.md`](../../data/reports/c1_findings.md).

## Alternatives considered

- **Keep the blocking call.** Identical rankings, and the first search waits for the last keyword.
- **Return raw-note hits as the final list.** Measured separately as C2. It is a different ranking.
