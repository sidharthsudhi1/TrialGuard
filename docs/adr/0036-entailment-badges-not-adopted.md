# AD-36 — Three deterministic entailment checks, none adopted as a badge

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

ConText scope (E1), numeric comparison against a number taken from the note (E2), and conjunct coverage (E3) do not ship as a `weak_entailment` badge. On the 50 human WS-5b items, E1 flagged 10 citations and damaged 6 sound ones (precision 0.40). E2 flagged none. E3 flagged one, and it was sound. The bar, written before the run, was at most 2 sound citations damaged and precision at least 0.80. `TG_WEAK_ENTAILMENT` stays off. Verdicts are unchanged. [`e1e3_findings.md`](../../data/reports/e1e3_findings.md).

## Alternatives considered

- **Force `unverifiable` when a check fires.** The plan allows that only at zero damage on the human set. None of the three got there.
- **An LLM or NLI verifier.** Already rejected (AD-3, AD-23, AD-29, AD-32). These checks were the deterministic alternative.
