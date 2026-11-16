# AD-41 — Small span detectors are not an entailment badge

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

MiniCheck (Flan-T5-Large) and LettuceDetect (ModernBERT base) do not ship as a `weak_entailment` badge. On the 50 human WS-5b items, MiniCheck reaches AUC 0.70 and its least damaging threshold still flags 6 sound citations to catch a third of the unsound ones. LettuceDetect reaches AUC 0.59 to 0.63 and rejects almost every citation at every threshold. The bar, written before the run, was at most 2 sound citations damaged with at least 25% of unsound ones caught. Premising the note sentence instead of the quote does not change either result. `TG_WEAK_ENTAILMENT` stays off. Bespoke-MiniCheck-7B was not run. [`e4_findings.md`](../../data/reports/e4_findings.md).

## Alternatives considered

- **Badge at MiniCheck's best threshold anyway.** Six sound citations in fifty is three times the allowed damage.
- **Set the threshold on the 300 model-labelled items.** Those labels are lenient and were ruled out as a bar before the run.
