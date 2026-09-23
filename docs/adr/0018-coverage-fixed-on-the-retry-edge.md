# AD-18 — Skipped criteria are fixed on the retry edge

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

The analyst skipping criteria is fixed on the retry edge, not in the prompt, and the fix is on by default. Under v4 it never answered 13.0% of TREC criteria; a criterion with no assessment cannot fail grounding, so it never reached the retry edge, and `rollup_trial` still called a trial `eligible` over the subset it did receive. v5's numbering fixed coverage at 26% of surfaced recall (AD-16) and v6's instruction alone did nothing (AD-17), so the prompt was ruled out by measurement. Re-asking for the missing criteria takes them 230 → 167 (13.1% → 9.5%), grounded 848 → 918, `eligible` precision 0.571 → 0.714 and end-to-end recall 0.0026 → 0.0033, while surfaced recall falls 12%. **That cost was itself an artifact** and is corrected in AD-22: with the parser fixed the retry improves surfaced recall instead of costing it, because the penalty was the retry re-asking about section headers. **The coverage figures here are also corrected ones** — the originals (229 → 71, 13.0% → 4.0%) were computed with a metric that subtracted answered from asked, which nets criteria the model answered without being asked against criteria it skipped, and so flattered the retry arm specifically; see AD-20 and `p1_correction.md`. Adopted despite that, because the recall removed was produced by not looking: v4's extra surfaced trials include ones called eligible over a list that was silently 13% short, which CLAUDE.md already calls unsound. SIGIR is the control — 12 criteria skipped there, and the fix moves surfaced recall not at all. `TG_RETRY_MISSING=0` reproduces pre-2026-09-10 retry numbers

## Alternatives considered

Leaving it open with two prompt routes eliminated (ships a roll-up the repo's own docs call unsound); adopting v5 for the same coverage at twice the recall cost and a 20x rise in self-referential quotes; keeping the fix behind a default-off flag like L4's, which would have made the sound path the one nobody runs
