# AD-24 — Truncation blocks eligible and nothing else

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

Truncation blocks `eligible` and nothing else. The cap could cut a trial's criteria and the roll-up would still call it eligible, with a muted UI note underneath reading *criteria list truncated, roll-up may be incomplete* -- an honest caveat placed under a claim the caveat contradicts, and CLAUDE.md has named that unsound since the cap was introduced. The asymmetry is the design: a disqualifier that was found is still found and no dropped criterion can un-find it, so `excluded` stands over a truncated list, while `eligible` is the only verdict quantifying over *every* criterion and so the only one a cut list cannot support. `truncated_block` separates "unresolved because facts were unstated" from "unresolved because criteria were dropped", which are different things to tell a clinician. **Measured effect on both cohorts: none** -- verdict distributions and surfaced precision and recall are byte-identical, because `eligible` already requires zero unresolved criteria and TREC averages 3.3 per trial, so a truncated trial essentially never resolves cleanly enough to reach the blocked branch. Shipped anyway as a guarantee rather than an improvement: 29.4% of the live corpus truncates, and the eval cohorts are biased toward trials selected for gold labels rather than length

## Alternatives considered

Leaving the UI note as the mitigation, which is what CLAUDE.md already called unsound; blocking `excluded` too, which would discard a finding the cap cannot affect; folding the block into plain `cannot_determine` without a flag, which tells a clinician the facts were unstated when the truth is that the system never looked; not shipping because the measured delta is zero, which confuses a soundness guarantee with an optimisation
