# AD-16 — Prompt v5 measured and not adopted

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

Prompt v5 addresses criteria by index instead of echoing their text. Measured on both cohorts and **not adopted**, despite passing its own acceptance test. Grounded counts and verdict distribution are superior on both — the opposite of L4, whose tell was grounded falling while abstention rose — but three things the acceptance did not ask about go the wrong way: TREC surfaced recall falls 26%, the self-referential quote count rises 4x on SIGIR and 20x on TREC (widening the machine-visible slice of the entailment gap AD-15 puts at 36%), and the latency case that revived L1 held on one cohort of two (−32% SIGIR, +3% TREC, because on TREC v5 answers 7% more criteria per call). Output per criterion fell 22–26%, not the ~50% a 43.4% echo share implied. Kept behind `TG_PROMPT_VERSION=v5`, v1–v4 untouched. **The defect it exposed outranks it:** under v4 the analyst silently answers 13.0% fewer criteria than it is asked on TREC — not truncation, `max_tokens` is 4096 against a 1,732-token longest response — and an unanswered criterion produces no assessment, so it cannot fail grounding, cannot trigger the retry edge, and vanishes into `needs_review` while the roll-up still calls the trial eligible over what it did receive

## Alternatives considered

Adopting v5 on the strength of its own acceptance criterion (passes, but buys a recall regression with a latency win that did not arrive); shipping the coverage fix inside the addressing change (confounds two effects — the clean follow-up is a v6 keeping v4's addressing plus only v5's "one object per criterion" instruction); clamping an out-of-range index instead of dropping it (files a verdict under a criterion nobody assessed, and the quote grounds against the trial's whole text either way, so nothing downstream could catch it)
