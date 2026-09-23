# AD-20 — criterion_unanswered is counted, not subtracted

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

`criterion_unanswered` is counted against the criteria list, not computed as asked minus answered. The subtraction is not the number of criteria the analyst skipped: the model also returns assessments naming criteria that were never asked, and each one cancels a genuinely skipped criterion out of the total. It went **negative** on SIGIR, which is impossible and is what exposed it, and it had already been used to state AD-18's headline — flattering the retry arm, because a retry produces more output and therefore more drift (unmatched entries nearly double on TREC when it is on). Corrected: the coverage fix is 13.1% → 9.5%, not 13.0% → 4.0%. `criterion_unmatched` is now reported beside it, because a count that can be cancelled by a second phenomenon should never have been one number

## Alternatives considered

Leaving the subtraction and treating the negative as a display quirk (it was load-bearing in a merged claim); dropping unmatched entries so the subtraction becomes valid again (changes what the roll-up sees to make a metric tidy, and those entries are real model output worth counting); re-deriving from `n_criteria` alone, which is the same subtraction wearing a different name
