# AD-43 — A unit-aware numeric check on the quote is not a badge

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

A deterministic check that compares a decisive verdict with the number in its own quote, after unit conversion (`verify/unit_check.py`), does not ship as a `numeric_conflict` badge. On G2b's templated notes it catches all 23 wrong verdicts with no false flags. On real notes in the analyst cache it raises 629 flags over 8,829 in-scope entries, and a seeded sample of 60 read by Claude is right on only 21 (precision 0.35 against a 0.80 bar written before the run). 27 of the 39 wrong flags are criteria whose verdict rests on a non-numeric condition (sex, pregnancy, diagnosis), and 12 are parser errors. [`e6_unit_findings.md`](../../data/reports/e6_unit_findings.md).

## Alternatives considered

- **Ship on G2b's result.** G2b suggested the rule and its criteria carry no other conditions, so it cannot judge it.
- **Keep only `met` flags.** 7 of 10 right in the sample: below the bar on too few items to set one.
- **Limit scope to purely numeric criteria.** Plausible, since every correct flag sat on one, but it needs its own registered bar and a fresh audit sample, not a retune on this sample.

## Consequences

The entailment gap stays open with a firmer number behind it: about 12% of decisive verdicts on templated numeric criteria are grounded and wrong (G2b), and real-note flags suggest numeric errors of roughly this kind exist in about 2.5% of in-scope cache entries (a rough estimate over mixed prompts and models).
