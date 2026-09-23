# AD-21 — The missing criteria were mostly the parser

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

The analyst's "missing criteria" were mostly not the analyst. Decomposed against the cached corpus, TREC's 18.9% shortfall is 8.4% entries that are not criteria at all (section headers, stranded labels, placeholders), 7.2% criteria the model answered while echoing them differently, and **3.3% genuine omission**. Four interventions had been aimed at a 13% compliance problem that was really ~3%, which is why v5's numbering worked by accident of format, v6's instruction did nothing, and the retry's gains were half measurement artifact. Fixed deterministically and without the model: the parser stops emitting non-criteria (structural rule, never length, since "Karnofsky 60-100%" and "Age below 18" are real criteria of three words), and `align_assessments` pairs by containment one-to-one instead of exact text. The residual ~3% is stated rather than closed, and the roll-up already reports affected trials as `needs_review` rather than claiming a verdict over them

## Alternatives considered

A length filter for junk (deletes every short real criterion); splitting compound lines on sentence boundaries (eligibility text is full of "1.5 mg", "i.e." and "No. 3"); relaxing the ten-character line minimum, which would recover "Pregnancy" and "Prisoners" but also admit wrapped-line fragments, so it needs its own measurement; another prompt version, now that the prompt is known not to be the constraint
