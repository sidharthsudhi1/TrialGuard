# G1: deleting the quoted sentence flips 65% of verdicts

Run 2026-11-10 on the 50 human WS-5b items. One directional edit deletes the sentence that overlaps the quote. One invariance edit moves an unrelated sentence. Success for directional is a flipped verdict or `cannot_determine` / `unverifiable`. Success for invariance is the same verdict. Scope was fixed before any mutated note was assessed. Raw: `g1_metamorphic.json`.

| edit | scored | success | rate | skipped, quote not locatable | errors |
|---|---|---|---|---|---|
| directional | 46 | 30 | 0.652 | 3 | 1 |
| invariance | 43 | 26 | 0.605 | 7 | 0 |

A third of directional edits leave the verdict in place after the sentence the quote came from is gone. Moving an unrelated sentence changes the verdict 40% of the time. Postgres cache reads failed once (`OperationalError`); the run used the disk cache and still scored.

## Decision

DONE-ADOPTED as a measurement. No served default changes. The rates are the criterion-level numbers this cohort can support without new human labels.
