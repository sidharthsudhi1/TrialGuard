# E6: a unit-aware numeric check on the quote

Run 2026-11-18. Rule and bars were written in the plan before the check touched any item. `$0`, no LLM. Raw: `e6_unit_check.json`, `e6_unit_audit.jsonl`, `e6_unit_audit_labels.json`. Code: `verify/unit_check.py`, `scripts/e6_unit_check.py`.

## Question

G2b found that every wrong decisive verdict carried a verbatim quote holding the right number. The open question was whether a check comparing the verdict with the number in its own quote, after unit conversion, is worth a badge on real notes.

## Result against the registered bars

| Bar | Result | Pass |
|---|---|---|
| Human WS-5b items: at most 2 sound citations flagged | 0 flags (3 of 50 items in scope) | yes, vacuously |
| Real notes: flag precision >= 0.80 and >= 10 correct flags | **0.35** (21 of 60 sampled flags correct) | **no** |
| G2b (development, not gated) | 23/23 wrong caught, 0 of 173 right flagged | n/a |

The analyst cache holds 63,983 files: 210,874 unique decisive (criterion, verdict, quote) entries, of which 8,829 have a numeric clause in scope, raising 629 flags. A random 60 (seed 20261118) were read by Claude. That audit is a model's, not a clinician's. Model-labelled e7 items: 1 flag, on a sound citation.

**Decision: rejected.** The check is perfect on templated notes and wrong two times in three on real ones.

## Why it fails on real notes

| Cause of a wrong flag | Count of 60 |
|---|---|
| Criterion has a non-numeric condition (sex, pregnancy, diagnosis, BMI), and the verdict rests on it | 27 |
| Parser error: upper bound without the word "age", `or older` exemption letting a real disjunction through, hours or gestational or maternal age read as patient age | 12 |
| Correct: the verdict contradicts the quoted number | 21 |

The rule treats a criterion with one numeric clause as purely numeric. Real criteria rarely are: "Women, age 30-70 years" with a 60-year-old man is a right `not_met`. That's 27 of 39 wrong flags, and it's a scope problem, not a parsing one. Templated G2b criteria have no other conditions, which is why the development set looked perfect.

## What the 21 correct flags show

They are real, and some are blunt: "Age 18-75 years" with a 72-year-old, `not_met`; "Absolute Neutrophil Count ≥ 1.5x10^9/L" with "ANC 1594/uL", `not_met`; "Women aged 20-40 years" with a 43-year-old, `met`; "Initial haemoglobin concentration <90 g/L" with "Hb is 9.0 g/dL", `met`. So grounded-but-wrong numeric verdicts do occur on real notes, roughly 0.35 × 629 ≈ 220 of 8,829 in-scope entries (~2.5%). This is only a rough estimate. The cache mixes prompt versions and models from earlier A/Bs, so it isn't a rate for the served v4 path.

## Exploratory (after the audit, not registered)

- `met` flags were right 7 of 10, `not_met` flags 14 of 50. A `met`-only rule would still miss 0.80 on this sample, and 10 items can't carry a bar.
- Every correct flag sits on a criterion that is numeric and nothing else, or close to it. A future item could limit scope to criteria with no non-numeric content words, re-register its bar, and draw a fresh audit sample. That would be a new item, not a rescue of this one.

## Decision

DONE-REJECTED. No badge, no default change. ADR: [AD-43](../../docs/adr/0043-numeric-quote-check-not-adopted.md).
