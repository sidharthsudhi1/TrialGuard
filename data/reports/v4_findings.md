# V4: TREC 2023, questionnaires instead of notes

Run 2026-11-16. 40 synthetic topics. Retrieval on Modal against the
`medcpt_excl` index (451,538 trials). Keywords are the served 70B extractor
($0.0017, no raw-note fallback). The analyst is prompt v4 on the keyword
top-100, DeepInfra 70B, $1.23, completion 1.0, 0 errors. Three topics have no
gold-eligible trial (`trec-20232`, `trec-202311`, `trec-202340`) and are not
scored. Every labelled eligible trial is in the index. Raw: `v4_retrieval.json`,
`v4_e2e.json`.

Not on the API. No served default changes.

## Retrieval

Mean per-patient recall of gold-eligible trials. The interval is the paired
difference, keywords minus the raw questionnaire.

| depth | keywords | raw questionnaire | keywords minus raw |
|---|---|---|---|
| 100 | 0.0678 | 0.0724 | -0.0047 [-0.0170, 0.0084] |
| 200 | 0.1222 | 0.1266 | -0.0043 [-0.0242, 0.0168] |
| 500 | 0.2320 | 0.2121 | +0.0200 [-0.0088, 0.0496] |

n=37. Every interval includes zero. On a structured questionnaire, keyword
extraction does not raise recall at pool depth. The micro retrieval ceiling
inside the top-100, which is what the analyst saw, is 0.0552 (11,667
gold-eligible trials).

## Analyst

Top-100, 3,643 trials assessed, 57 skipped because they had no criteria.
Completion 1.0.

| | |
|---|---|
| surfaced recall | 0.0327 [0.0247, 0.0427] |
| surfaced precision | 0.830 |
| pool base rate | 0.605 |
| surfaced lift | 1.37x [1.25, 1.53] |
| eligible-tier recall | 0.0063 |
| criterion unverifiable | 0.0092 (216) |
| criterion cannot_determine | 0.4345 (10,372 of 23,871 asked) |
| criterion grounded | 0.5495 (13,118) |

Trials: 2,138 excluded, 1,247 cannot_determine, 258 eligible.

The lift is real and the retrieval ceiling is the constraint. Surfaced recall
is 0.033 because the top-100 holds only 0.055 of the gold, not because the
tier throws the gold away. Unverifiable stays under 1%. About 43% of asked
criteria come back cannot_determine, which is the questionnaire leaving facts
unstated, not a grounding failure. Decisive verdicts are still grounded or
forced unverifiable.
