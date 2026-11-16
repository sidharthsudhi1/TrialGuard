# E1-E3: three deterministic entailment badges, none adopted

Run 2026-11-10. Gold is the 50 human WS-5b items (`e7_combined_gold.jsonl`, `label_source=human`). The 300 model-labelled items are in the same JSON and were not used to set the bar. Kill, written before the run: a `weak_entailment` badge ships only if a check flags at most 2 sound citations and at least 80% of its flags are unsound. Zero flags is a rejection. Nothing here changes a verdict. `TG_WEAK_ENTAILMENT` stays off. Raw: `e1e3_entailment.json`.

| check | flagged | sound damaged | unsound caught | precision | recall |
|---|---|---|---|---|---|
| E1 ConText scope | 10 | 6 | 4 | 0.40 | 0.22 |
| E2 numeric facts | 0 | 0 | 0 |  | 0.00 |
| E3 conjunct coverage | 1 | 1 | 0 | 0.00 | 0.00 |

18 of the 50 citations are unsound. E1 damages more sound citations than it catches. E2 can parse 4 of the 50 criteria (age bounds). On those four the note either agrees with a sound verdict or has no number the checker will use, so it never fires. E3 flags one sound citation and no unsound one.

The model-labelled set is the same shape and worse: E1 precision 0.24 (37 sound damaged), E3 precision 0.45. It does not rescue a threshold.

## Decision

DONE-REJECTED for all three as a shipped badge. The checks stay in `verify/entailment.py` so the measurement reruns. They do not stamp assessments unless `TG_WEAK_ENTAILMENT=1`.
