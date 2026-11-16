# G2: a seeded fact-table instrument

Run 2026-11-10. Seed 20261110. 200 patient-criterion pairs. Criteria are sentences from the local TREC 2021 and 2022 corpora that parse as age, creatinine, bilirubin, platelets, ANC, hemoglobin, or ECOG. The note is rendered from the fact table. Truth is the fact table. Raw: `g2_synthetic_gold.json`.

Two builds from the same seed match. Digest `19bba404d5aed68e`. The note parser agrees with the fact table on 200/200.

That agreement is a self-check. The template writes numbers in the forms the parser reads, so 1.0 says the note still contains the fact, not that the analyst is right. The analyst was not run. The adoption bar in the plan is reproducibility from a seed, which holds.

## Decision

DONE-ADOPTED as a measurement instrument (`eval/synthetic_gold.py`). No served default changes.
