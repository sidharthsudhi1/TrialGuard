# D1: spaCy plus context rules does not reach the name-recall gate

Run 2026-11-27 (Phase 40, interim). Gate (2): recall of at least 0.95 on 200+ synthetic notes with planted unlabelled names, with CI, at zero false positives on the cohort notes.

## Sets
- **Tuning set:** 240 notes, 6 templates. The rules were written against it.
- **Holdout 1:** 240 notes, 6 templates, frozen before the rules (commit 7cc3319). Its misses then informed one round of changes, so it is no longer clean.
- **Holdout 2:** 240 notes, 6 templates, frozen after holdout 1 was read (commit 9cbbf94). Nothing was tuned on it.
- **Names:** 62 first and 62 last names across many origins, including names that are also English words (Hope, Long, Will).

## Results (`agent/sanitize.py detect_names`)

| set | recall | 95% Wilson CI |
|---|---|---|
| tuning | 240/240 = 1.000 | [0.984, 1.000] |
| holdout 1 | 182/240 = 0.758 | [0.700, 0.808] |
| holdout 2 | 112/240 = 0.467 | [0.404, 0.530] |

**False positives.** None on the 184 Phase 10 notes or the TREC 2023 notes. Two notes are flagged and both are correct. `sigir-201515` opens "Karen is a 72-year-old". `trec-202148` opens "Fernandez is a 41 year man". These are true positives, so they are listed in the test rather than counted against the gate.

**Speed.** Median 6-10 ms per note on a laptop CPU, model `en_core_web_md`.

**What fails.** A lone first name in free text, such as "Spoke with Wei by phone", "Transport arranged by Kirra" or "Kirra asked about...". Raw NER tags these as PERSON only with drug and abbreviation false positives (Lasix, Vanco, Cdiff, Dilantin). A two-token name in a shape the rules do not know ("Called X Y", "Consent witnessed by X Y") also fails. The rules hit 1.0 on the shapes they were written for, which shows the rules overfit.

## Rejected along the way
- **First-name gazetteer** (Kantrowitz/NLTK, 7,578 names). It covers 37 of the 62 first names and misses every Asian, African, Pacific and Aboriginal name in the set. For Australian users that is a biased refusal. Rejected without measuring further.
- **Treating unknown capitalised words as names** (trial vocabulary plus dictionary). Holdout 2 recall reached only 0.73-0.82 at 4-11 false-positive notes, so it fails on both sides.

## Next candidate
A transformer de-identification model trained on clinical notes, such as a RoBERTa i2b2 de-id model or the Stanford deidentifier. It would be measured on Modal on the same three sets, with gate (5) latency measured on Fly-class CPU. Larger image and RAM are the costs to price. The rules stay in as a backstop either way, because they are precise.
