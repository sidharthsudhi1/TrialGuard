# D1: a clinical de-id model plus context rules refuses unlabelled names

Run 2026-11-27 (Phase 40). Adopted: `obi/deid_bert_i2b2` at score ≥ 0.3 on PATIENT/STAFF spans, plus the context rules in `agent/sanitize.py`. Raw: `d1_jobs.json`, `d1_entities.json`, `d1_scores.json`.

## Gates

| gate | result |
|---|---|
| (1) zero false positives on the 184 Phase 10 notes | **0**, and also 0 on the 40 TREC 2023 notes |
| (2) recall ≥ 0.95 on 200+ planted-name notes, with CI | **0.971 [0.941, 0.986]**, 233/240 on holdout 2 |
| (5) cold search median latency up by no more than 500 ms | **+100 ms** median per note on 2 CPUs (Modal, matching the Fly VM); 619 ms at the 4,000-character cap |

Two cohort notes are flagged, and both are correct. `sigir-201515` opens "Karen is a 72-year-old". `trec-202148` opens "Fernandez is a 41 year man". Each opens on the patient's name, which is exactly what this pass exists to refuse. `tests/test_phi.py` lists them by id, so they are not counted as false positives. This departs from the literal Phase 10 wording and needs the project owner's sign-off.

The lower CI bound on recall is 0.941, below 0.95. The point estimate clears the gate; the interval does not exclude a true recall slightly under it.

## Sets

Each set has 240 notes in 6 templates. Names come from a list of 62 first and 62 last names across many origins, including names that are also English words (Hope, Long, Will).

- **Tuning set:** the context rules were written against it.
- **Holdout 1:** frozen before the rules (commit 7cc3319). It was then used twice: to see why the rules failed, and to choose the score threshold.
- **Holdout 2:** frozen after holdout 1 was read (commit 9cbbf94). No rule, model or threshold was chosen on it, so this is the gate number.

## Candidates

The candidates ran on Modal on 2 CPUs with a 512-token window and stride 64. Each cell shows recall and the number of false-positive cohort notes. The last column is the median per cohort note.

| candidate | tuning | holdout 1 | holdout 2 | median ms |
|---|---|---|---|---|
| spaCy `en_core_web_md` + rules | 1.000 / 0 | 0.758 / 0 | 0.467 / 0 | 6-10 (laptop) |
| `obi/deid_roberta_i2b2` @0.5 | 1.000 / 1 | 1.000 / 1 | 0.967 / 1 | 1,109 |
| `obi/deid_bert_i2b2` @0.5 | 0.988 / 2 | 0.983 / 2 | 0.946 / 2 | 100 |
| `StanfordAIMI/stanford-deidentifier-base` @0.5 | 0.983 / 0 | 0.971 / 0 | 0.879 / 0 | 229 |
| `StanfordAIMI/stanford-deidentifier-only-i2b2` @0.5 | 0.983 / 0 | 0.896 / 0 | 0.854 / 0 | 265 |
| `dslim/bert-base-NER` @0.5 | 1.000 / 23 | 1.000 / 23 | 0.971 / 23 | 336 |
| **adopted: obi BERT @0.3 + rules + eponym filter** | 0.996 / 0 | 0.988 / 0 | **0.971 / 0** | 100 |

- **obi RoBERTa** has the best raw recall, but it fails gate (5): 1.1 s median and 5.7 s at the cap.
- **The Stanford models** are clean on false positives, but they miss lone first names.
- **obi BERT's two false positives** are eponyms where the model's span stopped mid-word: "Kayser-Fleischer' ring" and "Hoehn-Yahr Stage". The eponym check now reads to the end of the hyphenated word and includes "ring" and "stage". Those two suffixes came from cohort notes, so the false-positive gate is partly tuned.
- **The threshold** came from holdout 1: 0.3 to 0.5 tie on false positives, and 0.3 has the best recall.

## Rejected along the way

- **spaCy plus rules.** The rules overfit the shapes they were written for: 1.0 on the tuning set, 0.47 on holdout 2. Lone first names were missed, and raw spaCy PERSON tags pull in drug names (Lasix, Vanco, Cdiff).
- **A first-name gazetteer** (Kantrowitz/NLTK). It covers 37 of the 62 first names and none of the Asian, African, Pacific or Aboriginal ones, which makes it a biased refusal for Australian users.
- **Unknown capitalised word = name.** Holdout 2 recall was 0.73-0.82, with 4-11 false-positive notes.

## What the 7 holdout-2 misses look like

Mostly a lone first name in free text that is also an English word ("Grace", "Bill", "Sky"), or a name after an unseen verb ("Consent witnessed by Fatima Cross"). Recall is a property of these templates and names, not of real notes. The planted set measures shapes the cohorts do not contain.

## Serving

- **Image:** the model (~430 MB) is baked into the API image and checked offline at build time.
- **Startup:** `warm_models()` loads it at startup.
- **Order:** the refusal still runs at the boundary, before the keyword call and before any trace.
- **Tests:** unit tests replay the recorded entities. A CI step with `TG_DEID_LIVE=1` checks that the live model still matches the recording.

Cost: Modal CPU only, $0 LLM.
