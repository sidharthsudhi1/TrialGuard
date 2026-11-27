# B1: 1 in 8 surfaced trials contradicts a stated biomarker, and a quarter of gold is observational

Run 2026-11-27 (Phase 39). Measurement only; no served default changes. Scope and gate were written in `docs/MVP_PLAN.md` before any number was seen. Raw: `b1_contradiction.json`, `b1_audit.json`, `b1_study_types.json`, `b1_rankings.json`, `b1_jobs.json`.

## 1. Biomarker contradiction in the top 25

A contradiction is a surfaced trial whose inclusion criteria require a biomarker status that the note explicitly contradicts. The detector is deterministic (`eval/biomarker_contradiction.py`). It counts only stated statuses (HER2-positive, ER/PR negative, KRAS wild-type, no EGFR mutation). A marker the note leaves out is never counted as a contradiction. A trial that allows both statuses (cohorts, "+/-", "and/or", "HR+ or TNBC") requires neither. Every flagged pair was then read by hand against the trial text.

Patients are E3's oncology patients (disease-stem regex) with at least one gold-eligible trial. Rankings are the served retrieval shape at list pool 200 with cached keywords: the V4 TREC 2023 rankings, plus Modal runs for the rest. A note with no stated status cannot contradict a trial, so it counts its 25 trials with zero contradictions and needs no ranking.

| set | patients | surfaced | flagged | confirmed | detector precision | rate (audited) | 95% CI |
|---|---|---|---|---|---|---|---|
| all oncology, 4 cohorts | 23 | 575 | 29 | 22 | 0.76 | **0.038** | [0.010, 0.073] |
| notes stating a status | 7 | 175 | 29 | 22 | 0.76 | **0.126** | [0.063, 0.200] |
| same 7, every status flipped | 7 | 175 | 26 | 25 | 0.96 | 0.143 | [0.040, 0.257] |

The oncology set is small, and the report has to say so. SIGIR has 1 oncology patient, TREC 2021 has 14, TREC 2022 has 1 and TREC 2023 has 7. Only 7 of the 23 state any biomarker: 2 in TREC 2021 and 5 in TREC 2023, all breast cancer with HER2/ER/PR. EGFR, ALK, KRAS and the rest never appear in a gold-labelled oncology note, so the rate says nothing about lung or colorectal markers. The CIs are patient-bootstrap intervals over 23 and 7 patients.

**Metamorphic flips.** Each of the 7 notes had every stated status inverted (HER2-positive became HER2-negative, ER+/PR- became ER-/PR+). Its cached keywords were flipped the same way, with no keyword LLM call. Flipped keywords that contradict the flipped note were dropped (3 of 80: an implied HR status that did not move, and two "triple ..." keywords). The notes were then re-retrieved. Paired per patient, the audited rate moves by **+0.017 [-0.040, +0.069]**. A biomarker-blind retriever, meaning the original ranking scored against the flipped statuses, would score 0.189 [0.074, 0.320] flagged (not audited). The re-retrieved flipped notes score 0.149 flagged. So retrieval responds to the stated status a little, and with n=7 that is not separable from no response.

**Detector errors.** All 7 false positives fall on the two TREC 2021 HER2+/ER-/PR- notes. They come from clauses the detector reads as requirements:

- a treatment rule ("hormonal therapy in cases with HR+ tumor")
- an exclusion stated inside inclusion ("triple positive breast cancers are not eligible")
- a threshold disjunction ("ER/PR positive cells <10%, or HER2 positive")
- an extra lesion that is allowed, not required

A hard filter built on this detector as it stands would have dropped 7 trials these patients qualify for, against 22 correct removals.

## 2. Observational share of gold-eligible trials

Study type comes from CT.gov's current record for every gold-eligible trial (`lookup_study_types`, 2026-11-27). Shares use unique trials with a 95% Wilson interval.

| cohort | gold-eligible trials | observational | share | 95% CI | not found |
|---|---|---|---|---|---|
| SIGIR | 408 | 192 | **0.471** | [0.423, 0.519] | 3 |
| TREC 2021 | 5,123 | 1,641 | **0.320** | [0.308, 0.333] | 1 |
| TREC 2022 | 3,807 | 1,018 | **0.267** | [0.254, 0.282] | 0 |
| TREC 2023 | 7,454 | 1,827 | **0.245** | [0.236, 0.255] | 2 |

Expanded-access trials (8 to 16 per cohort) count as not observational.

## 3. Observational share of the served corpus

All 26,258 active `ctgov_live` trials (read-only query, 2026-11-27): 6,197 observational, **0.236 [0.231, 0.241]**, 0 not found.

## Decision

DONE as a measurement. Nothing adopted, per the gate.

- **Phase 43 (biomarker filter) is not cancelled.** Its cancel line is an audited rate under 2% of top-25. The point estimate is 3.8% and the CI [1.0%, 7.3%] straddles 2%, so the evidence does not settle it either way. Phase 43 has to beat the detector's precision before it filters anything: 0.76 on the original notes means 1 in 4 removals would hide a trial the patient qualifies for. Demotion is safer than a hard filter until that improves, and this audit set is the regression test.
- **Phase 42's default `INTERVENTIONAL` filter is not supported by gold.** A quarter to a half of the trials the cohort assessors judged eligible are observational. They are a slightly larger share of gold than of the corpus (0.245 to 0.471 against 0.236). An interventional-only default would remove that much of the gold before scoring, so the filter belongs behind a toggle and not on by default, unless a coordinator cohort says otherwise.
- **Schema.** `StudyType` is now fetched, stored as `trials.study_type` and part of the content hash. `scripts/migrate_study_type.py` must be applied to Neon before the code deploys. The first refresh after that rewrites every row as a metadata-only update, with no re-embed.

Cost: $0 LLM. Modal ran one 26k-trial and one 451k-trial retrieval pass for 9 queries. CT.gov lookups are free.
