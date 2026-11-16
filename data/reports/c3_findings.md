# C3: an 8B keyword model is faster to call and worse at recall

Run 2026-11-10. Llama-3.1-8B-Instruct-Turbo against the cached Llama-3.3-70B keyword lists. Mean per-patient recall, `TG_INDEX_EXCLUSION=0`. Non-inferior means the paired delta (8B minus 70B) has a 95% CI lower bound of at least -0.01, and the cold keyword call has p50 at most 2 seconds. Raw: `c3_keyword_model.json`.

The 8B model wraps the JSON in a sentence. The parser now reads the object out of that wrapper. Two TREC 2021 notes and one TREC 2022 note still did not parse and fell back to the raw note (2021 fallbacks recorded as 0 on the rerun because those two were recovered; 2022 records 1).

| cohort | n | 70B @100 | 8B @100 | delta [CI] | 70B @200 | 8B @200 | delta [CI] |
|---|---|---|---|---|---|---|---|
| TREC 2021 | 75 | 0.491 | 0.459 | -0.031 [-0.051, -0.014] | 0.668 | 0.629 | -0.039 [-0.060, -0.021] |
| TREC 2022 | 50 | 0.520 | 0.457 | -0.064 [-0.107, -0.025] | 0.674 | 0.611 | -0.063 [-0.104, -0.030] |

Every interval sits below -0.01. Cold keyword latency on TREC 2022, the arm that actually called the model, has p50 11.7 seconds. The TREC 2021 p50 of 1 ms is the rerun reading the disk cache from the first pass, not a generation time.

## Decision

DONE-REJECTED. C4, which was only going to run if this passed and latency still mattered, does not run.
