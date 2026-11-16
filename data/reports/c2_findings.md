# C2: a raw-note list is a weak provisional ranking

Run 2026-11-10. Mean of per-patient recall@100, patients with at least one in-corpus eligible trial. Index is the cached MedCPT file index with `TG_INDEX_EXCLUSION=0` (the TREC 2022 cache; TREC 2021 was run the same way so the two arms share a build). This is not the pooled headline recall. Raw: `c2_raw_note_recall.json`.

| cohort | n | raw note | keywords |
|---|---|---|---|
| TREC 2021 | 75 | 0.200 | 0.491 |
| TREC 2022 | 50 | 0.111 | 0.520 |
| SIGIR | 52 | 0.316 | 0.766 |

The raw note keeps well under half of the keyword recall on every cohort. Kill, written before the run: the final keyword ranking does not change when a provisional list is also computed, and assess stays off until that final list exists. Both hold (`tests/test_keyword_stream.py`, the search page disables assess while `provisional` is set).

## Decision

DONE-ADOPTED as a labelled preview, not as a ranking. `/api/search/stream` emits the raw-note list, then replaces it. The sync `/api/search` response is still the keyword list only.
