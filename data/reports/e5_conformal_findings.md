# E5: false exclusion falls under 10% only if one disqualifier no longer excludes

Run 2026-11-10. Cached assessments, top 100, `TG_LIST_POOL=50`, prompt v4. TREC 2021 index includes exclusion text; TREC 2022 uses the cached index that does not (`TG_INDEX_EXCLUSION=0`). Alpha 0.10, written before the run. A threshold t excludes only when at least t criteria disqualify. Raw: `e5_conformal.json`.

| cohort | gold-eligible assessed | FER at t=1 (served) | FER at t=3 | gold-excluded assessed | of those, released at t=3 |
|---|---|---|---|---|---|
| TREC 2021 | 1880 | 0.388 | 0.068 | 1622 | 0.534 |
| TREC 2022 | 1538 | 0.296 | 0.054 | 825 | 0.502 |

t=3 is the smallest threshold that holds alpha on each cohort, and it holds on the other cohort in both directions.

The same threshold would stop excluding about half of the trials the qrels mark excluded. The pre-registered rule only constrained false exclusion of gold-eligible trials. It did not constrain that release. The served rule stays one disqualifier. Switching it on this evidence would trade a 30-39% false-exclusion rate for missing half of the true exclusions, and that trade was not the bar.

## Decision

DONE-REJECTED as a change to `rollup_trial`. The number stands: a single `not_met` excludes gold-eligible trials 39% and 30% of the time on these assessed pools.
