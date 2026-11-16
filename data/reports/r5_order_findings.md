# R5: ordering needs_review by n_unknown does not raise nDCG

The exclusion-hit feature was inside R1. R1 lost to deep fusion, so that feature is not revived as a filter or as a ranker. What remained was the in-tier order. The UI already sorts `needs_review` by `n_unknown` ascending. This measures that order against leaving the retrieval order, nDCG@10 inside the non-excluded set, same cached run as E5. Kill, written before the run: change the served order only if n_unknown order wins on both TREC cohorts at BH p < 0.05.

| cohort | n | mean delta (n_unknown minus retrieval) | 95% CI | better / worse | p_bh |
|---|---|---|---|---|---|
| TREC 2021 | 75 | -0.024 | [-0.063, 0.015] | 33 / 42 | 0.399 |
| TREC 2022 | 50 | -0.007 | [-0.055, 0.040] | 26 / 24 | 0.931 |

Neither delta is distinguishable from zero. The point estimates are negative.

## Decision

DONE-REJECTED for an IR change. The clinician sort by unstated facts stays, because that is the tier contract, not because it ranks gold higher.
