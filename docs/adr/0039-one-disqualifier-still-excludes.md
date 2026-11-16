# AD-39 — One disqualifier still excludes

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

`rollup_trial` still excludes on a single disqualifier. On the cached top-100 pools, that rule false-excludes gold-eligible trials at 0.388 (TREC 2021, n=1880) and 0.296 (TREC 2022, n=1538). Requiring three disqualifiers brings those rates to 0.068 and 0.054, under the pre-registered 0.10, in both calibration directions. It also releases 0.534 and 0.502 of the trials the qrels mark excluded. The pre-registered rule did not bound that release. The served rule is unchanged. [`e5_conformal_findings.md`](../../data/reports/e5_conformal_findings.md).

Ordering `needs_review` by `n_unknown` instead of retrieval order does not raise nDCG@10 (deltas -0.024 and -0.007, neither significant). The clinician sort stays. [`r5_order_findings.md`](../../data/reports/r5_order_findings.md).

## Alternatives considered

- **Exclude only at three disqualifiers.** Meets the false-exclusion cap and drops about half of true exclusions.
- **Rank the tier by n_unknown because it retrieves better.** It does not, on this measurement.
