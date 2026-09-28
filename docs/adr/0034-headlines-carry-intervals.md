# AD-34 — Headlines carry intervals, and lift uses one denominator

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

Every end-to-end headline is reported with its n and a patient-bootstrap 95% CI, and experiments with more than one arm report Wilcoxon p-values adjusted with Benjamini-Hochberg within their family. That is the procedure Otero, Parapar & Barreiro (ECIR 2025) found keeps Type I error nominal at TREC topic counts. `end_to_end` now computes lift itself instead of leaving it to each report, and it counts labelled trials on both sides of the ratio, the same convention as surfaced precision.

Running the harness over the published table caught two errors ([`v1_findings.md`](../../data/reports/v1_findings.md)):

- **E3 used mismatched denominators.** It divided precision over labelled trials by a base rate over all assessed trials, which inflates lift. On its own cells, the corrected lifts are **all 1.37x [1.26, 1.49], oncology 1.51x [1.22, 1.99], non-oncology 1.34x [1.23, 1.46]**. E3 published 1.65x, 1.90x and 1.61x. So "the agent is better inside oncology" is not supported on lift, and neither cell clears 1.60x.
- **The n=20 headline came from favourable patients.** Under one config, the 20 patients behind every end-to-end headline give 1.56x at top-10 and the other 55 give 1.30x. Flag changes since then move lift by 0.03; patient selection moves it by 0.26.

The 10→100 widening holds up: 20 of 20 patients improve on both cohorts, BH-adjusted p ≤ 0.0001. Minimum detectable effect on surfaced recall is about 0.07 at n=20 and 0.036 at n=75 on TREC 2021, so full-cohort runs are the default for any claim smaller than that.

## Alternatives considered

- **Keeping Fisher's exact test for retrieval deltas.** Fisher treats trials as independent, but a patient's trials are correlated, which overstates n.
- **Bonferroni.** It's too conservative at 10+ arms; BH controls the false discovery rate, which is the risk that matters for an adopt/reject log.
- **Reporting E3's lift with a footnote.** The error changes the sign of a README claim, so the numbers are corrected rather than annotated.
