# V1: intervals on the headlines, and what they caught

Run 2026-09-28. $0: every number here comes from a cached-only rerun
(coverage 1.0000) or from committed JSON. No LLM calls.

`end_to_end` now keeps per-patient counts and reports a patient-bootstrap 95% CI
(10,000 resamples, seed 0) on retrieval recall, end-to-end recall, surfaced
recall and precision, pool base rate, and lift. Lift used to be computed by hand
per report; it now comes out of the harness. `eval/significance.py` gained
`bootstrap_ci`, `paired_bootstrap_ci`, `wilcoxon_paired`, `bh_adjust`,
`compare_family` and `min_detectable_effect`. Wilcoxon + Benjamini-Hochberg is
the procedure Otero, Parapar & Barreiro (ECIR 2025, arXiv:2501.03930) recommend
for multiple comparisons at TREC topic counts.

Reproduce:

```
TG_KEYWORD_DECAY=0 TG_GROUND_SYMBOLS=0 TG_ABSENCE_ACRONYMS=0 TG_STRICT_CRITERIA=0 \
TG_DEDUP_ANSWERS=0 TG_RETRY_KEEP_GROUNDED=0 TG_RETRY_MISSING=0 TG_PROMPT_VERSION=v4 \
  python -m trialguard.eval.end_to_end --cohort trec_2021 --n-patients 20 --top-k 100 --cached-only
# trec_2022 additionally needs TG_INDEX_EXCLUSION=0 (only noexcl is cached)
TG_PROMPT_VERSION=v4 python -m trialguard.eval.end_to_end --cohort trec_2021 --n-patients 0 --top-k 10 --cached-only
python scripts/v1_retro_ci.py
```

The published H1 table needs **every** post-2026-09-01 flag off. With current
defaults the analyst cache hit rate is 0.10, because `TG_STRICT_CRITERIA` changes
the cache key and `TG_KEYWORD_DECAY` changes which trials are retrieved. Raw:
`v1_retro_*.json`, `v1_retro_ci.json`.

## 1. The published end-to-end table, with intervals

n=20 per row, the same patients as H1/E1b.

| cohort | pool | surfaced recall | 95% CI | base rate | lift | 95% CI |
|---|---|---|---|---|---|---|
| TREC 2021 | top-10 | 0.0270 | [0.017, 0.040] | 0.429 | 1.59x | [1.35, 1.96] |
| TREC 2021 | top-100 | **0.1770** | [0.143, 0.216] | 0.345 | 1.62x | [1.48, 1.82] |
| TREC 2022 | top-10 | 0.0424 | [0.032, 0.060] | 0.547 | 1.27x | [1.12, 1.48] |
| TREC 2022 | top-100 | **0.2580** | [0.202, 0.335] | 0.369 | 1.45x | [1.31, 1.62] |

Every published point estimate reproduces within rounding. Surfaced recall and
base rate match exactly. Precision at TREC 2021 top-100 is 0.5604 against a
published 0.5546; grounding code has changed since in ways that no flag gates.

**The 10→100 widening survives correction.** It holds for 20 of 20 patients on
both cohorts. Mean per-patient delta is +0.181 [0.139, 0.232] on 2021 and +0.280
[0.216, 0.348] on 2022, with BH-adjusted p ≤ 0.0001. This is the strongest result
in the repo, and n=20 was enough for it.

## 2. E3's lift divided mismatched denominators

E3 (`e3_findings.md`) computed tier precision over **labelled** trials, which is
correct: an unjudged trial is a pool gap, not a false positive. But it computed
the base rate over **all** assessed trials, unlabelled included. Unlabelled
trials lower the base rate and leave precision alone, so the ratio inflates.
H1 and E1b used labelled trials on both sides and are unaffected.

Rerunning E3's exact cells (all 75 TREC 2021 patients, top-10, current defaults,
coverage 1.0) reproduces its published figures when the mixed denominator is used
and corrects them when it isn't:

| group | n | E3 published | mixed denominator (reproduced) | **consistent** | 95% CI |
|---|---|---|---|---|---|
| all | 75 | 1.65x | 1.69x | **1.37x** | [1.26, 1.49] |
| oncology | 14 | 1.90x | 1.90x | **1.51x** | [1.22, 1.99] |
| non-oncology | 61 | 1.61x | 1.65x | **1.34x** | [1.23, 1.46] |

(The small all/non-oncology gap between published and reproduced comes from
grounding changes since 2026-09-23. The oncology cell reproduces to four
figures.)

What changes:
- **"The agent is better inside oncology" is not supported on lift.** 1.51x vs
  1.34x, with overlapping intervals. E3's faithfulness split (unverifiable
  0.0138 vs 0.0355) is a separate metric and is not affected by this error.
- **"The lift holds in both cells above the published 1.60x" is false.** Both
  cells are below it.

## 3. The n=20 headline came from favourable patients

Same config, same cached verdicts, top-10, split by patient:

| patients | n | lift | 95% CI |
|---|---|---|---|
| the H1/E1b 20 | 20 | 1.56x | [1.34, 1.88] |
| the other 55 | 55 | **1.30x** | [1.19, 1.43] |
| all | 75 | 1.37x | [1.26, 1.49] |
| the H1/E1b 20, old flags | 20 | 1.59x | [1.35, 1.96] |

The flag changes move lift by 0.03. **Patient selection moves it by 0.26.** The
first 20 TREC 2021 topics are easier for the tier than the rest, and every
end-to-end headline was measured on them. The system still beats its pool's base
rate on every slice (no CI touches 1.0), so the tiered contract holds. Its size
at top-10 on the full cohort is about **1.37x, not 1.60x**.

Top-100 hasn't been measured on the full cohort yet. That's what V2 is for.

## 4. Power

Per-patient deltas from the widening give the scale of effect detectable at
α=0.05 and power 0.8 (normal approximation to the paired t-test):

| cohort | sd of per-patient Δ surfaced recall | MDE at n=20 | MDE at full n |
|---|---|---|---|
| TREC 2021 | 0.110 | 0.069 | **0.036** (n=75) |
| TREC 2022 | 0.154 | 0.096 | **0.061** (n=50) |

So at n=20, no intervention smaller than ~7 points of surfaced recall could have
been detected on TREC 2021. That's larger than most effects this repo has tried
to measure. At full n it roughly halves. The R-workstream adopt bar (+0.05
recall@100) is detectable on 2021 at full n and only marginally on 2022, so R
items should report both cohorts pooled as well as separately.

## Standing

- `end_to_end` reports `pool_base_rate`, `surfaced_lift`, `ci95` and
  `per_patient` on every run. Lift uses labelled trials on both sides.
- `--n-patients 0` scores the full cohort.
- README and CLAUDE.md lift claims are amended to carry intervals and the E3
  correction.
- AD-34 records the rule: every headline gets its CI and n, and multi-arm
  experiments report BH-adjusted p within their family.
