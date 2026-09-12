# Truncation blocks `eligible`: a guarantee, not an improvement

Run 2026-09-12. Both cohorts, 20 patients x top-10, on top of the strict parser
and containment matching.

## What changed

`rollup_trial` now takes `truncated` and refuses `eligible` when the criteria
list was cut to fit `MAX_CRITERIA`. Before this, the mitigation for the cap was
a muted note in the UI beside a verdict the note contradicts: *criteria list
truncated, roll-up may be incomplete* under a badge reading **Eligible**.
CLAUDE.md has named that unsound since the cap was introduced.

**Only `eligible` is blocked, and that asymmetry is the whole design.** A
disqualifier that was found is still found, and no criterion the cap dropped can
un-find it, so `excluded` stands over a truncated list. `eligible` is the only
verdict that quantifies over *every* criterion, so it is the only one a cut list
cannot support. `cannot_determine` for unstated facts and `cannot_determine`
because criteria were dropped are different states, so `truncated_block`
distinguishes them and the UI explains the second in those terms.

## Measured effect: none, on these cohorts

| | before | after |
|---|---|---|
| TREC trial verdicts | 74 cd / 7 elig / 118 excl | **identical** |
| TREC surfaced | prec 0.6486, recall 0.0316 | **identical** |
| SIGIR trial verdicts | 87 cd / 4 elig / 109 excl | **identical** |
| SIGIR surfaced | prec 0.3621, recall 0.1458 | **identical** |

**The block fired zero times across ~400 assessed trials.** That is not a null
result to be buried: it is the honest description of what this change buys.

The mechanism is arithmetic. `eligible` already requires *zero* unresolved
criteria, and TREC averages 3.3 `cannot_determine` per assessed trial. A
truncated trial has at least 24 criteria, so the chance that every one of them
resolves cleanly is very small -- `unknown` catches those trials long before
truncation would. The two conditions almost never co-occur on this data.

## Why it ships anyway

**29.4% of the live corpus truncates** (1,177 of 4,000 sampled). The eval
cohorts are a biased sample of that: their trials were selected for having gold
labels, not for being long. A user assessing an arbitrary recruiting trial meets
the truncated case roughly three times in ten, and the previous behaviour would
have let a clean-looking run produce `eligible` over a list that was silently
cut.

This is a soundness guarantee closing a hole the current data does not happen to
walk through, in the same category as the deterministic grounding check: its
value is that it cannot be wrong, not that it improved a number. Shipping a
guarantee whose measured effect is zero is the correct call when the alternative
is a claim the system cannot support; shipping it while *implying* it improved
something would not be.

## Caveats

- Zero of ~400 trials is consistent with a true rate anywhere up to about 0.7%
  on these cohorts. It is not evidence that the case never occurs.
- Untested against a trial that truncates *and* resolves every remaining
  criterion, because none appeared. The unit tests construct that case directly
  rather than waiting for the cohorts to produce one.
- The regression gate is unaffected and still passes: it reads criterion-level
  faithfulness, which this does not touch.
