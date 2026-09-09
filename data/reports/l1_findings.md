# L1 / WS-6c — Index-addressed prompt (v5): measured, not adopted, one real defect found

Run 2026-09-07. DeepInfra, Llama-3.3-70B-Instruct-Turbo. Two instruments, because
the claim has two halves that need different samples.

- **Cost:** 30 trials per cohort, both arms called fresh and paired inside one
  worker so provider drift hits both equally. `l1_prompt_cost_sigir.json`,
  `l1_prompt_cost_trec_2021.json`.
- **Quality:** the full 200-pair end-to-end set per cohort, 20 patients × top-10.
  `l1_v4_sigir.json`, `l1_v5_sigir.json`, `l1_v4_trec2021.json`,
  `l1_v5_trec2021.json`. Total spend for everything: **$0.23**.

## What was proposed

`pipeline_review.md` L1: ~29 s per trial is output-bound at ~16 tok/s, and 43.4%
of analyst output is the model retyping criteria it was just handed against 7.8%
that is the quote the verifier reads. Number the criteria, ask for the number
back, and output should roughly halve.

L4 is the cautionary precedent — it looked like pure efficiency, improved the
faithfulness proxy 26%, and made the system strictly worse — so L1 was required to
report verdict distribution and grounded counts, not a faithfulness floor.

## Half one: the efficiency premise did not hold

| paired, n=30 | SIGIR v4 → v5 | TREC 2021 v4 → v5 |
|---|---|---|
| median output tokens | 465 → **322** (0.69x) | 555.5 → **492** (0.89x) |
| median seconds | 21.82 → **14.92** (0.68x) | 71.97 → **74.27** (**1.03x**) |
| median input tokens | 683.5 → 747.5 | 924.5 → 989.0 |
| output tokens per criterion answered | 69.9 → **54.2** (0.78x) | 75.3 → **56.0** (0.74x) |
| objects returned / criteria asked | 239 / 238 → 234 / 238 | **309 / 333** → **332 / 333** |
| index errors | 0 | 2 out of range (0.6%) |

Per criterion the echo saving is real and consistent at 22–26%, not the ~50% the
43.4% echo share suggested: v5 still emits the quote, the rationale, the verdict
and the JSON scaffolding, and its own system prompt is 64 input tokens longer.

**On TREC, v5 is 3% slower per call** — because it answers 7% more criteria (see
below). Per criterion it is faster on both cohorts. The stated motivation for L1
was latency, and latency improved on one cohort of two.

Wall clock in the quality runs is **not** comparable between arms and is not
quoted here: the v4 arms were largely cache hits from earlier phases while every
v5 call was fresh. The paired harness exists because that comparison is otherwise
unavailable.

## Half two: v4 silently answers 13% fewer criteria than it is asked

This is the finding, and it is about v4, not v5.

| criteria the analyst never answered | SIGIR | TREC 2021 |
|---|---|---|
| v4 | 12 / 1814 (0.7%) | **229 / 1761 (13.0%)** |
| v5 | 2 / 1814 (0.1%) | 8 / 1761 (0.5%) |

Not truncation. `max_tokens` is 4096 and the longest response in the paired run
was 1,732 tokens, so nothing was cut off. The model is handed 24 criteria and
returns 21. Nothing surfaced it, because a criterion that is never answered
produces no assessment, so it cannot fail grounding, cannot trigger the retry
edge, and simply disappears into `needs_review`.

It is worse than a missing row. `rollup_trial` calls a trial `eligible` only when
every criterion is met — over the criteria it received. CLAUDE.md already states
that "'eligible only if all met' over a silently truncated list is unsound", and
that is exactly what was happening on TREC 13% of the time. Part of v4's higher
`eligible` count is verdicts reached over an incomplete list.

Numbering the criteria closes it almost entirely. That is a coverage effect of
addressing, not of the token saving.

## Half two, continued: the criterion-level table

Rates over **criteria asked**, which is the only denominator that compares the
arms honestly — v5 answers more, so any rate over "criteria returned" flatters
whichever arm answered less.

| | SIGIR v4 | SIGIR v5 | TREC v4 | TREC v5 |
|---|---|---|---|---|
| decisive (met + not_met) | 798 (0.4399) | **859 (0.4735)** | 832 (0.4725) | **896 (0.5088)** |
| unresolved (cd + unverifiable + unanswered) | 1016 (0.5601) | **955 (0.5265)** | 929 (0.5275) | **865 (0.4912)** |
| **grounded** | 807 (0.4449) | **875 (0.4824)** | 840 (0.4770) | **897 (0.5094)** |
| met | 245 | 266 | 354 | 387 |
| not_met | 553 | 593 | 478 | 509 |
| cannot_determine | 962 | 927 | 645 | 782 |
| unverifiable | 42 | **26** | 55 | **75** |
| self-referential quotes | 4 | **17** | 1 | **20** |

**This is not L4.** L4's tell was that grounded *fell* while abstention rose — the
faithfulness metric improving because the system stopped answering. Here grounded
rises on both cohorts, decisive rises, and unresolved falls. v5 trades abstention
for assertion, which is the direction H1 said the product needs.

The stated acceptance — *verdict distribution and grounded counts non-inferior on
both cohorts* — is therefore met, and exceeded.

## Why it is still not adopted

Three things the acceptance criterion did not ask about, all of which it costs.

**1. Surfaced recall falls 26% on TREC.**

| | SIGIR v4 → v5 | TREC v4 → v5 |
|---|---|---|
| surfaced recall | 0.1389 → **0.1528** | 0.0336 → **0.0250** |
| surfaced precision | 0.3390 → **0.4000** | 0.6986 → **0.7308** |
| lift over pool base rate | 1.53x → **1.81x** | 1.63x → **1.70x** |
| trials shown | 91 → 80 | 79 → **56** |
| trial verdicts (elig / cd / excl) | 5 / 86 / 109 → 6 / 74 / 120 | 8 / 71 / 121 → **3 / 53 / 144** |
| end-to-end recall | 0.0278 → 0.0278 | 0.0033 → **0.0020** |

Answering the 221 criteria v4 skipped finds more disqualifiers, so more trials are
excluded — 121 → 144 on TREC. Precision and tier lift improve because many of
those exclusions are right. But some are wrong, and they remove gold-eligible
trials from the surfaced set. Recall is the binding constraint this system is
scored on (H1), and a 26% cut to it on one cohort is not a rounding error.

**2. Self-referential quotes rise 4x on SIGIR and 20x on TREC.** Small in absolute
terms — 1.9% and 2.2% of grounded — but the direction is wrong and the mechanism is
plausibly caused by the design: with the criterion no longer echoed into its own
field, the model appears more willing to put criterion text into the *quote* field
instead. WS-5b just measured the entailment gap at 36%; a change that widens the
one machine-visible slice of it needs a better reason than a 3% latency win.

**3. The reason to take the risk did not materialise.** L1 was ranked low when it
was a cost argument and revived as a latency argument. Latency improved 32% on
SIGIR and −3% on TREC. That is not the case for changing the served prompt.

Adopting v5 would also switch the served path's prompt version, cold-starting the
analyst cache namespace — a real cost paid for a benefit that is cohort-split.

## Verdict

**Not adopted.** v5 stays behind `TG_PROMPT_VERSION=v5`, registered and unfrozen,
with v1–v4 untouched — the same treatment R4's reranker and L4's partial retry
got. The measurement is reproducible and the default path is unchanged.

**The coverage defect it exposed is the higher-value follow-up**, and it should be
attacked on its own rather than smuggled in with an addressing change. The clean
next experiment is a v6 that keeps v4's echoed-text addressing and adds only the
"return exactly one object per criterion, every criterion" instruction v5 carries.
That separates *coverage* from *addressing* and would say whether the 13% is fixed
by numbering or merely by being told. If it is the instruction, the fix is one
sentence with none of v5's costs.

## Caveats

- One model. The echo share, and therefore the saving, is a property of
  Llama-3.3-70B's output habits.
- n=30 paired calls per cohort for the cost half; medians, not distributions. Raw
  per-call rows are kept in the reports for exactly that reason.
- The quality arms differ in cache state, so `assess_s` is not a latency figure
  and is not quoted as one.
- TREC's tok/s during the paired run was 7.95 against SIGIR's 19.55. Both arms saw
  it equally, which is what pairing is for, but the absolute seconds are not
  comparable across cohorts.
- Two out-of-range indices on TREC (0.6% of returned objects) were dropped rather
  than clamped, so they became unanswered criteria rather than verdicts filed
  under the wrong criterion. Clamping would have been undetectable downstream: the
  quote grounds against the trial's whole text either way.
