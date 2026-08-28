# A1 — Where the agent's 88% goes

Run 2026-08-28. SIGIR, 20 patients, top-10, prompt v1, DeepInfra. Replayed from
the analyst cache through the real node order (`attach_kinds` →
`ground_assessments` → `rollup_trial_verdict`), so grounding is applied exactly
as the served path applies it. No LLM calls; cost $0.

**Sample: 200 (patient, trial) pairs, of which 25 are gold-eligible.** Every
number below rests on those 25. Treat magnitudes as indicative and directions as
solid; this needs a TREC replication before it goes in a paper.

## The agent discriminates. It is not guessing.

P(trial verdict | gold label):

| gold label | n | excluded | cannot_determine | eligible |
|---|---|---|---|---|
| eligible | 25 | 24.0% | 64.0% | **12.0%** |
| excluded | 38 | 31.6% | 68.4% | 0.0% |
| irrelevant | 50 | 58.0% | 42.0% | 0.0% |
| unlabelled | 87 | 50.6% | 47.1% | 2.3% |

"Eligible" is emitted **only** on gold-eligible trials and on unlabelled ones —
never once on a trial the cohort marks `excluded` or `irrelevant`. Exclusion
rate more than doubles from gold-eligible (24.0%) to irrelevant (58.0%). The
ranking signal is real; what is missing is recall, not discrimination.

## The loss is abstention, not false exclusion

Of 25 gold-eligible trials, 22 were not called eligible:

| | n | shape |
|---|---|---|
| `cannot_determine` | **16** | 1–8 unresolved criteria, median 3 |
| `excluded` | 6 | 1–4 killers, median 1; **4 of 6 killed by a single criterion** |

Why criteria are unresolved, across all gold-eligible pairs:

| cause | n |
|---|---|
| model abstained — fact absent from the note | **85** |
| grounding failure — claimed, could not quote | 3 |

96.6% of unresolved criteria are honest abstention. The verifier is not
rejecting the model's work; the patient note simply does not contain the fact
the criterion asks about. This is the system behaving exactly as designed, and
the roll-up converting correct uncertainty into a non-answer.

## Roll-up sensitivity

End-to-end recall against the 144 gold-eligible trials in the cohort
(retrieval ceiling 0.1736):

| roll-up rule | end-to-end recall | % of ceiling |
|---|---|---|
| as shipped (every criterion `met`, zero unresolved) | 0.0208 | 12% |
| tolerate ≤1 unresolved, no killer | 0.0556 | 32% |
| tolerate ≤2 unresolved, no killer | 0.0694 | 40% |
| tolerate ≤3 unresolved, no killer | 0.0833 | 48% |
| **no killer criterion (`cannot_determine` counts as surfaced)** | **0.1319** | **76%** |

Tolerating a fixed small number of unknowns recovers little, because the
unresolved count is spread (median 3, max 8). The step change comes from
dropping the completeness requirement entirely.

## Secondary defect: disjunctive criteria are ANDed

`NCT01540318` lists trauma mechanisms — "Falls greater than 20 feet", "Physical
assault involving the abdomen", "Blunt traumatic event with any of the
following" — under a parent that says *any of*. The parser splits them into
sibling criteria and the roll-up ANDs them, so a patient who arrives by car
accident fails three of four and the trial is excluded. The model answered every
one correctly.

Prevalence: **7.6%** of SIGIR trials and **9.3%** of TREC 2021 trials contain a
disjunctive header. Real, worth fixing, and not the main story.

The remaining sampled killers are model overreach on inference, e.g. exclusion
"known or suspected fractures of the femur or pelvis" marked met by quoting
"upper and lower extremity fractures".

## What this means

The binding constraint is not prompt quality, not hallucination (0.87%
unverifiable), and not retrieval. It is that **trial-level roll-up demands
complete information a short clinical note cannot supply.** TrialGPT's ≥87%
target is criterion-level matching accuracy; this project's headline is
trial-level all-or-nothing. They are different tasks, and the second is much
harder for reasons that have nothing to do with faithfulness.

Two consequences:

1. A trial with no disqualifier and three unknown criteria is clinically worth
   surfacing, flagged. Reporting a binary verdict discards that. The product
   likely wants a ranked output with an explicit unknown count.
2. Criterion-level accuracy — the metric that matches the benchmark and the
   thesis — has never been measured end-to-end. It should be, before any further
   prompt work.
