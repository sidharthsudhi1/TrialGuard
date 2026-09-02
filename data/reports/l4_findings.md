# L4 — Retry only the failed criteria: measured, rejected

Run 2026-09-02. TREC 2021, 20 patients, top-10, prompt v4, DeepInfra, matched
arms differing only in `TG_RETRY_FAILED_ONLY`. First attempts are cache hits in
both arms, so the only fresh calls are retries: $0.0043 total. Raw:
`l4_off_trec2021.json`, `l4_on_trec2021.json`.

## What was proposed

`pipeline_review.md` L4: a grounding failure re-assesses the whole trial at
~29 s, so "retry only the failed criteria rather than the full set. Median
trials have 6 criteria and typically few fail, so the retry call should be a
fraction of the original." Verification condition: *grounding-recovery rate must
not regress on TREC*.

It reads as a pure efficiency change — same answers, fewer tokens — which is
exactly why it needed measuring.

## Result

| | flag off | flag on |
|---|---|---|
| criterion unverifiable rate | 0.0395 | **0.0291** |
| surfaced recall | 0.0276 | 0.0296 |
| surfaced precision | 0.6885 | **0.5233** |
| lift over pool base rate (0.4294) | **1.60x** | **1.22x** |
| trials surfaced | 69 | 96 |
| trial verdicts (elig / cd / excl) | 7 / 62 / 131 | 7 / 89 / 104 |

The verifier metric improves by 26% and the tier contract loses a quarter of its
lift. Twenty-seven trials move from `excluded` to `cannot_determine`; the
`eligible` tier is untouched.

## The unverifiable drop is not recovery

Criterion-level counts over the same 200 trials tell the real story:

| | flag off | flag on | change |
|---|---|---|---|
| unverifiable | 60 (3.9%) | 45 (2.9%) | **−15** |
| cannot_determine | 619 (40.8%) | 668 (43.2%) | **+49** |
| not_met | 468 (30.8%) | 457 (29.6%) | −11 |
| met | 372 (24.5%) | 376 (24.3%) | +4 |
| **grounded** | **845 (55.6%)** | **838 (54.2%)** | **−7** |

If the partial retry were recovering quotes, `grounded` would rise and
`unverifiable` would fall into `met`/`not_met`. Instead `grounded` *falls* and
the vanished failures reappear as abstentions. Nothing was recovered: asked to
re-answer one isolated criterion under an explicit "this needs a verbatim quote
that was not found last time", the model stops asserting and returns
`cannot_determine`, which requires no quote and is not counted as a grounding
failure. The 11 lost `not_met` verdicts are the disqualifiers that were holding
those 27 trials out of the surfaced set.

**This is abstention laundering.** The faithfulness proxy improves precisely
because the system stopped answering. It is the failure mode the coverage
metric exists to catch, and it is why coverage and citation precision are read
jointly in this repo rather than either alone.

## Verdict

Rejected. It fails its own stated verification condition — grounding recovery
regresses, in the strict sense that fewer criteria end grounded — and it costs
0.38x of tier lift to save tokens on a path that fires on under 4% of criteria.
The efficiency premise was sound and the saving is real (retry calls carry 1–3
criteria instead of a median 6), but cost is not this system's binding
constraint, and H1 established that what the product needs is *more* decisive
assessment at depth, not less.

Code kept behind `TG_RETRY_FAILED_ONLY`, default off, exactly as R4's reranker
was kept: the measurement is reproducible and the default path is untouched.
The flag also had to exist for the A/B at all — changing the retry prompt
changes the analyst cache key, so shipping it unflagged would have silently
orphaned every committed v1–v4 retry entry.

## Caveats

- One cohort, n=20, 200 trials. The mechanism (abstention replacing assertion)
  is visible at criterion level with n=1,519, which is where the confidence
  comes from; the trial-level deltas rest on 27 trials.
- Not tested: a partial-retry prompt that forbids `cannot_determine` on
  re-ask. That might preserve decisiveness, but it would be pressuring the
  model to assert under exactly the conditions where it could not find
  evidence — the opposite of this system's thesis.
- `n_criteria` differs slightly between arms (1,519 vs 1,546) because the merge
  restores the full criteria list where an unflagged retry had returned fewer.
