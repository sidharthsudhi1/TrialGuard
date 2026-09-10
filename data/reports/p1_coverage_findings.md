# P1 — The analyst skips criteria. Two prompts failed; the retry edge fixed it.

Run 2026-09-10. DeepInfra, Llama-3.3-70B-Instruct-Turbo, prompt v4, 20 patients
× top-10 on both cohorts. First attempts are cache hits in both arms, so the only
fresh calls are retries: **$0.118** total. Raw: `p1_off_trec2021.json`,
`p1_on_trec2021.json`, `p1_off_sigir.json`, `p1_on_sigir.json`.

## The defect

Under v4 the analyst returns fewer assessment objects than it was handed
criteria: **229 of 1,761 on TREC 2021 (13.0%)**, 12 of 1,814 on SIGIR. Not
truncation -- `max_tokens` is 4096 against a 1,732-token longest response.

It is invisible by construction. A criterion with no assessment cannot fail
grounding, so it never reaches the retry edge; it does not appear in
`n_unknown`, because the roll-up only sees what it was given; and
`rollup_trial` still calls a trial `eligible` when every criterion it *received*
was met. CLAUDE.md already names that unsound. Nothing surfaced it until
`criterion_asked` was added in Phase 10.

Two prompt fixes were measured and rejected first: v5's numbering works but costs
26% of TREC surfaced recall (AD-16), and v6's instruction alone does nothing
(AD-17). The prompt cannot be talked into completeness, so this is the mechanism
fix -- ask again for what is missing, on the edge that already re-asks for named
criteria after a grounding failure.

## Result

| | TREC off | TREC on | SIGIR off | SIGIR on |
|---|---|---|---|---|
| criteria never answered | **229 (13.0%)** | **71 (4.0%)** | 12 (0.7%) | 9 (0.5%) |
| grounded | 840 | **916** | 807 | **820** |
| grounded / asked | 0.4770 | **0.5202** | 0.4449 | **0.4520** |
| met | 354 | 368 | 245 | 242 |
| not_met | 478 | 545 | 553 | 569 |
| unverifiable | 55 | 88 | 42 | 44 |
| unverifiable rate | 0.0359 | 0.0521 | 0.0233 | 0.0244 |
| unresolved / asked | 0.5275 | **0.4912** | 0.5601 | **0.5533** |
| trials eligible | 8 | 5 | 5 | 6 |
| `eligible` precision | 0.625 | **0.800** | 0.800 | 0.667 |
| surfaced precision | 0.6986 | **0.7097** | 0.3390 | **0.3509** |
| surfaced recall | 0.0336 | **0.0289** | 0.1389 | 0.1389 |

**This is not L4.** Grounded rises on both cohorts and unresolved falls on both.
L4's tell was the opposite -- the faithfulness proxy improving because the system
stopped answering.

**SIGIR is the control.** Only 12 criteria were being skipped there, so the fix
has almost nothing to do, and it does almost nothing: surfaced recall identical
to four decimal places, precision slightly up. A change that only moves the
cohort with the defect is behaving like a fix rather than like a side effect.

## The cost, stated plainly

TREC surfaced recall falls 14% (0.0336 → 0.0289, 79 trials shown → 68). Answering
the skipped criteria finds more disqualifiers, and gold says some of those
exclusions are wrong.

Adopted anyway, and the reason is not that 14% is small. It is that the recall it
removes was produced by not looking. v4's extra surfaced trials include trials
called `eligible` over a criteria list that was silently 13% short, which the
repo's own documentation calls unsound. The soundness gain is visible in the same
table: `eligible` precision rises 0.625 → 0.800 on TREC, and of the three
`eligible` calls lost, two were wrong.

`TG_RETRY_MISSING=0` restores the previous behaviour and is what reproduces retry
numbers committed before 2026-09-10, the same role `TG_KEYWORD_DECAY=0` plays for
rankings. It has to exist: this changes the retry prompt and therefore the cache
key of every retry entry.

## What it did not fix

71 criteria on TREC are still never answered. The retry is bounded at 2 by
AD-policy and the model still skips on the re-ask. 4.0% is not 0.

## Caveats

- One model, two cohorts, 20 patients each. The trial-level deltas rest on a
  handful of trials; the criterion-level ones on 1,761 and 1,814.
- `criterion_unanswered` counts criteria with no assessment after merging, so a
  criterion answered on retry counts as answered even if its verdict is weak.
- The retry costs a second full call on trials that came back short: $0.099 on
  TREC, $0.019 on SIGIR, against $0 for the cached first attempts.
