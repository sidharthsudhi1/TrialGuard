# P1 correction — the coverage numbers in AD-18 were computed with a broken metric

Run 2026-09-11. Supersedes the TREC figures in `p1_coverage_findings.md` and
AD-18. The fix is still a fix; it is smaller than reported, and it has a cost
that was invisible.

## What was wrong

`criterion_unanswered` was computed as `criteria_asked - criteria_answered`.
That is not the number of criteria the analyst skipped. The model also returns
assessments naming criteria that were **never asked** -- rephrased beyond
recognition, or invented -- and each one cancels a genuinely skipped criterion
out of the subtraction.

It surfaced as a negative count on SIGIR (`criterion_unanswered: -6`), which is
impossible and is what exposed it. The metric now counts asked criteria with no
assessment directly, and reports `criterion_unmatched` beside it.

The subtraction flattered the retry arm specifically, because a retry produces
more output and therefore more drift: unmatched entries nearly double on TREC
when the retry is on. The arm being argued for was the arm the error favoured.

## Corrected numbers, both arms re-measured

| TREC 2021 | off | on |
|---|---|---|
| criteria asked | 1761 | 1761 |
| **never answered** | **230 (13.1%)** | **167 (9.5%)** |
| answered but never asked | 45 | **81** |
| grounded | 848 | **918** |
| unverifiable rate | 0.0370 | 0.0448 |
| surfaced recall | 0.0329 | 0.0289 |
| surfaced precision | 0.6944 | 0.6567 |
| `eligible` precision | 0.5714 | **0.7143** |
| end-to-end recall | 0.0026 | **0.0033** |

| SIGIR | off | on |
|---|---|---|
| **never answered** | **43 (2.4%)** | **14 (0.8%)** |
| answered but never asked | 40 | **20** |
| grounded | 814 | **824** |
| surfaced recall | 0.1389 | 0.1389 |
| surfaced precision | 0.3390 | **0.3448** |

As previously claimed: **229 → 71, 13.0% → 4.0%**. Actually: **230 → 167,
13.1% → 9.5%**. A 27% reduction, not 69%.

## What is still true

- Fewer criteria are skipped on both cohorts.
- More criteria end grounded on both cohorts.
- End-to-end recall on TREC *rises* (0.0026 → 0.0033), which the earlier
  measurement had falling.
- `eligible` precision rises on TREC, 0.5714 → 0.7143.

## What is newly visible

**Unmatched entries nearly double on TREC, 45 → 81.** The retry answers more,
and some of what it answers corresponds to no criterion it was given. These
reach `attach_kinds` as `unknown` and the roll-up counts them unresolved, so
they do not fabricate verdicts -- but they are noise the first attempt did not
produce, and they were previously hidden inside the same subtraction.

## The remaining shortfall

9.5% on TREC, not the 4.0% reported. Three attempts to close it further were
measured and rejected on the way here:

- **Narrow the re-ask to just the skipped criteria.** 144 unanswered against 71
  under the old metric; worse. A narrowed ask has to be merged back by text
  match and what the merge cannot place is lost.
- **Remove the dangling "these criteria need a verbatim quote" header when
  nothing failed grounding.** 269 unanswered; worse still, and the reason was
  not the prompt.
- Those two runs disagreeing so wildly (71 / 144 / 269 for one configuration)
  is what exposed the real defect: **the full-list retry replaced attempt one
  wholesale**, so a retry that happened to answer less made coverage worse than
  not retrying. That is now an invariant -- `_backfill` guarantees a retry can
  only add -- and it is the part of this work that is unambiguously correct
  regardless of the aggregate numbers.

## Caveat

The three intermediate runs were measured with the broken metric, so their
absolute numbers are not comparable to the table above. They are quoted here
only for the spread that revealed the wholesale-replacement bug, which is a
property of their disagreement rather than of any one value.
