# Parser plus matching: the coverage shortfall was ~99% ours

Run 2026-09-12. Both fixes measured together end to end for the first time,
which needed a fresh analyst cache because the strict parser moves the
namespace. 444 calls, **$0.139**. (`run_usd` reads 0.0 in
`combined_trec_2021.json`: the daily ledger rolled over mid-run and the delta
went negative. The per-day totals above are the real figures.)

## Result

| TREC 2021 | baseline | all fixes |
|---|---|---|
| criteria asked | 1761 | 1560 |
| **never answered** | **230 (13.1%)** | **1 (0.06%)** |
| answered but never asked | 45 | **11** |
| grounded | 848 | **867** |
| grounded / asked | 0.4815 | **0.5558** |
| unverifiable rate | 0.0370 | 0.0414 |
| weak absence | 129 | **121** |
| surfaced precision | 0.6944 | 0.6486 |
| surfaced recall | 0.0329 | 0.0316 |
| end-to-end recall | 0.0026 | 0.0026 |

| SIGIR | baseline | all fixes |
|---|---|---|
| **never answered** | **43 (2.4%)** | **2 (0.1%)** |
| answered but never asked | 40 | **33** |
| grounded | 814 | **833** |
| grounded / asked | 0.4487 | **0.4630** |
| unverifiable rate | 0.0215 | **0.0180** |
| weak absence | 90 | **84** |
| surfaced precision | 0.3390 | **0.3621** |
| surfaced recall | 0.1389 | **0.1458** |

SIGIR improves on every axis measured. TREC improves on every criterion-level
axis and gives back a little at trial level.

## The retry's recorded cost was an artifact, and AD-18 is corrected again

Control run, same fixes, retry off:

| TREC, strict parser + alignment | retry off | retry on |
|---|---|---|
| never answered | 14 | **1** |
| grounded | 850 | **867** |
| surfaced precision | 0.6389 | **0.6486** |
| surfaced recall | 0.0303 | **0.0316** |

AD-18 recorded the retry as costing 12–14% of TREC surfaced recall and adopted
it anyway on soundness grounds. **With the parser fixed that cost disappears
and reverses**: the retry now improves surfaced recall and precision. The
earlier penalty was the retry dutifully re-asking about section headers and
placeholders and manufacturing noise from them.

## Two of my own estimates were too high

- **"3.3% genuine omission."** The control puts it at **0.9%** (14 of 1,560)
  before any retry. My decomposition used a cruder containment test than
  `align_assessments` -- no one-to-one constraint, no longest-first -- and a
  length heuristic for headers, so it attributed to the model what better
  matching resolves.
- **"The retry costs surfaced recall."** True only in the presence of the
  parser defect.

The corrected headline: of TREC's 13.1% shortfall, roughly **99% was the
parser and the matcher**, not the analyst.

## What got worse, stated rather than buried

TREC surfaced precision 0.6944 → 0.6486 and recall 0.0329 → 0.0316; lift over
the 0.429 pool base rate 1.62x → 1.51x. SIGIR moves the other way, 1.53x →
1.64x.

Not read as a real regression, and not dismissed either. TREC surfaced
precision has ranged 0.6267–0.7097 across configurations of this same system
during this investigation, so a 0.046 move sits inside the observed spread on
20 patients. Saying which way it really went needs more patients, not more
interpretation of these ones.

## Caveat

One run per configuration, 20 patients, top-10. The criterion-level numbers
rest on ~1,600 criteria and are solid; the trial-level ones rest on ~200 trials
and are not.
