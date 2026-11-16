# R4: the 8B analyst drops gold-eligible trials

Run 2026-11-16. Llama-3.1-8B-Instruct-Turbo, prompt v4, full analyst and
grounding, on the cached deep-fusion tail. TREC 2021 uses the excl pool and
TREC 2022 the noexcl pool, which is how `data/cache/ltr` was built. The slice
is ranks 101 to 136, the deepest equal depth that fit in $2.70 at 1.5 times
the calibration call ($0.000392). 125 patients, 4,500 tail trials. Spend
$0.674 on an isolated ledger. Raw: `r4_cascade.json`, calls in `r4_calls.jsonl`.

Five trials had no parsed criteria and were not assessed. None of them is
gold-eligible, so the eligible denominator is complete.
`r4_cascade.json` records `complete: false` and `decision: incomplete` for
that reason only: the script counts a trial with no tier as unfinished.

## Drop rate

A drop is an 8B tier of `excluded` on a gold-eligible trial in the slice.
The bar is 5%.

| cohort | gold-eligible assessed | called excluded | rate |
|---|---|---|---|
| TREC 2021 | 435 | 171 | 0.3931 |
| TREC 2022 | 326 | 116 | 0.3558 |
| both | 761 | 287 | 0.3771 |

Both cohorts are several times the bar. Counting the five unparsed trials as
not dropped would not move the rate, because none of them is gold-eligible.

The 70B phase did not run. The kill stops the cascade once the small model
drops more than 5% of gold-eligible trials, and there was no gain left to buy.

## Decision

DONE-REJECTED. The served path stays a 70B assessment of the head. The 8B
model is not used to screen the tail.
