# R5 — sweeping the RRF constant and pool depth

Run 2026-08-31. 40 cells (`k` in 1..200, pool in 10..200) on both cohorts.
Retrieval only, no LLM calls, keywords served from cache. SIGIR local, TREC on a
short-lived EC2 box. Cost: ~$0.20 of compute, $0 of inference.

Method: rankings are computed once per patient at pool=200 and truncated per
cell. Each ranking is already sorted, so a truncation to depth d is exactly what
pool=d would have produced — the whole grid costs one retrieval pass. Scoring
runs through the committed `evaluate_cohort_multi_k`, not a parallel
implementation, so the numbers are comparable to `baselines.json`.

## Result: do not change the default

`recall@50`, percentage change against the shipped `pool=50, k=60`:

| cell | SIGIR | TREC 2021 |
|---|---|---|
| pool=50 k=60 (shipped) | 0.5235 | 0.2859 |
| pool=50 k=40 | +5.7% | **-0.7%** |
| pool=50 k=20 | +5.1% | **-1.5%** |
| pool=50 k=10 | **+5.9%** | **-3.3%** |
| pool=50 k=5 | +3.8% | **-7.2%** |
| pool=100 k=20 | +5.8% | +1.4% |
| pool=200 k=20 | +2.6% | **+1.5%** |

**The cohorts disagree in sign.** SIGIR's best cell (`pool=50, k=10`, +5.9%) is
one of TREC's worst (-3.3%), and the disagreement widens as `k` falls: at `k=5`
SIGIR gains 3.8% while TREC loses 7.2%. On TREC the shipped default ranks 6th of
40 cells and sits within 1.5% of the best. There is no free recall here.

Had this been run on SIGIR alone it would have looked like a +5.9% recall and
+21% MRR win for a one-line change. That is the second time in three days a
SIGIR-only conclusion has failed to survive TREC (see `e1b_findings.md`), and
the first time the sign flipped rather than the magnitude.

## Why the disagreement is plausible

RRF's `k` damps the advantage of top ranks: at `k=60` rank 1 beats rank 50 by
1.8x, at `k=10` by 5.5x. Lowering `k` sharpens trust in each list's head.

That helps when the lists are individually good and hurts when they are noisy.
The cohorts sit in different regimes — SIGIR MRR is 0.3253 against TREC's
0.6257, and TREC labels ~76 eligible trials per patient against SIGIR's few. On
TREC many partially-relevant trials sit deep in each list and are worth
accumulating; sharpening throws them away.

## The one honest candidate

`pool=200, k=20` is the only cell positive on both cohorts: SIGIR +2.6%, TREC
+1.5%. Not adopted, for three reasons: both gains are small enough to be noise
at n=59 and n=75, no significance test was run, and pool=200 quadruples the
fusion input for the ~1.5% it might buy. Worth revisiting only with a paired
per-patient test.

## Standing

`k=60` and `pool=50` stay as shipped, now measured rather than inherited. A
negative result was one of the two outcomes this item was specified to produce.
