# R3 — weighting fusion by keyword rank

Run 2026-08-31. Ten weighting schemes on both cohorts at the validated defaults
(`pool=50`, `k=60`). Retrieval only, no LLM calls. SIGIR local, TREC on a
short-lived EC2 box.

## The premise is real

`query_transform.py` prompts the model: *"Order most-to-least important for
trial matching."* The ranking is requested, paid for in the keyword call, and
then discarded — `fusion.py:19` sums `1/(k+rank)` over every list with equal
weight, so the 12th keyword counts exactly as much as the 1st.

## Sweep: consistent direction on both cohorts

`recall@50` change against the shipped uniform weighting:

| scheme | SIGIR | TREC 2021 |
|---|---|---|
| uniform (shipped) | 0.5235 | 0.2859 |
| reciprocal `1/(1+i)` | **+10.8%** | **+3.4%** |
| log `1/log2(i+2)` | **+11.7%** | +3.0% |
| exp `0.7^i` | +10.4% | +1.3% |
| exp `0.8^i` | +9.0% | +1.2% |
| linear `(n-i)/n` | +7.7% | +0.9% |
| top-4 only | +5.5% | **-4.8%** |
| top-6 only | **-5.9%** | **-4.7%** |
| top-8 only | -0.2% | -3.2% |

Every *graded* scheme is positive on both cohorts; every *hard cutoff* is
negative on TREC. Smooth downweighting helps, truncation hurts. Unlike R5, the
cohorts agree in sign — only the magnitude differs.

## Significance: it does not clear the bar

Recall is paired by patient, so the test is paired over patients, not a pooled
proportion. Uniform vs reciprocal, `recall@50`:

| | SIGIR | TREC 2021 |
|---|---|---|
| patients | 52 | 75 |
| uniform | 0.5693 | 0.2860 |
| reciprocal | 0.6264 | 0.2956 |
| mean delta | +0.0570 | +0.0096 |
| bootstrap CI95 | **[-0.0019, 0.1177]** | **[-0.0072, 0.0266]** |
| Wilcoxon p | 0.0673 | 0.3031 |
| better / worse / tied | 23 / 11 / 18 | 40 / 24 / 11 |

Both confidence intervals cross zero. Fisher's combined p over the two cohorts
is **0.0998**. Not adopted.

(The recall@50 values here differ from the sweep table because the sweep scores
every patient through `evaluate_cohort_multi_k` with coverage adjustment, while
this test is restricted to the patients with in-corpus gold and is unadjusted.
Each is internally consistent; do not mix the two.)

## Standing

`fusion.py` is unchanged and stays uniform. The direction is consistent across
two cohorts, seven graded schemes, and inverts for cutoffs — which is more
structure than noise usually produces — but "more structure than noise usually
produces" is not a result. On the evidence it is a promising hypothesis with
p ~= 0.10, and the project's rule for this item was to adopt only on a real lift.

Worth revisiting with more patients, since both cohorts trend the same way and
TREC's 40/24 patient split is the kind of margin a larger n would resolve. The
sweep harness and per-patient test are committed, so a rerun is cheap.
