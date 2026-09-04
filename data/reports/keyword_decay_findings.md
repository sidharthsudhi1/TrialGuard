# Keyword-importance decay: R3 re-opened at the depth that matters

Run 2026-09-04. TREC 2021 (n=75) and TREC 2022 (n=50), cached MedCPT indices,
cached keywords, per-list pool 100. No LLM calls, $0. Raw:
`keyword_decay_trec2021.json`, `keyword_decay_trec2022.json`.

**Adopted.** `fusion.py` now weights each keyword's ranked lists by `1/i`,
where `i` is the keyword's importance rank. `TG_KEYWORD_DECAY=0` restores the
previous ranking.

## Why this was re-opened after R3 rejected it

R3 tested graded weighting and did not adopt it: directionally positive on both
cohorts but Fisher combined p=0.0998. That measurement was taken at
**recall@50**, and it was taken before H1.

H1 then established that the agent converts a retrieved pool at a
depth-independent rate, so surfaced recall scales with pool size — which moved
the metric that matters from recall@50 to recall@100–200. Re-measured there,
the same lever is significant on both cohorts.

The literature pointed at the same place. TrialGPT-Retrieval (Jin et al.,
*Nature Communications* 2024), the system this pipeline's architecture derives
from, scores candidates as

```
s_j = SUM_Ret SUM_i (1/i) * 1 / (Rank(Ret, w_i, t_j) + C),   C = 20
```

with a `1/i` keyword-importance decay this repo had dropped. It reports
recall@500 of 0.834 (GPT-3.5) / 0.862 (GPT-4) against the 0.766 measured here,
so the gap was worth attributing rather than assuming.

## Result: C x decay, swept jointly

The two halves had only ever been tested apart — R3 varied the weighting at
fixed `C`, R5 varied `C` at uniform weighting. Decay wins in **40 of 40 cells**
across both cohorts, and `C` barely matters once it is present.

TREC 2021 (n=75, gold 5,569):

| C | decay | r@10 | r@50 | r@100 | r@200 | r@500 |
|---|---|---|---|---|---|---|
| 60 | uniform *(shipped)* | 0.0503 | 0.2033 | 0.3374 | 0.4951 | 0.6795 |
| 60 | **1/i** | 0.0539 | 0.2112 | **0.3546** | **0.5250** | 0.6920 |
| 20 | 1/i | 0.0549 | 0.2110 | 0.3512 | 0.5118 | 0.6901 |
| 10 | 1/i | 0.0546 | 0.2097 | 0.3464 | 0.5001 | 0.6865 |

**TrialGPT's `C=20` is not the lever.** At uniform weighting the four `C` values
span 0.0038 of recall@200; adding decay at the existing `C=60` moves it 0.0300.
R5's conclusion that the constant does not respond to tuning stands.

## Significance, paired per patient

Wilcoxon signed-rank over per-patient hit counts, `C=60` uniform vs `C=60` 1/i:

| cohort | depth | base | decay | delta | better/worse | p |
|---|---|---|---|---|---|---|
| TREC 2021 | @50 | 0.2033 | 0.2112 | +3.9% | 33/25 | 0.2301 |
| TREC 2021 | @100 | 0.3374 | 0.3546 | +5.1% | 37/24 | **0.0386** |
| TREC 2021 | @200 | 0.4951 | 0.5250 | +6.1% | 39/17 | **0.0014** |
| TREC 2022 | @50 | 0.2341 | 0.2564 | +9.5% | 32/8 | **0.0001** |
| TREC 2022 | @100 | 0.3920 | 0.4153 | +6.0% | 30/14 | **0.0025** |
| TREC 2022 | @200 | 0.5588 | 0.5826 | +4.3% | 26/10 | **0.0063** |

Significant on **both** cohorts at recall@100 and recall@200. TREC 2021 at
recall@50 reproduces R3's null (p=0.2301), which is the point: the finding was
not wrong, it was measured where the effect is weakest.

## On the served path

Full `eval_retrieval` run, production defaults (pools 50/50), TREC 2021 n=75:

| | before | after | |
|---|---|---|---|
| recall@10 | 0.0798 | 0.0863 | +8.1% |
| recall@50 | 0.2859 | 0.2955 | +3.4% |
| recall@100 | 0.4227 | 0.4396 | +4.0% |
| recall@200 | 0.5454 | 0.5619 | +3.0% |
| MRR | 0.6257 | 0.6583 | +5.2% |

MRR rises with recall, so this is not depth bought with head quality. Latency is
unchanged: the weights are computed from list positions, not from new retrieval.

## Rejected alongside it: coverage-oriented fusion

A competing hypothesis was that RRF's "Authority Effect" — a document's score
rising with the *number* of lists retrieving it — buries specialist trials that
match one clinical aspect strongly, which would matter when a patient has ~76
diverse eligible trials. Two coverage-first operators were built and measured
against RRF on TREC 2021 at identical inputs (`fusion_operator_ab_trec2021.json`):

| operator | r@10 | r@100 | r@500 |
|---|---|---|---|
| RRF | **0.0503** | **0.3374** | **0.6795** |
| round-robin interleave | 0.0336 | 0.2185 | 0.5976 |
| max-normalised score | 0.0447 | 0.2352 | 0.6260 |

RRF wins at every depth. The Authority Effect is doing real work here and the
hypothesis is dead. Recorded because it was the more interesting theory and it
was wrong.

## Caveats

- TREC 2022 retrieves against the `medcpt_noexcl` index, the only variant
  cached for that cohort, as every prior committed 2022 number does. Absolute
  values are not comparable across cohorts; the paired within-cohort deltas are.
- `1/sqrt(i)` is close behind `1/i` and wins recall@500 on both cohorts. `1/i`
  was chosen for matching the published formula and winning the depths H1
  identified; the difference between them is inside noise.
- Not tested: keyword count. TrialGPT generates up to 32; this repo caps at 12.
  That is the remaining unexplained difference from the published configuration
  and it needs fresh keyword generation to measure.
