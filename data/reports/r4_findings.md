# R4 — clinical cross-encoder rerank (SIGIR only)

Run 2026-08-31 on SIGIR, keyword retrieve@200 -> rerank -> top-50, on 8 dedicated
vCPU. **The TREC arms were run but lost** — see "What is missing" below.

## Why re-test something already rejected

`phase2_rerank.md` rejected reranking at -0.173 recall@50, and the review
attributes that to the model being general-domain. That run in fact varied two
things at once: it used `ms-marco-MiniLM` *and* the full patient note as the
rerank query — the narrative-vs-eligibility mismatch AD-11 blames for the whole
retrieval ceiling. Model and query form were never separated.

## Result: still worse than no rerank

| arm | recall@10 | recall@50 | MRR | vs baseline | wall |
|---|---|---|---|---|---|
| **baseline (no rerank)** | 0.1827 | **0.5235** | 0.3253 | — | 56s |
| ms-marco + note | 0.1392 | 0.3508 | 0.2501 | **-33.0%** | 267s |
| medcpt + note | 0.1816 | 0.4740 | **0.3479** | -9.5% | 1260s |
| medcpt + keywords | 0.1746 | 0.4658 | 0.3043 | -11.0% | 1097s |

**The attribution was wrong, the conclusion was right.** Swapping ms-marco for
`ncbi/MedCPT-Cross-Encoder` recovers most of the damage (-33.0% -> -9.5%), so the
model was the dominant term, as the review guessed. But changing the query form
from note to keywords does nothing (-9.5% vs -11.0%), so the AD-11 mismatch story
does not transfer to reranking. A clinical cross-encoder is much less bad and
still loses to not reranking.

`medcpt + note` is the only arm that beats baseline on **MRR** (0.3479 vs
0.3253): it orders the head better while losing recall at depth, which is what a
precision-oriented reranker compressing 200->50 should do. Not useful here —
recall@pool is the primary metric and the agent consumes the whole pool.

## Latency disqualifies it independently

1260s for 59 patients is **~21s per patient** on 8 dedicated vCPU, against a
served search path of 780ms. Even had recall improved, this is not deployable
without a GPU. That holds regardless of what any other cohort shows.

## What is missing

The TREC arms completed on the box and were destroyed with it: the SIGIR report
was pulled at 13:19 UTC, no monitor was re-armed for TREC, and the scheduled
shutdown terminated the instance at 17:31 with the TREC report unpulled. ~2h of
compute lost. R5 flipped sign between cohorts, so SIGIR-only conclusions in this
repository have a poor record and this one is labelled accordingly.

Re-running costs roughly $0.90 and two hours. It is not ranked, because the
latency finding disqualifies deployment on any cohort's recall.

## Standing

`RERANK_MODEL` still defaults to ms-marco and rerank remains off the production
path. What changed is that `rerank()` now takes `model_name`, and the score cache
is keyed on (note, model) rather than note alone — without that a second model
silently reads back the first model's ranking and reports it as its own.
