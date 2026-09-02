# H2 — Listwise LLM rerank: measured, rejected

Run 2026-09-02. TREC 2021 and TREC 2022, 20 patients each, pool 500 → top-50,
batches of 50, prompt-screened by the served model (DeepInfra Llama-3.3-70B).
$0.14 and 24 min for both cohorts. Raw: `h2_listwise_trec_2021.json`,
`h2_listwise_trec_2022.json`.

## Why it was tried

H1 established that the agent converts retrieved trials at a depth-independent
rate, so surfaced recall scales with the pool it is handed — 6.4x / 6.0x from
top-10 to top-100 on the two TREC cohorts. That makes deep pools a *cost*
problem: a full assessment is ~29 s and ~$0.0005 per trial, so top-500 is not
servable. H2's premise was that a cheap screening pass (one call per 50
candidates, ~1/100th the tokens of assessing them) could carry a 500-deep pool's
recall into a 50-trial slice, and that unlike the cross-encoders R4 rejected the
judgment would transfer — it is the same kind the analyst already makes, on the
same text, rather than a web-search relevance score from another domain.

## Result

| | TREC 2021 | TREC 2022 |
|---|---|---|
| pool ceiling (recall@500) | 0.5757 | 0.6303 |
| baseline recall@50 | 0.1836 | 0.2413 |
| **reranked recall@50** | **0.1737** | **0.2041** |
| delta | **−0.0099** | **−0.0372** |
| per-patient split | 10 better / 9 worse | 6 better / 11 worse |
| cost | $0.077 | $0.064 |

Both cohorts land on the wrong side of zero, and the second lands further out
than the first. The kill criterion in `docs/structural_recall_plan.md` was
≥ +0.05 recall@50. Unlike R3 and R5, which disagreed between cohorts or shrank
under replication, this one agrees in sign on both — the rejection is durable.

## The mechanism, not just the aggregate

A flat aggregate can hide a working component behind a bad slice, so the kept
set was scored directly against a size-matched prefix of the same fused ranking:

| | trials | gold captured | recall |
|---|---|---|---|
| screener's kept set | ~91 | 403 | 0.2651 |
| retrieval's top-91 | 91 | 422 | **0.2776** |

The screener is not passive — it discards 82% of the pool (keeps a median of 90
of 500, range 36–148). It is discriminating, and what it keeps is *slightly less*
gold-rich than simply taking the head of the ranking at the same depth. So the
negative is not an artifact of the top-50 slice or the batch-interleaving order:
the screening judgment carries no signal beyond what retrieval already encodes.

## Reading

**The failure is informative about where the agent's value actually lives.** The
same model, on the same trials, is measurably discriminating when it assesses:
it never emits `eligible` on a gold-excluded trial and holds ~1.6x lift over
its pool's base rate through a 10x dilution (H1). Shown 50 trials as one-line
summaries and asked to screen, that discrimination disappears. What the analyst
does is read one trial's criteria against the note carefully; that does not
compress into a cheap glance across fifty.

This is consistent with, and strengthens, R4's rejection. Reranking has now
failed here three times — general-domain cross-encoder (−0.173), MedCPT's own
head (−9.5% at best), and now an LLM screener with full task context. The
fusion-stage conclusion from `pipeline_review_outcomes.md` extends: this pool
does not respond to reordering, by any mechanism tried.

**Cost was never the blocker.** The screening pass came in at ~$0.0038/patient
against a $0.01 budget, comfortably inside target. It is rejected on quality, so
the cost figure is recorded only to close that question: cheap reranking is
available and not worth buying.

## What it means for H1

H1 stands unchanged — it was measured independently and does not depend on this.
What H2's failure removes is the cheap path to *serving* deep pools. The options
that remain for turning H1 into product behaviour are all economic rather than
algorithmic:

- assess a deeper pool and pay for it (~$0.05 and ~10 min/patient at top-100),
- assess deeper only for a user who asks, making depth an explicit opt-in,
- or accept top-10 serving while reporting the top-100 number as the system's
  measured capability, which is what the README now does.

## Caveats

- n=20 patients per cohort. Both cohorts of record agree in sign; the
  matched-N diagnostic was run on TREC 2021 only.
- One prompt, one batch size (50), one pool depth (500). A different screening
  prompt might do better, but the matched-N diagnostic argues the ceiling is the
  task framing rather than the wording: the model is filtering confidently and
  wrongly, not failing to answer.
- Cost is DeepInfra FP8 Llama-3.3-70B; a stronger model might screen better and
  would cost more, which moves it out of the "cheap middle stage" this
  hypothesis was about.
