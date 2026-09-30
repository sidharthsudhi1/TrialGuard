# AD-35 — Per-keyword lists fed to RRF are 200 deep, not 50

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

Every per-keyword dense and lexical list fed to RRF is 200 deep, set by `fusion.list_pool()`. `TG_LIST_POOL=50` reproduces every earlier ranking. With 50-deep lists, a trial scores only by reaching the top 50 of some single keyword's list, so a trial that ranks moderately for many keywords scores nothing, and that is what an eligible trial usually looks like. On TREC 2021 / 2022, going from 50 to 200 lifts recall@100 by +0.018 / +0.022 and recall@200 by +0.073 / +0.067 (Wilcoxon + BH, p_bh ≤ 0.03). SIGIR is flat: 2,991 trials, and gold is already at the head. [`r1_findings.md`](../../data/reports/r1_findings.md), [`r1b_findings.md`](../../data/reports/r1b_findings.md)

This reverses a recorded result. `structural_recall_plan.md` §5 found 1000-deep fusion diluted recall@200. That was measured under uniform weights. Under the `1/i` keyword decay adopted on 2026-09-04, deep lists from the important keywords add consensus instead of noise. Two levers measured separately interacted.

The dense half alone delivers most of the gain (+0.016–0.021 @100, +0.055–0.059 @200), and it runs on the same exact matrix product in production as in eval, so that share carries over as measured. The lexical half is measured on rank-bm25 only. Production uses Postgres FTS. Its latency at `LIMIT 200` was measured on the production corpus (read-only `EXPLAIN ANALYZE`, 60 keywords, warmed, randomized order) and is identical to `LIMIT 50`: p50 0.46 vs 0.43 ms, p95 20.1 vs 20.5 ms. Only 28% of keywords match 200 trials at all. Its recall effect stays unmeasured because TREC is not in Postgres. Deploying waits only on an E6 warm-path check against the 1500 ms SLO.

## Alternatives considered

- **Pool 1000.** Recall@100 and @200 are the same as at 200; only @500 keeps improving, and that depth isn't served yet.
- **A learned reranker over these same signals (R1).** It never beat plain deep fusion, and on 5 of 8 cross-cohort cells it was significantly worse.
- **Deepening the dense lists only.** It captures most of the gain with zero backend risk, but it leaves the lexical increment unmeasured instead of measured.
