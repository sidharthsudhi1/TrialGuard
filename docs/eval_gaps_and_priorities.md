# Eval gaps and fix priorities (Stage A live)

Written 2026-08-16, after the Stage A deploy went live. This is a gap review, not
a new measurement. Every number here is read from committed reports; nothing was
re-run.

## What is already recorded (and does not need re-litigating)

A review pass initially flagged five "weak spots". Four were already documented
in the README and `docs/aws_trec_eval.md`:

- v4 TREC 2021 exclusion unsupported rate 31.2% vs inclusion 9.2% — README.
- Retry not significant on the v4 TREC mix (−10%, Fisher p=0.2514) — README.
- SIGIR cannot measure v4 (`by_kind.exclusion.n_criteria` is 0); the SIGIR v4
  rerun measures the `MAX_CRITERIA` 12→24 lift, not exclusion handling — README
  and `aws_trec_eval.md`.
- `recall@10` retired as a target, capped at `min(10, |gold|)/|gold|` — README.
- The CI gate stays anchored to Phase 8 SIGIR, and the v4 TREC report is
  explicitly *not* a new baseline — README.

The confounder note in `aws_trec_eval.md` matters and is easy to trip over: the
SIGIR v1→v4 movement (trial accuracy 0.2611 → 0.3722, unsupported rate 0.0287 →
0.0966) is the criteria-cap change, **not** a prompt-quality regression. Any
v2-vs-v4 comparison on SIGIR is confounded and should not be read as one.

## Gaps that remain

### G-1 — The metric-targets table does not carry the v4 caveat

README's prose is accurate. The summary table under **Metric targets** still
reports the hallucination row as ✅ with the v1/v2 retrieval-aware-retry numbers
(SIGIR −68.6%, TREC 2021 −70.2%, TREC 2022 −66.7%) with no marker that the
served config (v4, per `/api/health`) does not reproduce them on a cohort that
exercises exclusion. A reader who reads only the table over-reads the claim.

Fix: add the v4 qualifier to that row, or split the row by prompt version. Cheap,
and it is the summary most readers trust.

### G-2 — Verification does not improve trial accuracy on v4

Not previously noted. Both v4 cohorts, baseline vs verified arm:

| Cohort | baseline | verified |
|---|---|---|
| SIGIR | 0.3667 | 0.3722 |
| TREC 2021 | 0.4444 | **0.4333** |

The verified arm is flat on SIGIR and slightly *worse* on TREC 2021. This is
consistent with the design — grounding converts unsupported decisive verdicts
into `unverifiable`, and a trial that rolls up `cannot_determine` never matches
an `eligible`/`excluded` gold label — but it means the verification mechanism
buys faithfulness without buying trial-level correctness, and that tradeoff is
currently unstated. Worth saying explicitly rather than leaving a reader to infer
that verification improves every metric.

### G-3 — Served-path abstention is unmeasured

PHASE9 WS-6 calls for measuring what fraction of *served* assessments return
`cannot_determine`, on the grounds that a divergence from the eval rate means
either the live corpus differs from the cohorts or the serving path is broken.
`src/trialguard/api/routes.py` currently emits no verdict-outcome logging, so
this is unmeasurable on the live deployment.

This is the only gap that is about the deployed system rather than the reports.

### G-4 — Exclusion grounding — RESOLVED 2026-08-17

Fixed in v5; full write-up in
[`data/reports/phase9v5_exclusion_grounding.md`](../data/reports/phase9v5_exclusion_grounding.md).

Diagnosis was none of the three hypotheses below. Reading the cached traces, every
exclusion failure was one pattern: exclusion `not_met` asserts *absence of
evidence*, which no verbatim span can support, so the analyst returned an empty
quote or an invented negation and the verifier scored correct reasoning as
unfaithful. Absence is now verified mechanically against the patient note instead
of exempted. Exclusion unsupported rate 31.2% → 8.9%; TREC 2021 retry significance
restored (p=0.2514 → p=0.0048).

G-2 is partly addressed as a side effect: the verified arm now beats baseline on
trial accuracy (0.450 vs 0.444) instead of trailing it.

Original framing, kept for the record:



Recorded in README as a finding; not yet recorded as a work item. 31.2%
unsupported on exclusion means roughly one in three decisive exclusion verdicts
cites a quote that does not ground, and the retry barely moves it (0.664 → 0.688,
against inclusion 0.892 → 0.908). About 71% of remaining unsupported decisive
verdicts are exclusion.

Hypotheses worth testing in order, cheapest first:

1. The grounding source span passed for exclusion criteria differs from the span
   the analyst quoted from.
2. Inverted semantics ("must NOT have X") produce a paraphrase rather than a
   verbatim span.
3. The retrieval-aware retry does not carry the exclusion span the way it carries
   the inclusion one.

### G-5 — The 87% criterion-matching target is unmeasurable as specified

The cohorts carry trial-level qrels (`eligible` / `excluded`), not criterion-level
gold. Measuring criterion-matching accuracy against the TrialGPT benchmark number
would require per-criterion labels that do not exist in the current data and that
CLAUDE.md explicitly rules out hand-labelling from scratch.

Fix: rescope the target to something the harness can actually produce, or retire
it the way `recall@10` was retired, with the reason stated.

## Priority

| | Gap | Effort | Why this rank |
|---|---|---|---|
| P0 | G-1 metric-table caveat | ~1h | Public page; smallest fix with the largest correctness-of-claim effect |
| P1 | G-4 exclusion grounding | days | Largest technical lever; likely restores TREC significance |
| P2 | G-3 served-path abstention | ~half day | Only gap about the live system; validates that eval transfers to `ctgov_live` |
| P3 | G-2 trial-accuracy statement | ~1h | Documentation honesty, no code |
| P4 | G-5 retire or rescope 87% | ~1h | Removes a target that cannot be met as written |

G-2 and G-5 are documentation changes. G-1 is documentation with a public-facing
consequence. G-3 is a small serving change. G-4 is the only one that is real
engineering, and it is the one that would move the headline numbers.
