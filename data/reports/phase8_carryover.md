# Phase 8 WS-4 — Phase 4 carryover: prompt v2 and the retrieval-aware retry

Two measurements that were code-complete since Phase 4 and blocked on the Groq
free-tier daily cap. Both now run. **Total cost: $0.0958.**

Date: 2026-08-11 · Provider: DeepInfra (`meta-llama/Llama-3.3-70B-Instruct-Turbo`)

---

## 1. Prompt v2 lowers abstention *and* raises faithfulness

v2 was written against v1's over-abstention (~0.73), where `cannot_determine` was
explicitly "encouraged". The open question was whether buying coverage would cost
faithfulness — the obvious way to abstain less is to commit to shakier verdicts.

It did not. Same host, same 180-trial SIGIR subset, so this is a clean
within-host prompt comparison:

| Metric (verified arm) | v1 | v2 | Δ |
|---|---|---|---|
| `citation_precision` | 0.9713 | **0.9896** | +1.8pp |
| `unsupported_verdict_rate` | 0.0287 | **0.0104** | −1.8pp |
| `abstention_rate` | 0.7047 | **0.6543** | −5.0pp |
| `coverage` | 0.2953 | **0.3457** | +5.0pp |
| `trial_accuracy` | 0.2611 | **0.3778** | +11.7pp |

Baseline arm moves the same way: abstention −4.3pp, citation precision +3.7pp,
trial accuracy +7.8pp.

**Reading:** v1's abstention was largely the analyst declining to look for
evidence it could have found, not evidence genuinely being absent. v2's
instruction — scan both texts for a decisive fact *before* answering
`cannot_determine` — recovers those criteria without weakening the verbatim-quote
requirement. Trial accuracy rising 11.7pp is the downstream effect: fewer criteria
stuck at `cannot_determine` means more trials resolving to a real
eligible/excluded roll-up instead of `cannot_determine`.

**Retry is not made redundant by the better prompt.** On v2, the matched A/B is
still significant: relative change −0.8073, Fisher p=0.0006 over 177 matched
trials. The prompt and the verifier address different failure modes.

Reports: `phase8di_agent_sigir.json` (v1), `phase8v2_agent_sigir.json` (v2).

---

## 2. The retrieval-aware retry makes TREC significant

Phase 4 rewrote the retry to inject the **exact trial source span** plus the
failed criteria, replacing a generic "copy verbatim" nudge. The stated hypothesis
was that the generic nudge only recovered *paraphrase* failures — which is why it
worked on SIGIR and not on TREC — while TREC's failures were verbatim misses that
need the characters to copy from.

That hypothesis had never been measured. It holds on both cohorts:

| Cohort | Retry | baseline unsup. | verified unsup. | rel. change | Fisher p |
|---|---|---|---|---|---|
| TREC 2021 | Phase 3, generic | 0.1200 | 0.1126 | −6% | not computed |
| **TREC 2021** | **Phase 4, span** | 0.1333 | **0.0397** | **−70.2%** | **0.0103** |
| TREC 2022 | Phase 3, generic | 0.1230 | 0.0940 | −24% | not computed |
| **TREC 2022** | **Phase 4, span** | 0.1484 | **0.0495** | **−66.7%** | **0.0168** |

Both are significant at α=0.05. This retires the standing claim in `CLAUDE.md`
and the README that TREC 2021/2022 are "not significant at n~60" — that was true
of the *generic* retry, not of the retrieval-aware one.

### The confound, and why the retry is still the best explanation

These runs changed **two** things against the Phase 3 reports: retry logic *and*
provider. That is not a controlled experiment, and the finding should not be
quoted without this paragraph.

The arms decompose the confound:

- **Baseline arms involve no retry at all** (`max_retries=0`), so baseline-vs-baseline
  is a pure host comparison: 0.1200 → 0.1333 (2021) and 0.1230 → 0.1484 (2022).
  Close, and if anything DeepInfra's baselines are slightly *worse*.
- **Verified arms diverge sharply**: 0.1126 → 0.0397 and 0.0940 → 0.0495.

If the host were driving the improvement, the baselines would have moved with it.
They didn't. The effect is in the retry.

A strictly clean test would re-run the Phase 3 generic retry on DeepInfra
(~$0.02). Not done; recorded here as the outstanding control.

Reports: `phase8v1_agent_trec_2021.json`, `phase8v1_agent_trec_2022.json`.

---

## Caveats

- **`n_criteria` denominators differ between prompts and hosts** (1244 v1 vs 1190
  v2 on the same 180 trials; ~1–4%). The analyst enumerates criteria slightly
  differently, so these are comparable *rates*, not a strictly matched
  per-criterion comparison. Same ~1% effect noted in the parity report. Worth
  watching if it grows — it would point at truncation/`_salvage` behavior rather
  than at grounding.
- **v2 on TREC exists only on Groq** (`phase4v2_agent_trec_*`, complete). Not
  re-run here; a v1-vs-v2 TREC comparison would be cross-host.
- **Baselines unchanged.** `baselines.json` still enforces the Groq-derived
  floors. Updating them is a deliberate decision that should follow from a
  post-migration baseline, not ride along with the migration.

---

## Cost

| Run | Spend | Fresh calls | Wall clock |
|---|---|---|---|
| `phase8di_agent_sigir` (WS-3 parity, v1) | $0.0243 | 129 | 48 min |
| `phase8v2_agent_sigir` (v2 A/B) | $0.0390 | 203 | 74 min |
| `phase8v1_agent_trec_2021` (retry A/B) | $0.0181 | 82 | 34 min |
| `phase8v1_agent_trec_2022` (retry A/B) | $0.0144 | 78 | 29 min |
| **Total (WS-3 + WS-4)** | **$0.0958** | **492** | ~3 hr |

Every call billed from DeepInfra's own reported cost, not a local price table.

Four measurements that were blocked for weeks on a free-tier quota cost **under
ten cents**. That is the Phase 8 cost story in one line: the constraint was never
money, it was the shape of the free tier.
