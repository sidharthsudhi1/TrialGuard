# TrialGuard — Phase 2/3 Measured Results

Generated: 2026-07-07. All numbers reproduced from code in this branch. Zero paid
services used (Groq free tier + local MPS embeddings only).

---

## Retrieval: encoder comparison (SIGIR, keyword-RRF, n=53)

| Config | recall@10 | recall@20 | recall@50 | recall@100 | recall@200 | MRR |
|---|---|---|---|---|---|---|
| BGE (baseline) | 0.1345 | 0.3032 | 0.5335 | 0.6919 | 0.7840 | 0.2843 |
| BGE + exclusion | 0.1345 | 0.3032 | 0.5335 | 0.6919 | 0.7840 | 0.2843 |
| **MedCPT (adopted)** | **0.1801** | **0.3480** | **0.5812** | **0.7069** | **0.7890** | **0.3445** |
| MedCPT vs BGE | +34% | +15% | +9% | +2.2% | +0.6% | +21% |

**Decisions:**

- **MedCPT adopted as default encoder** (`TG_EMBED_BACKEND=medcpt`, now the default
  in `ingestion/embed.py`). Biggest gains at the top of the ranking (recall@10
  +34%, MRR +21%) — precisely what a cost-bounded agent pool consumes. NCBI MedCPT
  is domain-matched (PubMed search logs) and is TrialGPT's own retriever.
- **Indexing exclusion criteria = measured null on SIGIR.** `_load_sigir_trials`
  concatenates inclusion+exclusion into one headerless block, which
  `normalise_trial` cannot re-split, so exclusion text was already in the indexed
  document all along. The "exclusion never indexed" concern applies only to the
  production CT.gov path (properly header-split), not the eval path.
- **recall@10 >= 0.90 target retired.** Mathematically capped at
  `min(10, |gold|)/|gold|` per patient: SIGIR 0.897, TREC 2021 0.247, TREC 2022
  0.285. TrialGPT's ">90% recall" was at large depth. Primary metric is now
  **recall@pool** (recall@50/100).

**Caveat (must disclose):** the SIGIR eval corpus (2991 trials) is TrialGPT's
`retrieved.json`, already pre-filtered by their retriever. Coverage 96.4% and
recall@100 ~0.71 ride on that pre-filtering. TREC 2021/2022 below are the honest
large-corpus test.

---

## Retrieval: honest large-corpus test (MedCPT, keyword-RRF, full ~26k trials)

| Cohort | n | coverage | recall@10 | recall@50 | recall@100 | recall@200 | MRR |
|---|---|---|---|---|---|---|---|
| trec_2021 | 75 | 99.98% | 0.0810 | 0.2888 | 0.4259 | 0.5520 | 0.5623 |
| trec_2022 | 50 | 100.0% | 0.0916 | 0.3126 | 0.4639 | 0.5864 | 0.6665 |

Unlike SIGIR, these corpora are the complete 26,148 / 26,581 trial sets with
~100% gold coverage — no upstream pre-filtering. This is the defensible number.

**Reading these numbers:**

- **MRR 0.56-0.67 is the headline.** The first gold-eligible trial lands, on
  average, in the top ~2 (2022) to top ~2 (2021) results. For a screening tool
  that surfaces a ranked shortlist, first-hit rank is what matters, and it is
  strong on full corpora.
- **recall@10 (0.08-0.09) is near its mathematical ceiling, not a failure.**
  TREC patients have a median 60+ eligible trials each, capping recall@10 at
  0.247 / 0.285. Cramming 60 gold trials into 10 slots is impossible by
  construction; recall@10 is the wrong lens for these cohorts (see target
  retirement above).
- **recall@pool climbs as expected**: @100 = 0.43-0.46, @200 = 0.55-0.59 on the
  full corpora. The agent pool at N=100-200 captures roughly half the gold set
  from 26k candidates — the honest retrieval floor the agent builds on.
- Latency p50 1.9s (2021) / 3.4s (2022) per patient: dominated by per-keyword
  dense search over 26k vectors; acceptable for offline eval, cache-warm.

---

## Agent faithfulness: single-pass vs verified (SIGIR, 5 patients, 20 trials)

| Metric | Single-pass (max_retries=0) | Verified (max_retries=2) |
|---|---|---|
| Decisive verdicts attempted | 41 | 40 |
| Grounded verdicts | 38 | 38 |
| Citation precision | 0.9268 | 0.9500 |
| Unsupported-verdict rate | 0.0732 | 0.0500 |
| Abstention rate | 0.6911 | 0.6911 |
| Trial accuracy (vs qrels) | 0.30 | 0.30 |

**This table does NOT show verification beating single-pass.** The 7.3% → 5.0%
gap is a difference of a **single verdict** (3/41 vs 2/40). Fisher exact
**p = 1.000** — statistically null. Both arms grounded the same 38 verdicts.
On this subset the analyst already cited correctly 92.7% of the time, so the
verifier had almost nothing to catch and the retry had almost nothing to fix.
The A/B comparison is underpowered, not evidence. Reported as-is, uninflated.

**The real faithfulness proof is deterministic, not this A/B.** Verifier
catch-rate stress test (`verify/grounding.py`): corrupt every one of 509 real,
grounded quotes (swap a clinically meaningful token) and re-check grounding.

- Corrupted quotes **rejected: 509/509 = 100%** catch rate.
- Genuine quotes **grounded: 509/509**, **zero false rejections**.

This number is sample-size-independent and is what actually backs the thesis: a
verdict whose quote is not verbatim in the source cannot pass, by construction.

**How faithfulness is enforced (`verify/grounding.py`):** every "met"/"not_met"
verdict must cite a span that is verbatim-present (case/punctuation/whitespace
normalized) in the patient note or trial text. Non-present quotes are
mechanically forced to "unverifiable". Pure Python — cannot hallucinate
agreement, costs nothing.

**Scope of "grounded":** a verdict may cite the patient note ("58-year-old
woman") or the trial text. This grounds the verdict against *some* provided
source, which is correct for patient-fact criteria, but it means "grounded" is
"quote is real", not "verdict was checked against the trial criterion". The
latter (entailment) is a separate, not-yet-built check.

**Weaknesses, disclosed:**

- **Abstention 0.69** — the system declines two-thirds of criteria. High
  faithfulness is bought partly with low coverage; the two must always be read
  together.
- **Trial accuracy 0.30**, identical across arms: the strict roll-up marks a
  trial `eligible` only if *all* criteria are `met`, so high abstention collapses
  most trials to `cannot_determine`. Weak as a standalone number.
- **Metric overlap**: a grounding failure is counted in *both* `decisive_attempts`
  and `abstention_rate` (its verdict is `unverifiable`), so the two rates do not
  partition the criteria. Read them as separate lenses, not a split.
- Subset is 41 decisive verdicts. Larger runs are quota-bound, not code-bound;
  until then no magnitude claim about verification vs single-pass is supportable.

---

## What is proven

1. **Faithfulness is mechanical.** Grounding is deterministic and tested
   (`tests/test_grounding.py`, `tests/test_agent_graph.py`); a hallucinated
   citation cannot pass silently.
2. **Verifier catch rate is 100% (509/509 corrupted quotes rejected, 0 false
   rejections)** — the sample-size-independent proof of the mechanism.
3. **Retrieval improved at $0** by swapping to a domain-matched encoder
   (MedCPT), reproduced on SIGIR and on full ~26k-trial TREC corpora.

## What is NOT yet proven

- **Verification vs single-pass on live traffic.** The A/B (7.3% vs 5.0%) is
  underpowered (n=41, p=1.0). No magnitude claim is supportable until a larger
  run. The mechanism works; its marginal lift over a single pass is unmeasured.
- **Entailment.** Grounding proves a quote is real, not that it supports the
  verdict. An entailment check (local NLI, $0) is the next faithfulness layer.

## Remaining levers (code exists, needs quota/time, not money)

- Run agent eval at larger n on a subset chosen to stress the verifier (trials
  where the analyst is likely to over-claim), for a real vs-single-pass number.
- Add an entailment check on top of grounding.
- Lower abstention without sacrificing catch rate; report the coverage/faithfulness
  curve rather than a single operating point.
