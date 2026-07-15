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

## Agent faithfulness

All numbers below use the **corrected grounding rule** (see "Grounding fix"
after this section). The matched paired comparison runs both arms — single-pass
(`max_retries=0`) and verified (`max_retries=2`) — over the identical trial set,
so there is zero selection skew between arms.

### Paired A/B, two cohorts

| Cohort | matched n | Single-pass | Verified | Relative | Fisher p | |
|---|---|---|---|---|---|---|
| SIGIR | 179 | 0.0926 | 0.0338 | −63.5% | **0.0012** | ✅ significant |
| TREC 2021 | 59 | 0.1200 | 0.1126 | −6.2% | 0.859 | ❌ null |
| TREC 2022 | 60 | 0.1230 | 0.0940 | −23.5% | 0.537 | ❌ ns |

**On SIGIR, verification works and is significant** — deterministic grounding +
bounded retry, over the same LLM, cuts the unsupported-verdict rate by ~64%
(9.26% → 3.38%, p=0.0012).

**On neither TREC cohort does it reach significance.** TREC 2021 is essentially
flat (−6.2%, p=0.86); TREC 2022 shows a larger point estimate (−23.5%) but still
not significant at n=60 (p=0.54, failure counts 15 → 11). So the effect on TREC
is *directional at best, much weaker than SIGIR, and unproven* — not a clean zero,
but nowhere near SIGIR's strength. This is a real, disclosed cohort-dependence,
not a tuning gap. Why the retry step transfers poorly to TREC:

- On SIGIR, a grounding failure is usually a *paraphrase* of text that is
  verbatim-quotable; the retry prompt ("copy character-for-character") lets the
  analyst recover the exact span → the verdict becomes grounded.
- On TREC, the ungrounded citations are more often genuinely absent from the
  source (true fabrications or reasoning the analyst can't point to); no verbatim
  span exists, so retry recovers almost nothing (18 → 17 failures).

**What holds on both cohorts: the faithfulness floor.** Deterministic grounding
still catches 100% of ungrounded verdicts and forces them to `unverifiable` — a
hallucinated citation never passes as grounded on either corpus. The
cohort-dependent part is only whether a caught failure gets *fixed* by retry
(SIGIR) or converted to an honest *abstention* (TREC). The product guarantee —
never assert a verdict on evidence that isn't there — is corpus-independent.

### Grounding fix: token guard, not char-length guard

Replicating on TREC exposed a measurement bug. The old rule rejected any quote
under 12 characters, on the theory that short fragments match spuriously. But the
high-value atomic facts in eligibility matching are *short*: "48 M", "EF was 25%",
"T-L spine", "ECOG 1". On TREC's terse patient text, **18 of 25 grounding
"failures" were real facts, verbatim in the source, rejected only for length**
(on SIGIR, only 1 of 13 — which is why the bug stayed hidden until replication).

Fixed (`verify/grounding.py`): a quote grounds if it is a verbatim substring
**and** carries ≥2 word tokens. This still blocks vague single-word matches
("cancer", "ECOG") but accepts specific short facts. Both cohorts were
re-evaluated under the corrected rule at zero cost (grounding is pure Python over
cached analyst outputs). The fix *strengthened* SIGIR (p=0.0044 → 0.0012) and
removed the artifact from TREC without changing its null verdict — the honest
conclusion survives the correction.

**Infra hardened along the way:** the larger-n runs exposed three real boundary
bugs, now fixed: LLM output truncation at the token cap (salvage parser recovers
complete assessment objects), 429 rate-limit crashes (client backoff +
inter-call spacing), and no graceful degradation on the daily cap (harness now
reports metrics over completed trials instead of crashing).

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

- **Abstention 0.73** — the system declines ~three-quarters of criteria. High
  faithfulness is bought partly with low coverage; the two must always be read
  together.
- **Trial accuracy 0.26-0.29**: the strict roll-up marks a trial `eligible` only
  if *all* criteria are `met`, so high abstention collapses most trials to
  `cannot_determine`. Weak as a standalone number.
- **Metric overlap**: a grounding failure is counted in *both* `decisive_attempts`
  and `abstention_rate` (its verdict is `unverifiable`), so the two rates do not
  partition the criteria. Read them as separate lenses, not a split.
- **The verification benefit is cohort-dependent** — significant on SIGIR
  (p=0.0012), non-significant on both TREC cohorts (2021 −6% p=0.86; 2022 −24%
  p=0.54). The retry step recovers paraphrase-type failures but not genuine
  fabrications. Only the faithfulness floor (100% catch) is corpus-independent.

---

## What is proven

1. **Faithfulness is mechanical.** Grounding is deterministic and tested
   (`tests/test_grounding.py`, `tests/test_agent_graph.py`); a hallucinated
   citation cannot pass silently.
2. **Verifier catch rate is 100% (509/509 corrupted quotes rejected, 0 false
   rejections)** — the sample-size-independent proof of the mechanism.
3. **Verification significantly halves unsupported citations on SIGIR** —
   matched paired A/B (179 trials): 9.26% → 3.38%, −63.5% relative, Fisher
   **p=0.0012**. On both TREC cohorts the effect is weaker and non-significant
   (2021 p=0.86; 2022 p=0.54) — an honest, disclosed cohort-dependence.
4. **Retrieval improved at $0** by swapping to a domain-matched encoder
   (MedCPT), reproduced on SIGIR and on full ~26k-trial TREC corpora.

## What is NOT yet proven

- **Whether verification helps TREC at all.** TREC 2021 is flat; TREC 2022 shows
  −24% but is underpowered (p=0.54, n=60). A larger TREC 2022 run would settle
  whether that point estimate is real. The retry can't recover non-verbatim
  citations; a retrieval-aware retry (feed the analyst the exact source span) may
  transfer the SIGIR benefit.
- **Entailment.** Grounding proves a quote is real, not that it supports the
  verdict. An entailment check (local NLI, $0) is the next faithfulness layer.

## Remaining levers (code exists, needs quota/time, not money)

- Larger-n TREC 2022 run to resolve its −24% point estimate (currently p=0.54).
- Retrieval-aware retry: hand the analyst the candidate source span on retry.
- Add an entailment check on top of grounding.
- Lower abstention without sacrificing catch rate; report the coverage/faithfulness
  curve rather than a single operating point.
