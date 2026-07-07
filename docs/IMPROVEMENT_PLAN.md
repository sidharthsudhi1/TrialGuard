# TrialGuard — Critical Assessment & Improvement Plan

Generated: 2026-07-07. Grounded in the codebase as of `phase/2-retrieval`
(Phase 0-2 complete, Phase 3 agent eval harness next).

Scope: close the retrieval gap, build the agent correctly, make hallucination
measurable, keep cost at $0. No paid APIs, no hosted GPUs.

---

## 0. Key discovery: the recall@10 target is miscalibrated

The recall@10 >= 0.90 target is mathematically impossible on two of three cohorts.
Recall@10 is capped at `min(10, |gold|) / |gold|` per patient. Measured from qrels:

| Cohort | Patients | Median eligible/patient | Max possible recall@10 |
|---|---|---|---|
| sigir | 53 | 6 | 0.897 |
| trec_2021 | 75 | 62 | **0.247** |
| trec_2022 | 50 | 59.5 | **0.285** |

TrialGPT's ">90% retrieval recall" was measured at large candidate depth, not @10.
The metric that actually matters is **recall@pool** — the fraction of gold trials
inside the candidate pool handed to the agent. Current standing (SIGIR, keyword
config): recall@100 = 0.69 raw / 0.72 coverage-adjusted, recall@200 = 0.81 adjusted.

**Real gap: recall@100 from ~0.72 to ~0.85-0.90. Fixable at $0.**

---

## 1. Critical assessment

### 1.1 Retrieval

**Most important problem: wrong target metric, plus two concrete signal losses.**

1. **Exclusion criteria are never indexed.** `eligibility_text_for_embedding`
   (`src/trialguard/ingestion/embed.py:59`) builds doc text from title +
   inclusion criteria only. BM25 tokenization (`src/trialguard/eval/file_index.py:89`)
   does the same. Exclusion text carries disease mentions, biomarkers, and
   prior-treatment signal — half the eligibility text is discarded at index time.
2. **Wrong-domain models.** BGE is general-domain. The ms-marco cross-encoder
   rerank actively hurt (recall@50 delta = -0.173, see `data/reports/phase2_rerank.md`).
   MedCPT (NCBI, free on HF, CPU-friendly, 110M params) is trained on PubMed
   search logs and is what TrialGPT's own retriever uses.
3. **Eval scope hole.** Only SIGIR (n=53) has been evaluated; TREC indexes were
   never built (`data/indexes/` contains sigir only). One cohort at n=53 is not
   defensible. Additionally, the SIGIR eval corpus is built from TrialGPT's
   `retrieved.json` — a corpus pre-filtered by *their* retriever — so recall
   numbers ride on their retrieval. Must be disclosed in any report.

Consequence if unfixed: chasing an impossible number, shipping a weak agent
pool, and a benchmark chapter that falls apart under scrutiny.

### 1.2 Agent design

**Most important problem: the planned Verifier is the same LLM checking itself.**
Same llama-3.3-70b, same blind spots. A model that hallucinated "ECOG 0-1
required" plausibly re-hallucinates it on re-read. Correlated errors mean
hallucinations get stamped GROUNDED, and the faithfulness thesis dies quietly.

Missing piece: a **deterministic grounding check**. The Analyst must emit exact
quote spans from trial text; a Python substring match verifies the quote exists —
zero LLM calls, zero cost, cannot hallucinate agreement. The LLM (or local NLI)
verifier then only judges entailment (does the quote support the verdict), not
existence. Without this split, the GROUNDED stamp is an opinion.

Second gap: **abstention gaming is unmeasured.** "cannot determine" as a
first-class output is correct, but a system that abstains on everything is
perfectly faithful and useless. Coverage/decision-rate must be reported jointly
with faithfulness.

### 1.3 Eval harness

**Most important problem: no criterion-level gold wired in, no hallucination
metric defined, no baseline arm.** qrels are trial-level only, so the
"criterion-matching accuracy >= 87%" target is currently unmeasurable. The
TrialGPT repo ships ~1,015 manually reviewed patient-criterion annotations that
`cohorts.py` does not download. CLAUDE.md demands the hallucination rate "beat
single-pass baseline (measured, not assumed)" — no single-pass arm exists.

Metrics absent that the "self-verifying" claim requires:

- **Citation precision** — fraction of quoted spans that exist verbatim in the
  source trial text (deterministic check).
- **Unsupported-verdict rate** — verdicts lacking a grounded citation. This is
  the hallucination rate.
- **Verifier catch rate** — deliberately inject corrupted citations, measure
  detection. Without this the verifier's own recall is unknown.
- **Abstention rate + coverage** — reported jointly with accuracy (anti-gaming).
- **Single-pass baseline** — Analyst-only, same prompts, no verifier loop.

Free bonus sitting unused: qrels score 1 = excluded, 2 = eligible gives
trial-level eligible-vs-excluded classification gold, already downloaded.

### 1.4 Cost structure

**Most important problem: eval-time agent calls exceed Groq free tier by ~40x.**
Groq free tier for llama-3.3-70b is roughly 30 req/min, 1k req/day, ~100k
tokens/day (verify current limits). The math: 53 patients x 50-trial pool x
(analyst + verifier) = 5,300 calls minimum at ~1.5-2k tokens each, roughly
8-10M tokens — 80-100 days of token quota for one eval run. Worst-case retries
(cap 2) multiply by up to 3. Per-criterion calls (10+ criteria per trial) would
be a further 10x — never do this.

Hidden costs:

- Verifier retry loop doubles-to-triples call volume.
- Keyword extraction silent fallback (`query_transform.py:70-71`) re-invokes the
  LLM on every call under persistent failure, and masks errors from eval results.
- Rerank cache keyed by note hash only, not model/prompt version — stale cache
  silently masks changes. Keyword cache has the same problem.

Mandatory mitigations: batch all criteria per trial into ONE call; deterministic
verifier (zero LLM); LLM response cache keyed `(patient_hash, nct_id,
prompt_version)`; iterate on a subset (10 patients x 20 trials), full run paced
overnight.

### 1.5 Architectural risks

1. **Verifier-analyst correlation** (see 1.2) — the single biggest thesis threat.
2. **Groq single-provider SPOF** — already ate one model decommission
   (llama-3.1-70b); free-tier limits can change any week. LLM response cache +
   a thin provider seam mitigate.
3. **Unversioned prompt caches** — keyword cache files are keyed by note hash
   only; a prompt improvement will not invalidate them, and eval silently
   measures stale keywords.

---

## 2. Improvement plan (ordered by impact on faithfulness + measurability)

| # | Change | Why it is the right lever | $0? | Depends on |
|---|---|---|---|---|
| 1 | **Deterministic citation verifier**: Analyst outputs exact quotes; Python substring check; fail -> retry (cap 2) -> unverifiable | Makes faithfulness mechanical, not model-opinion. Core thesis mechanism | Yes — pure Python | Analyst JSON schema with `quote` field |
| 2 | **Redefine retrieval target**: recall@pool(100) primary; document the @10 ceiling math in the report | Honest, defensible metric; stops chasing an impossible number | Yes | Nothing |
| 3 | **Phase 3 harness metrics**: citation precision, unsupported-verdict rate, abstention+coverage, criterion accuracy vs TrialGPT's 1,015 annotations, trial-level eligible/excluded vs qrels | "Self-verifying" claim is unmeasurable without these | Yes — data already public | TrialGPT annotation download in `cohorts.py` |
| 4 | **Single-pass baseline arm** in harness | CLAUDE.md non-negotiable: "measured, not assumed" | Yes (fits quota via #7) | #3 |
| 5 | **Swap BGE -> MedCPT encoders**, re-embed | Domain-exact model, TrialGPT's own choice; best shot at closing the recall gap | Yes — HF free, CPU | Re-embed cache (~hours CPU) |
| 6 | **Index exclusion criteria** in dense + BM25 doc text | Recovers discarded signal; one-line change each | Yes | Re-embed (bundle with #5) |
| 7 | **LLM response cache** for analyst/verifier keyed (patient, trial, prompt_version); version keyword + rerank caches too | Only way eval fits free tier; makes runs reproducible | Yes | Nothing |
| 8 | **Run TREC 2021/2022 evals** | n=53 one-cohort -> n=178 three-cohort defensibility | Yes — CPU embed time only | #5/#6 first (avoid double re-embed) |
| 9 | **Verifier-corruption test**: perturb citations, measure catch rate | Proves the verifier works vs rubber-stamps | Yes | #1, #3 |
| 10 | Retry rerank with MedCPT cross-encoder | ms-marco failed; domain reranker may flip the result; only after encoders fixed | Yes | #5, #8 |
| 11 | Optional: local NLI model (DeBERTa-MNLI, CPU) for entailment check | Model diversity vs Groq self-agreement, at $0 | Yes | #1 |

---

## 3. The 3 highest-leverage changes

### 3.1 Deterministic quote grounding — `src/trialguard/verify/grounding.py` (new)

Analyst JSON per criterion: `{criterion, verdict, quote, nct_id}`. Grounding
check: normalized substring match of `quote` against stored trial text. No
match -> route back to Analyst (max 2) -> `unverifiable`.

**Buys:** hallucinated citations become impossible to pass silently;
faithfulness is measured by code, not judgment; verifier LLM cost drops to
near-zero. This is the thesis, mechanized.

### 3.2 Agent metrics + criterion gold + baseline arm — `src/trialguard/eval/agent_metrics.py` (new)

Add TrialGPT criterion-level annotation download to `cohorts.py`. Metrics:
criterion accuracy (vs ~1,015 gold pairs), citation precision,
unsupported-verdict rate, abstention rate, trial-level accuracy (vs qrels 1/2
labels) — all reported for `single-pass` vs `verified` arms side by side.

**Buys:** the headline result — "verification cut unsupported verdicts from X%
to Y% at Z% coverage cost." Without this, the project has architecture but no
proof.

### 3.3 MedCPT + exclusion criteria — `src/trialguard/ingestion/embed.py`

Swap to `ncbi/MedCPT-Query-Encoder` / `ncbi/MedCPT-Article-Encoder` (768-dim,
drop-in match for the existing pgvector schema). Doc text = title + inclusion +
exclusion. Rerun `eval_retrieval.py --all-cohorts`.

**Buys:** best available shot at recall@100 >= 0.85 across three cohorts — a
bigger, better-grounded agent pool and a retrieval chapter that survives
comparison with TrialGPT, still at $0.

**Priority rule:** 3.1 and 3.2 prove the thesis; 3.3 stops garbage-in at the
pool stage. Skip reranking, UI, and additional cohorts until these three land.
