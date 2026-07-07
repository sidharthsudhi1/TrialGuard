# TrialGuard — Development Plan

**A self-verifying, multi-agent clinical-trial eligibility intelligence system.**

This is a high-level engineering plan: it defines *what* gets built in each phase, the *architectural decisions* that shape the system, and the *research* behind each decision. It deliberately avoids implementation detail (no code, no function signatures) so it stays useful as a north star rather than a spec that rots.

---

## 1. Problem & Premise (one-paragraph recap)

Matching patients to clinical trials is a real, validated, and largely unsolved bottleneck. The NIH's own 2024 *TrialGPT* work (published in *Nature Communications*) and Mass General Brigham's *RECTIFIER* trial both demonstrated that LLM-assisted screening can roughly double enrolment rates and cut screening time by ~40%. The dangerous failure mode is an AI that confidently declares a patient *eligible* based on a hallucinated or misread eligibility criterion — a patient-safety issue, not a UX annoyance. **TrialGuard's thesis: faithfulness is the product.** Every eligibility verdict is either backed by a verified citation from the source trial, or explicitly flagged as unverifiable.

---

## 2. Data Source Research

### 2.1 Primary source — ClinicalTrials.gov API v2

| Property | Finding | Implication for design |
|---|---|---|
| Coverage | 500,000+ studies, 221 countries | More than enough; we scope to one condition class |
| Access | Public REST API, **no key, no auth, no signup** | Zero cost, zero credential management |
| Format | JSON (primary), CSV, OpenAPI 3.0 spec | Clean ingestion, predictable schema |
| Base endpoint | `/api/v2/studies` with `query.cond`, `query.intr`, `filter.overallStatus`, `filter.phase` | Lets us pull a focused, filtered corpus |
| Rate limit | ~50 requests/min (generous; 429 on breach) | Respect with ~1.5s spacing; batch ingestion offline |
| Pagination | Cursor / page-token based | Standard loop; cache pages locally |
| Eligibility data | Lives in `eligibilityModule` (inclusion/exclusion criteria, age, sex, healthy-volunteer flag) | This is our core reasoning target |
| Enumerated fields | Status & phase are fixed enums (e.g. `RECRUITING`, `PHASE3`) — case-sensitive | Reliable filtering, no fuzzy parsing |
| Known data-quality traps | Arrays can be empty/null (conditions, locations); dates are inconsistently formatted; rich text is CommonMark | Defensive ingestion + a normalisation step are mandatory, not optional |
| History | v2 launched 2024; classic API retired June 2024 | Build only against v2 |

**Decision:** ClinicalTrials.gov API v2 is the sole live data source. It is free, real, large, structured-yet-messy, and high-stakes — exactly the profile that makes the project both feasible at $0 and credible to a hiring manager.

### 2.2 Evaluation data — reuse published labelled cohorts (major find)

We do **not** hand-label an eval set from scratch. The NIH TrialGPT team released the labelled cohorts they used, all publicly downloadable:

- **SIGIR 2016** patient–trial matching cohort — labels: *irrelevant / potential / eligible*.
- **TREC Clinical Trials 2021** corpus.
- **TREC Clinical Trials 2022** corpus.
- Combined: **183 synthetic patients** (physician-authored from real medical data) and **75,000+ trial eligibility annotations**, plus a manually reviewed set of ~1,015 patient–criterion pairs.

**Decision:** Use these published cohorts as the gold standard for the eval harness. This grounds our metrics in a peer-reviewed benchmark, lets us compare against a known baseline (TrialGPT reported ~87% criterion-matching accuracy and >90% retrieval recall), and removes the single biggest solo-project risk: producing trustworthy ground truth. We supplement with a small hand-built set only for adversarial/"cannot determine" cases the published data underweights.

### 2.3 Licensing & ethics note

ClinicalTrials.gov is U.S. public-domain government data. The TREC/SIGIR cohorts are research datasets with citation requirements (cite the original papers). All patient profiles used in demos are **synthetic** — no real patient data ever enters the system. This is stated plainly in the README and is itself a maturity signal.

---

## 3. Architectural Decisions

Each decision is recorded as: **what we chose → what we considered → why**. The research backing the choices is in §2 and the inline notes below.

### AD-1 — Orchestration: LangGraph (not a linear chain)
- **Considered:** plain LangChain LCEL chain; LlamaIndex query engine; bespoke Python control flow.
- **Why LangGraph:** the core differentiator is a *verifier that routes back to the analyst* when it finds an ungrounded claim. That is a cyclic, stateful graph with a conditional edge and bounded retries — precisely what LangGraph models natively and what a linear chain cannot express. It also produces a diagram you can whiteboard in an interview.

### AD-2 — Retrieval: hybrid (dense + lexical) with reciprocal rank fusion
- **Considered:** dense-only semantic search; keyword-only (BM25); ColPali-style visual retrieval.
- **Why hybrid:** trial eligibility text mixes semantic concepts ("HER2-positive" ≈ "HER2 overexpression") with exact tokens (drug names, lab thresholds, ECOG scores) where lexical match matters. The NIH TrialGPT work used exactly this pattern — hybrid retrieval fused with reciprocal rank fusion — and reported >90% recall. We follow the proven approach rather than inventing one. (ColPali was dropped earlier: its multi-vector embeddings are impractical on free-tier GPUs and the data is text, not page-images.)

### AD-3 — Two-pass "analyst → verifier" reasoning
- **Considered:** single-pass eligibility judgment; self-consistency voting; chain-of-thought only.
- **Why two agents:** separation of generation and verification is what drives the hallucination rate down. The **Analyst** drafts a criterion-by-criterion assessment with citations; an independent **Verifier** re-reads the source and either stamps each claim `GROUNDED` or returns a structured rejection that loops back. Bounded to two retries to avoid infinite loops (a real production failure mode).

### AD-4 — Structured, criterion-level output (not free-text answers)
- **Considered:** free-text natural-language verdict; binary eligible/ineligible flag.
- **Why criterion-level JSON:** matching the literature's unit of analysis (the patient–criterion pair) makes the system *evaluable*, *citable*, and *honest* — each criterion carries its own assessment, citation, and confidence, and "cannot determine" is a first-class outcome.

### AD-5 — Embeddings: small open sentence-transformer, run on CPU
- **Considered:** OpenAI/hosted embedding APIs; large open embedding models on GPU.
- **Why small + local:** keeps cost at $0 and the pipeline reproducible by anyone cloning the repo. Embedding a few-thousand-trial corpus is a one-time offline job; a compact model (e.g. MiniLM-class) is adequate for first-stage recall, with reranking carrying the precision load.

### AD-6 — Vector store: pgvector on a managed free tier (Postgres)
- **Considered:** Pinecone; Qdrant; FAISS flat file.
- **Why pgvector:** one datastore for both vectors and trial metadata, SQL-native, free tier sufficient for the scoped corpus, and a defensible "I understand infra" choice. We still *benchmark* against a managed alternative and write up the trade-off rather than just asserting one.

### AD-7 — LLM inference: free hosted open models (e.g. Groq free tier)
- **Considered:** local quantised LLM on free GPU; paid frontier API.
- **Why hosted free open models:** fast enough for an interactive demo, $0, and swappable. The architecture treats the model as a pluggable backend so we can later benchmark open vs. frontier quality on the same harness. Eval-time LLM-as-judge calls are **cached to disk** so repeat eval runs cost zero requests and never hit rate limits.

### AD-8 — Observability: tracing from day one (Langfuse free tier)
- **Considered:** add logging later; print statements.
- **Why early:** every agent step, retry, and rejection is traced. This is the LLMOps story hiring managers ask for, and it's how we debug the verifier loop. Retrofitting tracing is painful; building it in is nearly free.

### AD-9 — Compute: Kaggle/Colab free notebooks for batch jobs only
- **Considered:** always-on GPU; local machine only.
- **Why notebooks for batch:** indexing/embedding is bursty and offline — perfect for free 30h/week notebook GPU. The live demo runs CPU-only inference against hosted model APIs, so no GPU is ever kept warm. Model artifacts persist to cloud storage between sessions.

### AD-10 — Serving & demo: Gradio on Hugging Face Spaces
- **Considered:** FastAPI + custom React; Streamlit; local-only.
- **Why HF Spaces:** free, public, clickable URL for recruiters, and the natural home for an ML demo. A thin FastAPI layer can be added if an API surface is needed, but the priority is a shareable artifact, not a product.

---

## 4. Development Phases

Each phase ends with a **shippable artifact** so the project is presentable even if later phases slip. Phases are sequenced so the *eval harness exists before the agent is tuned* — you cannot improve what you cannot measure.

### Phase 0 — Foundations & scoping
- Pick one condition class (recommend **oncology** — richest trial volume, clearest stakes, best overlap with published eval cohorts).
- Stand up the repo, environment, secrets handling, tracing skeleton, and free-tier accounts.
- **Artifact:** clean repo + a README stating the problem, thesis, and architecture diagram.

### Phase 1 — Data ingestion & corpus build
- Pull a focused corpus (~2,000–5,000 trials) from the v2 API with defensive handling of empty arrays and inconsistent dates; normalise eligibility text.
- Embed and load into pgvector; verify retrieval sanity by hand.
- Download and parse the SIGIR/TREC cohorts into a common internal format.
- **Artifact:** queryable trial corpus + parsed gold eval cohorts.

### Phase 2 — Retrieval pipeline
- Implement hybrid dense + lexical retrieval with reciprocal rank fusion and a reranking step.
- Measure recall@k and MRR against the gold cohorts; target parity with the published ~90% recall benchmark.
- **Artifact:** a measured retriever with a recall/latency report.

### Phase 3 — Eval harness *(built before tuning)*
- Define the metric suite: eligibility accuracy, criterion-citation accuracy, hallucination rate, verifier catch rate, correct-refusal rate, latency (median/p95).
- Wire LLM-as-judge with on-disk caching; add a regression gate that fails on metric drops.
- Establish a **single-pass baseline** to beat.
- **Artifact:** one-command eval run + baseline numbers.

### Phase 4 — The agent (Analyst → Verifier loop)
- Build the LangGraph graph: Planner → Retriever → Eligibility Analyst → Verifier (conditional back-edge, max 2 retries) → Ranker/Reporter.
- Enforce structured criterion-level output and the explicit "cannot determine" path.
- Iterate against the harness; demonstrate the verifier driving hallucination rate down vs. baseline.
- **Artifact:** working end-to-end agent + before/after metrics proving the verifier earns its place.

### Phase 5 — LLMOps & hardening
- Full tracing dashboards (cost, latency, retry counts, rejection reasons).
- Caching, graceful rate-limit handling, prompt versioning in git, the CI-style eval gate.
- The pgvector-vs-managed benchmark write-up.
- **Artifact:** an observable, regression-gated system with a documented ops story.

### Phase 6 — Demo, docs & narrative
- Deploy Gradio demo to HF Spaces; record a 3-minute walkthrough.
- Write the README as a case study: problem → thesis → architecture → results → cost. Include the architecture diagram, metric tables, and trace screenshots.
- A short "cost engineering" section documenting the $0 stack as a deliberate constraint.
- **Artifact:** public live demo + portfolio-ready repo + recorded demo.

---

## 5. Risks & Mitigations (high level)

| Risk | Mitigation |
|---|---|
| Eligibility criteria too ambiguous to label reliably | Lean on published gold cohorts; treat ambiguity as a first-class "cannot determine" outcome rather than forcing a verdict |
| Free-tier LLM rate limits during eval | Cache all judge calls to disk; batch eval runs; run overnight on notebooks |
| Verifier loop fails to converge | Hard cap retries at 2; on exhaustion, mark unverifiable and surface to user |
| Scope creep (whole registry, many conditions) | Lock to one condition class in Phase 0; breadth is a stretch goal, not a requirement |
| Notebook session resets lose work | Persist all model artifacts/checkpoints to cloud storage each session |

---

## 6. Definition of Done

The project is portfolio-complete when:
1. A recruiter can click a live demo and get a cited, ranked trial shortlist for a synthetic patient.
2. The repo shows a measured reduction in hallucinated criteria from the single-pass baseline, on a published benchmark.
3. The eval harness runs in one command and gates regressions.
4. Tracing, the architecture diagram, and a documented cost story are all present.
5. Total running cost is **$0/month**.

---

## 7. Key References (for the README)

- Jin et al., *Matching patients to clinical trials with large language models* (TrialGPT), *Nature Communications*, 2024 — methodology, hybrid retrieval + RRF, benchmark numbers, public datasets.
- NIH/NLM TrialGPT program page and dataset release (SIGIR 2016, TREC CT 2021/2022).
- Mass General Brigham *RECTIFIER* randomised trial — real-world evidence that AI screening ~doubles enrolment.
- ClinicalTrials.gov API v2 documentation (NLM Technical Bulletin, 2024).

*This is a living plan. Architectural decisions are versioned; if one is reversed during the build, record the reversal and the reason rather than deleting it — that decision log is itself portfolio material.*
