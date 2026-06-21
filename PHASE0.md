# Phase 0 — Foundations: Running Doc

**Goal:** Clean repo, working environment, secrets skeleton, tracing skeleton. One-command test suite passes. README with architecture diagram present.

---

## Checklist

### Repo & structure
- [x] Create `/Documents/TrialGuard/` with `src/`, `data/`, `tests/`, `notebooks/`
- [x] `.gitignore` (excludes `.env`, raw data, notebooks)
- [x] `pyproject.toml` with all phase 1–4 deps declared upfront
- [x] `README.md` with problem, thesis, Mermaid architecture diagram, phase table, AD log
- [ ] `git init` + initial commit
- [ ] Push to GitHub (public repo — becomes the portfolio artifact)

### Python environment
- [x] Python version confirmed: **3.12.3** (user said 3.11, actual is 3.12 — no issue)
- [ ] `python -m venv .venv` created
- [ ] `pip install -e ".[dev]"` succeeds
- [ ] `pytest tests/` passes (smoke test: config defaults)

### Secrets & config
- [x] `.env.example` with all keys documented
- [x] `src/trialguard/config.py` — pydantic-settings, loads from `.env`
- [ ] User copies `.env.example` → `.env` (never committed)

### Tracing skeleton
- [x] `src/trialguard/tracing.py` — `trace_span()` context manager, no-op fallback when keys absent
- [ ] Verify no-op path works (run `python -c "from trialguard.tracing import trace_span"`)

### Free-tier accounts (user action required)
- [ ] **Langfuse** — sign up at https://cloud.langfuse.com → get Public + Secret key → add to `.env`
- [ ] **Groq** — sign up at https://console.groq.com → get API key → add to `.env`
- [ ] **Hugging Face** — sign up at https://huggingface.co → get token → add to `.env` (needed Phase 6)
- [ ] **Neon** (pgvector) — sign up at https://neon.tech → create DB `trialguard` → get `DATABASE_URL` → add to `.env` (needed Phase 1)

### Definition of done for Phase 0
All boxes checked, `pytest tests/` green, README renders on GitHub with architecture diagram visible.

---

## Decisions made in Phase 0

| Item | Decision | Reason |
|---|---|---|
| Condition class | **Oncology** | Richest trial volume, best overlap with SIGIR/TREC eval cohorts |
| Python version | 3.12.3 | Actual installed version; fully compatible with all deps |
| LLM backend | Groq (Llama 3.1 70B) | Free, fast, swappable — model is a pluggable backend |
| pgvector host | Neon free tier | SQL-native, zero cost, scales to our corpus |

---

## Questions / blockers

*(Update this section as questions arise during the phase.)*

| # | Question | Status | Resolution |
|---|---|---|---|
| 1 | GitHub: new repo or existing? | **Open** | Need user to create public repo and push |
| 2 | Neon vs Supabase for pgvector? | **Open** | Both free; Neon preferred (Postgres-native, no ORM lock-in) |

---

## Next: Phase 1

Once all Phase 0 boxes checked:
- Pull ~2,000–5,000 oncology trials from ClinicalTrials.gov v2 API
- Normalise eligibility text; handle empty arrays and inconsistent dates
- Embed + load into pgvector
- Download and parse SIGIR/TREC gold cohorts

See `PHASE1.md` when ready.
