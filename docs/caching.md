# Caching & rate-limit story (Phase 5, WS-3)

The $0 constraint is engineering, not luck. Three disk caches plus a daily token
budget mean a full multi-day eval spends each fresh Groq token once, resumes
across days, and lets the CI regression gate run offline for free.

## The three caches

| Cache | Path | Key | What a miss costs | Invalidation |
|---|---|---|---|---|
| Analyst outputs | `data/cache/analyst/{key}.json` | `sha256(prompt_version \| nct_id \| patient_note)[:20]` | 1 Groq call | Additive only — a new `TG_PROMPT_VERSION` gets a new namespace; **v1 is frozen** (backs the Phase 3/4 numbers) and is never invalidated |
| Retrieval keywords | `data/cache/keywords/{note_hash}.json` | note hash | 1 Groq call | Stable; regenerate by deleting the file |
| Eval embeddings | `data/indexes/{source}_{tag}_embeddings.npy` | source + index tag | full MedCPT re-embed of the corpus (minutes, no Groq) | Set `TG_INDEX_EXCLUSION=0` for TREC or it re-embeds 26k trials every run |

All three are gitignored (`data/cache/`, `data/indexes/`), so they never bloat the
repo — and so CI cannot depend on them. That is why the regression gate runs on
committed artifacts only (reports + golden fixture), not on a cache replay.

## Daily token budget

`agent/ratelimit.py` holds every rate-limit knob (`TPM_CAP`, `TPD_CAP`,
`MAX_RETRIES`, `ANALYST_DELAY`) that used to be an inline magic number.

- **Backoff:** the analyst's ChatGroq client keeps `max_retries=MAX_RETRIES` so a
  429 is retried honoring `Retry-After`; fresh calls sleep `ANALYST_DELAY` to stay
  under the ~12k TPM window.
- **`TokenBudget`:** persists today's spend to `data/cache/groq_budget.json`. Before
  a fresh call the analyst calls `check(estimate)`; if it would cross `TPD_CAP`
  (~100k), it raises `BudgetExhausted` instead of hammering the API into 429s.
  After the call it records the response's real `usage_metadata.total_tokens`. The
  count resets on date rollover, so a run spanning days resumes against the correct
  daily budget.

## Graceful degradation

Two signals collapse to one behavior — stop spending, report what completed:

- **Local, pre-emptive:** `BudgetExhausted` from the budget gate.
- **Server, reactive:** a `429` / `rate_limit` after the client's retries.

`eval/agent_metrics._run_arm` catches both, sets `rate_limited=True`, and returns
metrics over the trials that finished rather than crashing. Set `TG_CACHED_ONLY=1`
to force pure-cache mode — no fresh call is ever made, which is what lets every
cohort regenerate its significance test and coverage curve from cache at $0, and
is the same mode the CI gate relies on.
