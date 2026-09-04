# Production readiness: what real users would break

Written 2026-09-04, against the deployed Stage A API and the tree at the
keyword-decay commit. Scope: what stands between this and traffic from people
who are not the author.

Two things shipped with this assessment (§1.1, §2.1). The rest is ranked and
unbuilt, with the reasoning recorded so it is not re-derived later.

---

## 0 — The regulatory frame, because it decides the design

The FDA's revised Clinical Decision Support guidance (final January 2026) keeps
software outside the device definition when four criteria hold. The third is the
one that shapes this system:

> the software is intended for the purpose of **enabling an HCP to independently
> review the basis for the recommendations** that such software presents, so
> that it is not the intent that the HCP rely primarily on any such
> recommendations to make a clinical decision.

TrialGuard's thesis is that criterion, implemented. Every decisive verdict
carries a verbatim span the reviewer can locate in the source, or it is marked
`unverifiable` and carries nothing. The quote-in-source highlighting is not a
nice interaction detail; it is the mechanism by which a clinician reviews the
basis rather than trusting the output.

**Consequences for anything built after this.** Do not add a feature that
produces a recommendation whose basis cannot be inspected. Do not let
`unverifiable` render as a soft "maybe" — it is the system declining to make a
claim, and collapsing it into a verdict removes the reviewability the whole
posture rests on. Do not describe output as a screening decision.

This is a research artifact and is not being submitted to anyone. The point of
recording the frame is that the cheapest way to lose the property is to add a
convenience feature that quietly discards it.

---

## 1 — Safety

### 1.1 PHI reaches the LLM and the trace store  `[SHIPPED 2026-09-04]`

The system claimed synthetic-only in a notice and enforced nothing, while the
served path traces full prompts to Langfuse. A pasted identifier left this
infrastructure and persisted in a third-party store.

`detect_phi` now refuses at the API boundary — before the LLM call, before the
trace, before the job is created. It matches the regex-reachable subset of HIPAA
Safe Harbor identifiers: SSN, email, labelled record numbers, dates of birth,
full calendar dates, phone numbers, street addresses, IPs, URLs, labelled
patient names.

Three decisions worth keeping:

- **Refuse, not redact.** Redaction means accepting real PHI, processing it, and
  storing a modified copy. That is a different product than the one described.
- **Categories in the error, never the matched text.** An error message that
  echoed the identifier would write it into the logs the gate protects.
- **Validated for zero false positives on all 184 cohort notes.** A gate that
  blocks legitimate synthetic notes gets switched off within a day, which is
  worse than no gate. This caught a real one: `born on 39th week` is a
  gestational age, and the first pattern read it as a date of birth.

**Residual risk, stated plainly.** Regexes cannot find names, and free-text
names are the most likely real identifier a user would paste. The honest
description is that this stops casual and accidental PHI, not a determined
paste. Closing it means an NER-based detector (Philter, Presidio, or a clinical
NER model); published F1 for PHI detection runs ~0.96 for commercial clinical
NLP against ~0.79 for zero-shot GPT-4o prompting, so this is a model to
integrate rather than a prompt to write.

### 1.2 Prompt injection is detected, not neutralised  `[open]`

`detect_injection` covers known signatures and the analyst prompt fences the
note as data. The residual risk is already documented in `sanitize.py`: the
patient note is itself a grounding source, so an attacker who plants plausible
evidence text in the note can produce a *grounded* wrong verdict. Grounding
proves a quote is verbatim, not that its source is trustworthy.

**Fix, when it matters:** ground decisive verdicts against trial text only, and
treat note-sourced quotes as a distinct, visibly weaker class. That is a change
to the faithfulness contract, so it needs its own A/B, not a patch.

---

## 2 — Reliability

### 2.1 Assess concurrency and pool safety  `[SHIPPED]`

Workers raised to the measured provider ceiling of 10 (1,095 calls, zero errors;
16 failed 35 of 100 on client timeouts). Worst case is 10 assess leases plus 4
retrieval fan-out against a pool ceiling of 20, and psycopg2 raises rather than
waits when exhausted, so the headroom is a correctness property.

### 2.2 Jobs die with the process  `[open, P0 for real users]`

`JobStore` is an in-process dict with TTL eviction. Fly restarts machines
routinely — deploys, host migrations, OOM. Every in-flight assess job vanishes,
and its SSE stream hangs until the client gives up. The user has paid for the
LLM calls and sees nothing.

The analyst cache softens this — a retry re-reads completed trials for free —
but the job and its event log are gone, so nothing reconnects.

**Fix:** move jobs and their events to Postgres, which already backs the analyst
cache and the spend ledger. The `cache_entries` pattern is the precedent. Add a
`Last-Event-ID` cursor to the SSE endpoint so a reconnecting client resumes
rather than restarts.

### 2.3 The corpus goes stale silently  `[open, P1]`

`scripts/refresh.py` exists and updates recruiting status in place without
re-embedding. Nothing schedules it. The review already observed a
`NOT_YET_RECRUITING` trial being served from a corpus documented as
recruiting-only, so drift is not hypothetical: users are shown trials that are
no longer enrolling.

**Fix:** schedule the refresh, and surface `last_updated` per trial in the UI so
a stale record is visible rather than implied.

### 2.4 Single region, single instance  `[open, accepted]`

API in `syd` beside Neon in `ap-southeast-2`. No redundancy. Correct for a
portfolio artifact; named so it is a decision rather than an oversight.

---

## 3 — Cost and abuse

Working today: per-IP sliding-window limits on both endpoints, a per-request
trial cap with an opt-in deep tier, and a global daily USD ledger with a
circuit breaker that is atomic in Postgres.

The gap is that limits are **per IP**, and a daily cap is **global**. One
determined client rotating IPs can exhaust the shared budget and take the demo
down for everyone until midnight. That is a denial-of-service on the wallet, and
the only real fix is identity — Stage B auth with per-user quota. Until then the
cap is doing its job: the failure mode is a dark demo, not a surprise invoice.

---

## 4 — Observability

`served_monitor.py` reads the `served`-tagged traces and diverges when
abstention or grounding-failure rates leave their committed bands. It is
committed but **dormant**: the workflow skips until `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` exist as repo secrets. Both workflows are SHA-pinned, so
the supply-chain path that made storing them uncomfortable is closed.

Nothing alerts on API errors, latency, or budget exhaustion — only on model
behaviour drift. A first-pass fix is Fly's own metrics plus an alert on the
`/api/health` fields the endpoint already returns.

---

## 5 — Ranked

| | item | why now |
|---|---|---|
| **P0** | ~~PHI refusal at the boundary~~ | **shipped** — the only item that was actively unsafe |
| **P0** | Postgres-backed jobs + SSE resume | a restart currently loses paid work with no recovery |
| **P1** | Scheduled corpus refresh + visible `last_updated` | users are shown trials that stopped enrolling |
| **P1** | Arm the served monitor | the instrument exists and reads nothing |
| **P2** | NER-based PHI detection | closes the names gap regexes cannot reach |
| **P2** | Stage B auth + per-user quota | the only real answer to shared-budget exhaustion |
| **P2** | Error/latency alerting | currently only model drift is watched |
| — | Multi-region | correctly out of scope |

The honest summary: after §1.1 the system is safe to show people, and it is not
yet reliable enough to depend on. The difference is §2.2 — losing a user's paid
work on a routine deploy is the failure a real user would hit first and forgive
least.
