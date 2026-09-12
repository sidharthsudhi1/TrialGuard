# Production readiness: what real users would break

Written 2026-09-04, against the deployed Stage A API and the tree at the
keyword-decay commit. **Revised 2026-09-07** after Phase 10 built most of it.
Scope: what stands between this and traffic from people who are not the author.

Two things shipped with the original assessment (§1.1, §2.1). Phase 10 closed
§2.2, §2.3 and §4, measured §1.2 and rejected its proposed fix on evidence, and
gave §1.3 — the entailment limit — its first number. Each section carries its own
status tag; §5 is the ranked list as it now stands. Where a fix was rejected
rather than deferred, it is struck from the ranking rather than left implying
future work.

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

### 1.2 The note is a grounding source  `[measured 2026-09-07; fix rejected, risk restated]`

`detect_injection` covers known signatures and the analyst prompt fences the
note as data. The residual risk documented in `sanitize.py` stands: the patient
note is itself a grounding source, so evidence planted in it produces a
*grounded* wrong verdict. Grounding proves a quote is verbatim, not that its
source is trustworthy.

The proposed fix — ground decisive verdicts against trial text only — was A/B'd
on both cohorts and **rejected**. 84% of grounded quotes exist only in the note,
because the evidence that a patient satisfies a criterion lives in the patient
record while the trial states the requirement. Enforcing it took the criterion
unverifiable rate from 3.1% to 42.6% (SIGIR) and 3.5% to 50.0% (TREC) and
emptied the `eligible` tier. It also keeps the wrong survivors: of the 8
trial-grounded quotes on SIGIR, 2 restate their own criterion and 6 quote a
different one, so none is evidence about a patient.

**The risk is therefore restated, not closed.** A quote is verified to appear
verbatim *in the inputs*, and for patient facts the input is the user's note. A
user who states false facts receives verdicts faithful to those false facts —
garbage-in, not hallucination, and indistinguishable to any verifier reading the
same note. Only a link to a real record could tell them apart, which is ruled out
by the rule that no real patient data ever enters this system.

What shipped is visibility: `grounded_in` on every assessment, `note_only_grounded`
on every request, `TG_GROUND_TRIAL_ONLY` kept default-off so the measurement is
reproducible. See `data/reports/ws5_guardrails_findings.md` and AD-14.

### 1.3 Grounding checks existence, not entailment  `[measured 2026-09-07, open]`

A verbatim quote can fail to establish the verdict it supports. Sampled at 50
grounded decisive verdicts (seeded, 25 per cohort, adjudicated on whether the
quote establishes the verdict rather than whether the verdict is right): **36%
do not**, 95% CI 24-50% (SIGIR 48%, TREC 24%). Not a 36% error rate — the quote
is real in every case and many verdicts are still correct — but the size of the
gap between *cited* and *shown*.

AD-3 rules out an LLM verifier for enforcement on correlated-error grounds and
that reasoning survives having measured the gap. The machine-checkable subset (a
quote restating its own criterion) is 0.12-0.37%, recorded as `self_referential`,
and is a floor rather than an estimate.

**The NLI counter-proposal was evaluated 2026-09-12 and is not shippable yet**
(AD-23, `nli_feasibility.md`). `deberta-large-mnli` ranks entailment on the
adjudicated set at AUC 0.79 but is miscalibrated to the point of uselessness:
median p(entail) is 0.027 for citations a human judged sound, so any natural
threshold rejects everything, and the one precise operating point rests on 3
items. The blocker is now specific, and it is a labelling exercise rather than
an engineering one: several hundred adjudicated items, ideally two raters,
before a threshold is defensible. A biomedical NLI checkpoint is the untried
variable and the harness takes `--model`.

Items and per-item reasons: `data/reports/ws5b_entailment_sample.json`.

---

## 2 — Reliability

### 2.1 Assess concurrency and pool safety  `[SHIPPED]`

Workers raised to the measured provider ceiling of 10 (1,095 calls, zero errors;
16 failed 35 of 100 on client timeouts). Worst case is 10 assess leases plus 4
retrieval fan-out against a pool ceiling of 20, and psycopg2 raises rather than
waits when exhausted, so the headroom is a correctness property.

### 2.2 Jobs die with the process  `[SHIPPED 2026-09-06]`

Jobs and their events live in Postgres (`jobs`, `job_events`), beside the analyst
cache and the spend ledger the `cache_entries` pattern already established. The
SSE endpoint honours `Last-Event-ID`, so a reconnecting client resumes from its
cursor with no gaps and no duplicates.

Two things the build corrected in the plan above. The analyst cache does **not**
soften a lost job for real traffic: a free-text note sets `skip_cache_write`, so
nothing a non-preset job computed is cached anywhere. Recovery is the durable
event log alone. And orphan detection needs two conditions, not one — see AD-13
and its two amendments. `fly.toml` suspends idle machines, so a heartbeat age
test alone reports a live suspended job as dead, while an instance test alone
fails a healthy job the moment `auto_start_machines` puts a second machine behind
the hostname.

Orphaned jobs fail visibly and the user retries; nothing requeues itself, and a
disowned worker stops spending (`heartbeat()` returns whether the job is still
the caller's to run). A job store that cannot accept work now returns 503 rather
than a bare 500, which the failure-injection suite found.

### 2.3 The corpus goes stale silently  `[SHIPPED 2026-09-06; verified in production 2026-09-09]`

The refresh runs on a Fly scheduled machine (`scripts/deploy_api.sh` destroys and
recreates it on every deploy, so it can never run last release's code), records
its counts and finish time to `cache_entries`, and surfaces them at `/api/health`
as `corpus_refresh`. The probe alerts when that stamp stops advancing, which is
the silent failure a schedule introduces. `last_updated` is shown per trial in
the UI and marked stale past a threshold.

First two cycles on the deployed schedule: 2,982 trials had moved their
`lastUpdatePostDate` since the corpus was loaded and were re-embedded, then an
immediate second cycle reported `embedded: 0` with no model load at all. The
2,982 is the size of the defect below, measured.

The `NOT_YET_RECRUITING` observation turned out to be a **documentation** defect:
`ctgov.py` includes that status deliberately. The real defect beside it was that
the refresh diffed on `status` alone, so a trial whose eligibility criteria were
revised kept both its stale text and its stale embedding — meaning a verdict could
carry a verified citation to text CT.gov no longer publishes. The diff now keys on
`lastUpdatePostDate` and re-embeds what moved.

### 2.4 The analyst does not answer every criterion  `[found 2026-09-07, FIXED 2026-09-10]`

Under prompt v4, Llama-3.3-70B silently returns fewer assessment objects than it
was handed criteria: **13.0% fewer on TREC 2021** (229 of 1,761), 0.7% on SIGIR.
Not truncation — `max_tokens` is 4096 and the longest measured response was 1,732
tokens.

Invisible until Phase 10 instrumented it, and structurally so: an unanswered
criterion produces no assessment, so it cannot fail grounding, cannot trigger the
retry edge, and disappears into `needs_review`. Meanwhile `rollup_trial` calls a
trial `eligible` when every criterion it *received* was met, which CLAUDE.md
already names as unsound over a truncated list.

Numbering the criteria closes it (13.0% → 0.5%), but prompt v5 carries costs of
its own (AD-16). **The cheaper hypothesis is now measured and dead**: v6 — v4
plus only the instruction to answer every criterion — moved coverage by −0.65 pp
over 1,071 paired criteria (AD-17, `v6_coverage_findings.md`). Both v5 and v6
carry that instruction; only v5 recovers coverage, so the gain belongs to
numbering rather than to the instruction.

**Improved on the retry edge, not in the prompt** (AD-18, corrected by AD-20;
`p1_coverage_findings.md` and `p1_correction.md`). Criteria never answered fall
230 to 167 on TREC (13.1% to 9.5%) and 43 to 14 on SIGIR, grounded rises 848 to
918 and 814 to 824, `eligible` precision rises 0.571 to 0.714 on TREC and
end-to-end recall 0.0026 to 0.0033. It costs 12% of TREC surfaced recall.
On by default; `TG_RETRY_MISSING=0` reproduces the previous behaviour.

The load-bearing part is an invariant rather than a number: the full-list retry
used to replace attempt one wholesale, so a retry that answered less made
coverage *worse* than not retrying. `_backfill` now guarantees a retry can only
add.

**Decomposed 2026-09-11, and it was mostly never the analyst** (AD-21,
`coverage_residual.md`). Of TREC's 18.9% shortfall: 8.4% were entries that are
not criteria at all, 7.2% were criteria the model answered while echoing them
differently, and **3.3% genuine omission**. Two deterministic fixes followed —
the parser no longer emits section headers and stranded labels, and matching is
by containment rather than exact text — taking never-answered to 147 on TREC
and 6 on SIGIR, and pulling 74 and 28 real criteria into the assessed window.

**Measured together 2026-09-12** (AD-22, `combined_coverage_result.md`):
criteria never answered fall to **1 of 1,560 on TREC** and 2 of 1,799 on SIGIR.
Roughly 99% of the shortfall was the parser and the matcher, and the retry's
recorded cost to surfaced recall was itself an artifact of the junk criteria,
reversing once they are gone.

**Residual under 1%, stated rather than closed.** It is three things: genuine
omission, criteria the parser never emitted at all (the ten-character line
minimum removes "Pregnancy" and "Prisoners" alongside "Age:"), and compound
lines the model correctly splits. Each remaining route is an ingestion change
that could destroy correct criteria, and the roll-up already reports affected
trials as `needs_review` rather than claiming a verdict over them.

### 2.5 A quote cannot prove absence  `[found 2026-09-10, measured, open]`

An exclusion answered `not_met` claims the patient does **not** match a
disqualifier. `is_absence_grounded` exists for that claim, but it runs only as a
fallback, so any verbatim quote pre-empts it and the model can satisfy the
verifier by quoting anything at all. Spotted in the served UI: *History of
previous lung malignancy or other metastatic malignant tumors* answered
`not_met` and cited with *58-year-old woman with stage IV non-small cell lung
cancer*, a citation saying she has one.

**129 of 840 grounded criteria on TREC (15.4%)**; 48 of 107 quote-grounded
exclusion `not_met` rows across both cohorts.

**Not enforced, and that was measured rather than assumed.** Making absence
primary destroys about half the class, which are legitimate refutations: *2+
aortic insufficiency* against *Severe aortic regurgitation*, *18-week sized
uterus* against *Uterine size > 32 weeks*. Nothing deterministic separates a
refutation from a contradiction, since both quote the same subject and differ by
negation, severity, laterality or count. So the row is stamped `weak_absence`,
counted per request and in the eval, and the UI says *quote verified, but it
does not establish absence* rather than showing a clean grounded badge. AD-19.

A sharper instance of 1.3's entailment gap, and unlike the rest of it, its size
is known exactly.

### 2.6 The cap dropped every disqualifier  `[found and FIXED 2026-09-12]`

`build_typed_criteria` filled the `MAX_CRITERIA` cap with inclusion criteria and
gave exclusion whatever was left, so a trial with more than 24 inclusion
criteria had **none of its exclusion criteria assessed at all**: 287 of 4,000
live trials sampled (7.2%), 10.3% of TREC, 5.2% of SIGIR. Those trials could
only ever come back `eligible` or `cannot_determine`, because every disqualifier
was dropped before the analyst saw it.

Not a smaller sample of the same thing. Exclusion criteria are the reasons a
patient does not qualify, so dropping all of them biases the system toward
saying yes, on exactly the trials with the most to check.

Each kind now gets half the budget and the unused half goes to the other. 287
affected trials become 0. Found by inspecting a live assessment rather than by
any test, which is the reason it survived: no eval metric reads the
inclusion/exclusion split of what was *asked*.

**Related and still open:** `rollup_trial` does not know a list was truncated,
so it can still call a trial `eligible` over one. CLAUDE.md names that unsound
and requires callers to surface truncation, which the UI does with a muted note.
Making truncation block an `eligible` verdict is a verdict-semantics change that
moves the regression gate, so it is named here rather than bundled into the fix.

### 2.7 The demo could not cache its own presets  `[found and FIXED 2026-09-12]`

The web app hardcoded two preset notes; the API's cache allowlist came from the
SIGIR queries fixture. They were never the same strings, so `_is_preset` was
False for every note the demo could produce, every run set `skip_cache_write`,
and each one paid a fresh LLM call per trial at ~29 s. Forever.

One definition now, served by `/api/limits`, so the notes the client offers and
the notes the API will cache cannot drift apart again. Verified live: a preset
assess went 31 s cold to 0 s warm, where before it would have been 31 s every
time. Free text still skips the write, which is what stops an
attacker-controlled note being persisted or growing the store without bound.

### 2.8 Single region, single instance  `[open, accepted]`

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

## 4 — Observability  `[SHIPPED 2026-09-07, one item outstanding]`

`served_monitor.py` reads the `served`-tagged traces and diverges when abstention
or grounding-failure rates leave their committed bands. It is now **armed**: a
missing secret fails the run rather than skipping it. The monitor spent its whole
existence green and silent because the skip branch made "not configured"
indistinguishable from "nothing diverged". To stand it down, disable the workflow
rather than removing the secrets.

**Outstanding:** `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` still have to be
added as repo secrets — that cannot be done from the repo. Until then the nightly
run fails loudly, which is the intended state.

A trace-reading monitor is blind exactly when the API is down, because an outage
produces *fewer* traces rather than different ones. `eval/served_probe.py` closes
that: it hits the live endpoints and gates liveness, pool health, search latency
and result count, budget exhaustion and corpus staleness. It runs first in the
nightly workflow, unconditional on any secret, and again after every deploy
against tighter bounds (`scripts/deploy_api.sh`).

Latency now has an SLO (`docs/deploy_stage_a.md`, `data/reports/served_slo.json`).
Building it produced the finding that **"780 ms warm" is not what a user meets**:
the first search after a resume from suspend measured 20,576 ms server-side with
the keyword cache already warm, settling to 542-570 ms by the third call. The
probe measures cold and warm separately and gates only warm — gating the first
call would fail every deploy, and not measuring it is how it stayed invisible.
p95 is explicitly **not** claimed: a probe is n=1.

Faithfulness is no longer only a nightly batch. Every `/api/assess` request
reports its own `unverifiable_rate`, `grounded_rate` and `note_only_grounded` on
the `done` event and on the trace, so a run that starts producing ungrounded
verdicts at 09:00 is visible in minutes rather than at 21:00. Emitted off the
request path: `emit_scores` ends in a blocking flush measured at 4.1 s.

---

## 5 — Ranked

| | item | why now |
|---|---|---|
| **P0** | ~~PHI refusal at the boundary~~ | **shipped** — the only item that was actively unsafe |
| **P0** | ~~Postgres-backed jobs + SSE resume~~ | **shipped** — a restart no longer loses paid work |
| **P1** | ~~Scheduled corpus refresh + visible `last_updated`~~ | **shipped** — including the revised-criteria re-embed the plan had missed |
| **P1** | ~~Arm the served monitor~~ | **shipped** — plus a probe, because traces cannot see an outage |
| **P1** | Add the two Langfuse repo secrets | the only Phase 10 item that cannot be done from the repo |
| **P1** | The analyst answers 13% fewer criteria than it is asked (TREC) | a trial roll-up over an incomplete criteria list is unsound, and nothing surfaced it |
| **P2** | NER-based PHI detection | closes the names gap regexes cannot reach |
| **P2** | NLI second verifier for entailment | §1.3 puts the gap at 36%; a model to evaluate, not a check to add |
| **P2** | Stage B auth + per-user quota | the only real answer to shared-budget exhaustion |
| — | Trial-only grounding | **measured and rejected** (§1.2, AD-14) — not deferred |
| — | Multi-region | correctly out of scope |

The honest summary has moved. §2.2 is closed, so the system no longer loses paid
work on a routine deploy, and §4 no longer reports "nothing diverged" while
reading nothing. What is left is not a reliability gap but a **claim** gap, and
it is named rather than hidden: a citation proves a quote exists in the inputs,
and for patient facts the input is whatever the user typed (§1.2), and existence
is not entailment (§1.3, 36%). Both are properties of the architecture rather
than bugs in it. The system is safe to show people and dependable enough to use;
what it cannot do is verify the patient.
