# WS-5 — Guardrails: what closed, what is structural, what is now visible

Run 2026-09-07. Prompt v4, DeepInfra, TREC 2021 and SIGIR, 20 patients × top-10.
Both 5a arms are cache replays of the same analyst responses — grounding is pure
Python, so the two arms differ only in a re-scoring pass and cost $0.005 in
retries. Raw: `ws5a_off_sigir.json`, `ws5a_on_sigir.json`,
`ws5a_off_trec2021.json`, `ws5a_on_trec2021.json`,
`ws5b_entailment_sample.json`.

## 5a — Ground decisive verdicts in trial text only: measured, rejected

### What was proposed

`production_readiness.md` §1.2 and `sanitize.py`: grounding proves a quote is
verbatim, not that its source is trustworthy. The patient note is user-supplied,
so an attacker who plants trial-shaped evidence in the note obtains a *grounded*
wrong verdict. Proposed fix: ground decisive verdicts against trial text only and
count note-sourced quotes as a visibly weaker class.

### Result

| | SIGIR off | SIGIR on | TREC off | TREC on |
|---|---|---|---|---|
| grounded criteria | **545** | 88 | **818** | 151 |
| met | 164 | 4 | 340 | 10 |
| not_met | 375 | 79 | 470 | 127 |
| unverifiable | 36 | **491** | 52 | **742** |
| criterion unverifiable rate | 0.0311 | **0.4258** | 0.0348 | **0.4997** |
| trials eligible | 4 | **0** | 8 | **0** |
| end-to-end recall | 0.0208 | **0.0** | 0.0033 | **0.0** |
| surfaced precision | 0.3182 | 0.2391 | 0.7083 | 0.4405 |

Both cohorts, same shape: grounded criteria fall ~84%, the unverifiable rate rises
13.7x on SIGIR and 14.4x on TREC, and the `eligible` tier empties completely. Every
trial becomes `needs_review`.

### Why: the evidence is in the note by construction

The provenance split is the finding, and it does not need the strict arm to see:

| grounded quotes by source | SIGIR | TREC 2021 |
|---|---|---|
| patient note only | **458 (84.0%)** | **673 (82.3%)** |
| trial text | 8 (1.5%) | 67 (8.2%) |
| absence check | 79 | 78 |

Sampled note-grounded verdicts, verbatim:

```
[note] met     Age greater than 18 years        <- "67-year-old"
[note] met     African American                 <- "A 58-year-old African-American woman"
[note] met     Clinical diagnosis of hypertension <- "She is known to have hypertension"
[note] not_met Taking calcium channel blocker   <- "She currently takes no medications"
```

Every one is correct behaviour. The evidence that *this patient* satisfies a
criterion lives in the patient record; the trial states the requirement, not
whether it is met. Requiring the quote to appear in the trial's eligibility text
asks for something that is not there by the structure of the task.

Worse, it keeps the wrong survivors. Of the 8 trial-grounded quotes on SIGIR, 2
merely restate their own criterion (see 5b) and the other 6 quote a *different*
criterion's text. None of them is evidence about a patient. Trial-only grounding
does not tighten the evidence standard; it discards the only class of evidence
that exists and retains a residue that proves less than what it removed.

### Verdict

Rejected. Not tunable — a smaller version of this change is a smaller version of
the same category error.

The security concern behind it is real and is restated correctly rather than
mitigated falsely: **grounding's contract is that a quote appears verbatim in the
inputs, and for patient facts the input is the user's note.** A user who writes
false facts about a patient receives verdicts faithful to those false facts. That
is garbage-in, not hallucination, and no verifier reading the same note can
distinguish the two. Only a link to a real record could, and that is out of scope
for a system whose first rule is that no real patient data ever enters it.

What ships instead is visibility, which was the half of the proposal that
survived contact: `grounded_in` is recorded on every assessment, `/api/assess`
reports `note_only_grounded` per request, and the end-to-end report carries
`note_only_grounded_rate`. The flag stays behind `TG_GROUND_TRIAL_ONLY`, default
off, so the measurement is reproducible — the same treatment R4's reranker and
L4's partial retry got.

### Caveat

Both arms ran `--cached-only`, so a grounding failure could not buy a fresh
retry. In production the strict arm would retry and might recover some quotes, so
its cost here is an **upper bound**. The upper bound is the right direction for a
reject decision, and the mechanism — 84% of evidence being note-sourced — is not
something retries can move.

## 5b — Grounding checks existence, not entailment: measured, stated, open

Two numbers, because one is exact and narrow and the other is broad and sampled.

### Deterministic lower bound: 0.12% – 0.37%

A quote that is a span of *the criterion it was filed under*, and appears nowhere
in the patient note, proves the criterion was printed rather than that the patient
matches it. That is machine-detectable at full coverage
(`is_self_referential`, recorded as `self_referential`, never enforced):

| | SIGIR | TREC 2021 |
|---|---|---|
| self-referential / grounded | 2 / 545 (0.37%) | 1 / 818 (0.12%) |

This is a floor, not an estimate. A quote citing the wrong patient fact is
equally unsupported and no string comparison can see it.

### Sampled estimate: 36% (95% CI 24% – 50%)

Seeded random sample (seed 20260907) of 50 grounded decisive verdicts, 25 per
cohort, each adjudicated against one question: *does this verbatim quote establish
this verdict?* — not *is the verdict correct*. Items and per-item reasons are
committed in `ws5b_entailment_sample.json`.

| | non-entailing | rate | 95% CI |
|---|---|---|---|
| SIGIR | 12 / 25 | 0.48 | 0.30 – 0.67 |
| TREC 2021 | 6 / 25 | 0.24 | 0.12 – 0.43 |
| **combined** | **18 / 50** | **0.36** | **0.24 – 0.50** |

The failures are not exotic. Three recurring shapes:

- **Evidence pointing the other way.** "Granulocyte count at least 1,500/mm^3"
  answered `not_met`, quoting "The complete blood count and biochemical profile
  are normal" — which argues the criterion is met.
- **One conjunct of a compound criterion.** "Patients aged 18-90 presenting with
  dysphagia or food impaction" answered `met`, quoting "A 52-year-old". Age is
  established; the clinical half is never addressed.
- **An adjacent fact standing in for the required one.** "Patients with acute
  renal failure" answered `not_met`, quoting a urinalysis result, which does not
  speak to renal failure.

SIGIR is twice TREC's rate. Its criteria are longer and more often compound, which
is exactly the shape the second failure mode needs.

### What this does and does not say

It does **not** say a third of verdicts are wrong. The quote is real in every one
of these — the verifier's 100% catch rate on corrupted quotes is unaffected, and
many of the flagged verdicts are still the right answer, reached for a reason the
citation does not carry. What it measures is the size of the gap between "cited"
and "shown", which `grounding.py` has always declared in its own docstring and
which this is the first number for.

AD-3 rules out an LLM verifier on correlated-error grounds, so this stays open
rather than fixed. Stated in the README beside the faithfulness claim, which is
the deliverable WS-5 asked for.

### Caveat

One adjudicator, no second rater, so no inter-rater agreement is claimed, and
several items are genuinely marginal — taking the three most marginal SIGIR items
as entailing moves the combined rate to 0.30. The sample is seeded and the
judgments are committed precisely so the reading can be disputed on the same data.

## 5c — Per-request faithfulness: shipped

The served monitor is a nightly batch, so a run that starts producing ungrounded
verdicts at 09:00 was caught at 21:00. The `done` event and the Langfuse trace now
both carry that request's own `unverifiable_rate`, `grounded_rate` and
`note_only_grounded`.

Emitted **off** the request path. `emit_scores` ends in a blocking
`client.flush()`, measured at **4.1 s** against a configured Langfuse — inline, it
would have added that to the wall clock of every completed assessment, paying
latency to report on latency.
