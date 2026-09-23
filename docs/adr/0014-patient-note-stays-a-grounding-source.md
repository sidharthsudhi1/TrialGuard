# AD-14 — The patient note stays a grounding source

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

The patient note stays a grounding source. Grounding's contract is that a quote appears verbatim **in the inputs**, and for the facts that decide eligibility the input is the user's note — the trial states the requirement, not whether this patient meets it. Measured on both cohorts: 84% of grounded quotes exist only in the note, and restricting decisive verdicts to trial text raised the criterion unverifiable rate from 3.1% to 42.6% (SIGIR) and 3.5% to 50.0% (TREC), emptying the `eligible` tier. It also keeps the wrong survivors — of the 8 trial-grounded quotes on SIGIR, 2 restate their own criterion and 6 quote a different one, so none is evidence about a patient. The planted-evidence risk is therefore restated rather than mitigated: a user who writes false facts receives verdicts faithful to those false facts, which is garbage-in and is indistinguishable to any verifier reading the same note. What ships is provenance — `grounded_in` per assessment, `note_only_grounded` per request — with `TG_GROUND_TRIAL_ONLY` kept, default off, so the measurement is reproducible

## Alternatives considered

Trial-only grounding (measured, rejected above); a weaker note-sourced *class* that still counts toward decisive verdicts (same numbers, plus a distinction nothing acts on); refusing notes that look like trial text (the overlap is legitimate clinical language, so this refuses real patients)
