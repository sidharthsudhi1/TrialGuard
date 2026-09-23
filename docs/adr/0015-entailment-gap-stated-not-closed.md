# AD-15 — The entailment gap is stated with a number, not closed

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

The entailment gap is stated with a number rather than closed. A verbatim quote can fail to establish the verdict it supports, and on a seeded 50-item sample 36% did (95% CI 24–50%). AD-3 rules out an LLM verifier for enforcement on correlated-error grounds, and that reasoning is unchanged by having measured the gap — a judge drawn from the same model family fails on the same criteria the analyst fails on. Using one *once*, offline, to size the problem was considered and rejected in favour of adjudicating a sample by hand, because the number's whole purpose is to be trustworthy where the automated path is not. The machine-checkable subset (a quote restating its own criterion) is recorded as `self_referential` at 0.12–0.37% and is a floor, not an estimate

## Alternatives considered

An NLI model as a second verifier (a real option, and the honest reason it is not here is that it is a model to integrate and evaluate, not a check to add — P2, same shelf as NER-based PHI); an LLM judge in CI (AD-3); leaving the limit as prose, which is what `grounding.py`'s docstring already did
