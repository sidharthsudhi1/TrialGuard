# AD-23 — An off-the-shelf NLI model is not the answer

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

An off-the-shelf NLI model is not the answer to the entailment gap, and now that is measured rather than assumed. `deberta-large-mnli` against the 50 hand-adjudicated items of `ws5b_entailment_sample.json` **ranks** entailment respectably (AUC 0.79) and **cannot be thresholded**: it is severely miscalibrated on this task, with median p(entail) of 0.027 for citations a human judged sound, so every threshold near the natural boundary rejects all 50 and reports the 0.36 base rate as precision. The one precise operating point, p<0.002, catches 3 of 18 unsound citations and destroys none, and three items is not a threshold. Everywhere else it destroys sound citations at roughly the base rate, which weakens the GROUNDED badge in exactly the way L4's abstention laundering weakened the faithfulness proxy. The failure is structural: MNLI premises are sentences containing the information, while these are clinical fragments such as *2+ aortic insufficiency*. P2 therefore moves from an assertion to a stated blocker, namely several hundred adjudicated items and ideally two raters before any threshold is defensible, which is a labelling exercise rather than an engineering one

## Alternatives considered

Shipping it at p<0.02, where it catches 89% of unsound citations and rejects 11 of 32 sound ones; concluding from the first run that the model had no signal, which is what testing 0.1/0.25/0.5/0.75/0.9 appeared to show and was wrong; treating AUC 0.79 as sufficient, since ranking without calibration cannot gate anything
