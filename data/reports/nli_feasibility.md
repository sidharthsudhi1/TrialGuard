# Can an NLI model verify entailment? Promising, not shippable, and now it has a number

Run 2026-09-12. `microsoft/deberta-large-mnli` against the 50 hand-adjudicated
items in `ws5b_entailment_sample.json`. No spend, nothing shipped, no change to
the served path. Raw: `nli_probe.json`.

## Why this was worth running

WS-5b measured the entailment gap at 36%: a citation can be genuinely verbatim
and still not establish the verdict it supports. AD-3 rules out an LLM verifier
on correlated-error grounds, and `production_readiness.md` has carried an NLI
model at P2 ever since as "a model to evaluate, not a check to add" -- a
sentence that had no evidence behind it in either direction.

The 50 adjudicated items make it answerable for free. Each is a grounded
decisive verdict with a human judgment of whether its quote establishes it, so
it is a labelled set, not a demo.

## Result

**There is real signal, and it is unusable at any natural threshold.**

| p(entail) < | rejects | precision | recall | sound citations destroyed |
|---|---|---|---|---|
| 0.001 | 0 | — | 0.000 | 0 |
| **0.002** | **3** | **1.000** | **0.167** | **0** |
| 0.005 | 11 | 0.545 | 0.333 | 5 |
| 0.01 | 21 | 0.476 | 0.556 | 11 |
| 0.02 | 27 | 0.593 | 0.889 | 11 |
| 0.03 | 35 | 0.486 | 0.944 | 18 |
| 0.05 | 39 | 0.462 | 1.000 | 21 |

**AUC 0.79**, base rate 0.36.

The first attempt tested 0.1 / 0.25 / 0.5 / 0.75 / 0.9 and concluded the model
had no discriminative power: at p<0.5 it rejects all 50 items and reports
precision 0.36, which is exactly the base rate. That reading was wrong. The
model is severely miscalibrated on this task -- median p(entail) is **0.027**
for citations a human judged sound -- so every threshold near the natural
decision boundary rejects everything. Ranking separates the classes perfectly
respectably; the probabilities do not mean what they usually mean.

The distinction matters: a model that ranks well and calibrates badly is a
threshold problem, and a model that ranks at 0.5 is the wrong model.

## Why it still cannot ship

**The only precise operating point rests on three items.** p<0.002 rejects 3
citations, all 3 genuinely unsound, and destroys none. That is the shape a
verifier needs -- but a threshold tuned on 3 positives is not a threshold, and
the next 3 items could move it by an order of magnitude.

**Everywhere else it destroys sound citations at roughly the base rate.** At
p<0.02, where it catches 89% of unsound citations, it also rejects 11 sound ones
-- and this system's product claim is that a GROUNDED badge means something. A
verifier that removes a third of true citations to catch most false ones makes
the badge weaker, not stronger, in exactly the way L4's abstention laundering
made the faithfulness proxy look better while the system got worse.

**The failure mode is structural, not incidental.** MNLI premises are sentences
that contain the information; these premises are short clinical fragments
("2+ aortic insufficiency", "62-year-old"). The model is being asked to do
something adjacent to its training task, which is why the probabilities collapse
toward zero.

## What this changes

`production_readiness.md` P2 moves from an assertion to a measured position:
an off-the-shelf NLI model **ranks** entailment on this data at AUC 0.79 and
**cannot be thresholded** on 50 items. The specific next requirement is now
stated rather than implied: several hundred adjudicated items, ideally with a
second rater, before any threshold is defensible. That is a labelling exercise,
not an engineering one, and it is the honest blocker.

Still out of scope for the same reason as before: this is a model to integrate
and evaluate, not a check to add, and it sits beside NER-based PHI detection on
that shelf.

## Caveats

- One model. A biomedical NLI checkpoint (SciFact, MedNLI) would very plausibly
  calibrate better on clinical fragments and was not tried; the harness takes
  `--model`, so that is one command.
- 50 items, single adjudicator, no inter-rater agreement, so the labels
  themselves carry uncertainty the AUC does not show.
- The hypothesis templating is one phrasing of four. NLI results move with
  hypothesis wording, and no sweep was run.
- Zero-shot only. A calibration layer fitted on held-out adjudications is the
  obvious next step and needs the larger set first.
