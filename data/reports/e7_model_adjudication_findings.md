# E7 — model adjudication of the 300, and what it is worth

Produced 2026-09-23 by `claude-opus-5` following
[`docs/e7_adjudication_protocol.md`](../../docs/e7_adjudication_protocol.md).
Labels: [`e7_model_adjudication.jsonl`](e7_model_adjudication.jsonl).

> **This is model output, not human gold.** It is deliberately not written to
> `e7_merged_gold.jsonl` and must not be quoted as the adjudicated rate. AD-3
> rules out an LLM verifier for enforcement on correlated-error grounds, and
> AD-15 considered and rejected using one offline to size this problem. What
> follows is a pre-screen with a measured error profile, which is a different and
> lesser thing than a labelled gold set.

## The labels

All 300 items carry `entails` and a one-line reason.

| | count | rate |
|---|---|---|
| non-entailing | 86 | **0.2867** |
| entailing | 214 | 0.7133 |
| human baseline (WS-5b, n=50) | 18 | 0.3600 |

By criterion kind, and by how the row was grounded:

| kind | non-entailing | entailing |
|---|---|---|
| inclusion | 31 | 77 |
| exclusion | 41 | 67 |
| unknown | 14 | 70 |

| grounded_by | non-entailing | entailing |
|---|---|---|
| quote | 80 | 200 |
| absence | 6 | 14 |

Exclusion criteria are the worst-performing kind (38% non-entailing against 29%
for inclusion), which is consistent with AD-19: an exclusion answered `not_met`
is a claim about absence, and a verbatim span is a poor instrument for it.

## What these labels are worth, measured

The 50 WS-5b items are excluded from the 300 by construction, so they are a
held-out set with human labels already on them. I re-adjudicated **40 of them
blind** — the 10 whose labels had been displayed earlier in the session are
excluded as contaminated — and compared.

| | |
|---|---|
| held-out items | 40 |
| raw agreement | **36/40 = 0.900** |
| **Cohen's kappa** | **0.7626** |
| human non-entailing rate | 0.3250 |
| model non-entailing rate | 0.2750 |

Kappa 0.76 sits in the band this project's own protocol calls *usable, report it
alongside every downstream number* — not the >0.8 that would let labels carry a
threshold on their own.

## The disagreements have a direction, and it matters

All four are listed, because four is small enough to read:

| # | human | model | why the human said no |
|---|---|---|---|
| 10 | false | true | sigmoidoscopy does not establish absence of prior colectomy |
| 20 | true | false | a cool foot is a sign of limb ischemia |
| 22 | false | true | heavy smoking argues the criterion is met, not not_met |
| 42 | false | true | naming an orbital site does not establish absence of liver involvement |

**Three of four run the same way: the model accepted an indirect inference where
the human required the quote to establish the claim directly.** Items 10 and 42
are both "an adjacent finding implies the absence of something else" — which is
precisely the *adjacent fact standing in* failure shape the protocol warns about.
The model applied the rule to others and then committed it itself.

That is a systematic leniency, not noise. Its consequence is direct: **the 28.7%
rate is probably an underestimate.** On the same 40 items the model scored 27.5%
where the human scored 32.5%. Scaling naively would put the 300 nearer 34%, close
to the 36% of record — but that is a ratio off 40 items and is an indication, not
a correction.

## How to use this

**Do not** report 0.2867 as the entailment gap. The number of record stays 36%
(AD-15) until humans label the 300.

**Do** use it as a pre-screen. Two uses are sound:

1. **Triage.** The 86 items marked non-entailing are where a human's attention is
   worth most. If a rater confirms most of them and spot-checks the 214, the
   labelling cost falls sharply without the gold depending on the model.
2. **A second-rater baseline.** When a human labels the 300, compute kappa
   against these. The held-out 0.76 predicts roughly what to expect, and a large
   departure is evidence something drifted in the protocol.

**The leniency is the thing to watch for.** A human reviewing these should look
hardest at the 214 `true` labels, not the 86 `false` ones — that is the direction
the model errs.

## What this does not settle

**Correlated errors are untested here.** AD-3's argument is about a judge drawn
from the analyst's own family. The analyst is Llama-3.3-70B and the adjudicator
here is a different family, which weakens the correlation argument but does not
eliminate it — both are instruction-tuned models trained on overlapping corpora,
and neither the analyst nor this adjudicator was checked for shared blind spots.

**n=40 for the agreement figure.** Kappa 0.76 carries a wide interval at that
size. Four disagreements is enough to see a direction and not enough to quantify
it.

**One adjudicator, one pass, no re-rating.** The human baseline is also a single
rater with no inter-rater agreement claimed (WS-5b's own caveat). So this compares
one unvalidated rater against another.
