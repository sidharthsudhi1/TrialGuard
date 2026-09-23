# E7 — moving the entailment gap

Measured 2026-09-23. Two parts: closing the variable AD-23 left open, and
building the thing AD-23 said was actually blocking. Cost **$0** throughout.

AD-15 put the gap at 36%: a verbatim quote can be real and still not establish
the verdict it supports. AD-23 evaluated `deberta-large-mnli` as a second
verifier, found it unthresholdable, and left one named loose end:

> A biomedical NLI checkpoint is the untried variable and the harness takes
> `--model`.

## Part 1 — the biomedical checkpoint, tested. It does not rescue it.

Against the same 50 hand-adjudicated items (18 non-entailing, base rate 0.36):

| checkpoint | AUC | median p(entail), sound | median, unsound | separation | zero-damage threshold |
|---|---|---|---|---|---|
| `microsoft/deberta-large-mnli` (AD-23) | **0.7899** | 0.0269 | 0.0065 | 4.1x | none |
| `pritamdeka/PubMedBERT-MNLI-MedNLI` | 0.7648 | 0.0191 | 0.0010 | **18.2x** | none |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 0.6901 | 0.0085 | 0.0033 | 2.6x | none |

**AD-23's diagnosis was half right.** It attributed the failure to domain
mismatch: MNLI premises are sentences, these are clinical fragments. A
MedNLI-trained checkpoint, whose premises *are* clinical text, does separate the
classes far better — median sound/unsound separation rises 4.1x to 18.2x. So the
domain hypothesis has real support.

But it does not become usable. Its best precision is 0.60 at p<0.002, catching 12
of 18 unsound citations while destroying 8 sound ones. **No checkpoint has an
operating point that catches anything without destroying sound citations**, which
is the bar, because the product claim is that a GROUNDED badge means something.
Ranking improved; usability did not follow.

The larger, more modern general NLI model is the worst of the three, which rules
out "just use a better NLI model" as cleanly as the biomedical result rules out
"just use an in-domain one".

**AD-23's conclusion stands, and its named loose end is now closed rather than
open.** Three architectures, three training regimes, no shippable threshold on 50
items.

## Part 2 — the blocker, built

AD-23's real finding was that the blocker is a labelling exercise rather than an
engineering one: several hundred adjudicated items, ideally two raters, before
any threshold is defensible. That work could not start, because **WS-5b's 50
items were drawn ad hoc and only the output survived** — there was no sampler to
extend and no merge step for a second rater.

`scripts/e7_entailment_sample.py` is both.

**Draw** (`--draw N`) walks the cached corpus, collects every grounded decisive
verdict, excludes the 50 already adjudicated, and stratifies by (cohort, kind).
It runs cached-only, so it costs nothing and cannot spend. 300 items are drawn
and ready:

| | sigir | trec_2021 |
|---|---|---|
| inclusion | 54 | 54 |
| exclusion | 54 | 54 |
| unknown | 30 | 54 |

Stratification is deliberate: a proportional draw would swamp exclusion criteria
with the far more numerous inclusions, and AD-19 put the sharpest slice of this
gap on exclusion `not_met`. `unknown` is a real third kind in the schema, emitted
by `attach_kinds` when no anchor identifies the criterion, and it is 28% of the
draw — a population WS-5b never reported separately.

The population matches WS-5b rather than drifting from it: **6.7% absence-grounded
here against 6.0% there** (20 of 300, 15 with an empty quote by construction,
against 3 of 50). Those need the rater to answer a different question — whether
the *absence* of the criterion's terms establishes the verdict — and the
worksheet keeps `grounded_by` on every row so they can be told apart.

**Merge** (`--merge A B`) takes two independently filled worksheets and reports
Cohen's kappa, raw agreement, and the disagreement list. It **excludes**
disagreements from the merged gold rather than silently resolving them, and says
so in the output, because a rate computed over agreed items only is biased toward
the easy cases. Kappa was checked against `sklearn.metrics.cohen_kappa_score` and
matches to six decimal places.

## What was deliberately not done

**Nothing was auto-labelled.** An LLM judge is ruled out for enforcement by AD-3
on correlated-error grounds, and AD-15 explicitly considered and rejected using
one offline to size the problem, because the number's whole purpose is to be
trustworthy where the automated path is not. Auto-filling 300 `entails` fields
would have produced a large gold set with exactly the property that makes the
existing 50 worth more than it.

So the worksheet ships with 300 nulls. That is the honest state: the engineering
is done and the judgment is not.

## What this does not claim

**50 items is still the evidence base for Part 1.** Three checkpoints agreeing
that no threshold works is stronger than one, but all three were evaluated on the
same 50 items by the same single rater, with no inter-rater agreement. If the
expanded set changes the base rate, the operating points move with it.

**One framing, untested alternatives remain.** Every probe uses the bare quote as
premise. AD-23's own diagnosis suggests a fix nobody has tried: give the model the
quote's *surrounding sentence* as premise, which would put it on-distribution
without changing the model. The gold items store only the quote, so testing it
needs re-extraction from source.

**Kappa is untested on real data.** It is verified against sklearn on synthetic
input; no two humans have yet rated the same item here.

## Next

The engineering path is exhausted until the labels exist. Two raters, 300 items,
the question fixed as *does this quote establish this verdict* — not *is the
verdict correct*. Then re-run all three checkpoints against the merged gold and
see whether an operating point appears at n=350 that does not exist at n=50.
