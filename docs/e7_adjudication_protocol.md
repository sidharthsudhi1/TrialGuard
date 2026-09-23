# E7 adjudication protocol

How to label the 300 items in `data/reports/e7_adjudication_worksheet.jsonl`.

This is the blocker AD-23 named: several hundred adjudicated items, ideally two
raters, before any NLI threshold is defensible. The engineering is done. What
follows is the judgment, and the quality of the labels is the whole value of the
exercise, so the protocol is specific.

---

## 0. The one question

> **Does the verbatim quote establish this verdict?**

That is the only question. It is not:

- ❌ Is the verdict correct?
- ❌ Is the patient actually eligible?
- ❌ Is this a good criterion, or a well-written trial?
- ❌ Would a clinician agree with the conclusion?

A verdict can be **right** while its citation **fails to show it**. Those items
are the point of the measurement. Judge the link between the quote and the
verdict, nothing else.

Worked example of the distinction:

| | |
|---|---|
| criterion | `Granulocyte count at least 1,500/mm^3` (inclusion) |
| verdict | `not_met` |
| quote | `The complete blood count and biochemical profile are normal` |
| **entails** | **false** |
| why | A normal CBC argues granulocytes *are* adequate. The quote points the opposite way from the verdict. |

The verdict may still be right for reasons outside the quote. Irrelevant. The
citation does not establish it, so `entails = false`.

---

## 1. What each verdict claims

You are checking the quote against the claim implied by (`kind`, `verdict`):

| kind | verdict | the claim the quote must establish |
|---|---|---|
| inclusion | `met` | the patient **satisfies** this requirement |
| inclusion | `not_met` | the patient **does not satisfy** this requirement |
| exclusion | `met` | the patient **matches** this disqualifier |
| exclusion | `not_met` | the patient **does not match** this disqualifier |
| unknown | either | treat as inclusion; the parser could not type it |

`unknown` is 84 of the 300 and is a real category, not a defect — it is emitted
when no anchor identifies the criterion as inclusion or exclusion. Read the
criterion text and judge it on its plain meaning.

---

## 2. The three failure shapes to expect

AD-15 found these recur. Recognising them is most of the work.

**a) Evidence pointing the other way.** The quote is real and relevant and
supports the *opposite* verdict.

> criterion `all patients presenting to the ED with shortness of breath` ·
> verdict `not_met` · quote *"complaining of shortness of breath"* → **false**
> — the quote is evidence FOR the criterion.

**b) One conjunct of a compound criterion.** The criterion requires two or more
things; the quote establishes one.

> criterion `Two or more symptoms of acute appendicitis for at least 24 hours
> or radiologic evidence` · verdict `met` · quote *"abdominal pain"* → **false**
> — one symptom, no duration, no radiology.

**c) An adjacent fact standing in for the required one.** The quote is about the
right body system or topic but does not address the criterion.

> criterion `high energy trauma, e.g. motor vehicle accidents` · verdict
> `not_met` · quote *"She is concerned about breaking her hip as she gets older"*
> → **false** — hip-fracture worry says nothing about high-energy trauma.

> criterion `No active hepatic disease` · verdict `met` · quote *"Several of
> these are biopsied and are all benign adenomas"* → **false** — names a hepatic
> finding rather than establishing absence of disease.

---

## 3. What a sound citation looks like

Mark `entails = true` when the quote, read alone, settles the claim. Numeric and
threshold reasoning counts, and so does a clear clinical implication.

> `Severe hypertension (DBP >= 110 or SBP > 200)` · `not_met` ·
> *"his blood pressure is 151/91 mm Hg"* → **true** — below both thresholds.

> `adults 18-70 years of age` · `met` · *"A 25-year-old woman"* → **true**.

> `Negative pregnancy test` · `met` · quote states the negative test → **true**.

> `Patients with clinically incomplete KD` · `not_met` · quote lists
> *conjunctivitis, strawberry tongue, inflammation of the hands and feet* →
> **true** — complete-KD features argue against "incomplete".

You are allowed to do arithmetic and apply ordinary clinical knowledge. You are
not allowed to import facts that are not in the quote.

---

## 4. Absence-grounded items (empty quote)

**20 of the 300 have `grounded_by = "absence"`, and 15 of those have an empty
quote.** These are not broken rows. An exclusion answered `not_met` claims the
patient does *not* match a disqualifier, which is a claim about absence of
evidence that no verbatim span can support (AD-19). The system verifies them by
checking the criterion's distinctive terms are genuinely absent from the note.

For these, the question becomes:

> **Does the absence of this criterion's terms from the note establish the
> verdict?**

> `Patient with a renal transplant` · `not_met` · *(empty, absence-grounded)* →
> **true** — no renal-transplant terms anywhere in the note.

Mark `false` when absence is not good enough — typically when the criterion
describes something a note would not necessarily mention even if true. Say so in
`adjudication`, because this subset is small and each judgment carries weight.

---

## 5. Mechanics

### Produce two independent copies

```bash
cd /Users/sidharthsudhi/Documents/TrialGuard

# one CSV per rater, from the same worksheet
.venv/bin/python scripts/e7_entailment_sample.py \
  --to-csv data/reports/e7_adjudication_worksheet.jsonl --out rater_a.csv
cp rater_a.csv rater_b.csv
```

Open in any spreadsheet. Fill two columns:

- **`entails`** — `true` or `false`. Leave blank if genuinely undecidable; blank
  stays unrated and is dropped, which is better than a coin flip.
- **`adjudication`** — one short line of reasoning. Mandatory in practice: it is
  what makes a disagreement resolvable later, and it is how the existing 50 items
  are auditable.

Do not touch any other column. `row` exists so a sorted or reordered sheet still
maps back.

**The two raters must not see each other's sheets before both are finished.**
Inter-rater agreement measured after comparing notes is not agreement, and the
kappa becomes meaningless.

### Convert back and merge

```bash
.venv/bin/python scripts/e7_entailment_sample.py --from-csv rater_a.csv --out rater_a.jsonl
.venv/bin/python scripts/e7_entailment_sample.py --from-csv rater_b.csv --out rater_b.jsonl

.venv/bin/python scripts/e7_entailment_sample.py --merge rater_a.jsonl rater_b.jsonl
```

`--from-csv` refuses anything in `entails` that is neither blank nor a
recognised boolean, and writes nothing on error. A typo read as "false" would
bias the rate in exactly the direction the exercise is measuring.

The merge writes `data/reports/e7_merged_gold.jsonl` and a summary with Cohen's
kappa, raw agreement, and the disagreement list. **Disagreements are excluded,
not resolved** — a rate computed over agreed items alone is biased toward easy
cases, so the summary says so and the excluded items need a third pass.

---

## 6. Reading the kappa

| kappa | reading |
|---|---|
| > 0.8 | strong; the labels can carry a threshold |
| 0.6 – 0.8 | usable, report it alongside every downstream number |
| 0.4 – 0.6 | weak; the question is being read two ways, fix the protocol before labelling more |
| < 0.4 | the labels do not mean one thing; do not build on them |

If kappa lands under 0.6, the right response is to read the disagreements
together, sharpen this document, and re-rate — not to average the two raters.

---

## 7. Practical notes

- **300 items, roughly 20-40 seconds each** once calibrated: about 2-3 hours per
  rater. Do it in sittings; fatigue shows up as drift toward whichever label is
  easier.
- **Calibrate first.** Both raters should label the same 20 items, compare, and
  settle disagreements *before* starting the real 300. Those 20 are practice and
  are re-rated in the real pass.
- **The base rate to expect is around 36%** non-entailing, from the existing 50.
  If you land far from that, it is worth asking whether the question drifted —
  but do not steer toward it.
- **Do not look up the trial or the patient.** Judge only the `criterion`,
  `verdict` and `quote` in front of you. The verifier has no more context than
  that, so neither should the label.

---

## 8. What happens next

Once `e7_merged_gold.jsonl` exists, re-run the three checkpoints against it:

```bash
.venv/bin/python -m trialguard.eval.nli_probe --model microsoft/deberta-large-mnli
.venv/bin/python -m trialguard.eval.nli_probe --model pritamdeka/PubMedBERT-MNLI-MedNLI
.venv/bin/python -m trialguard.eval.nli_probe --model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```

The question they answer: does an operating point exist at n≈350 that did not
exist at n=50 — one that catches unsound citations without destroying sound
ones? At n=50 none of the three had one ([`e7_findings.md`](../data/reports/e7_findings.md)).

`nli_probe.py` currently reads `ws5b_entailment_sample.json` as its gold; point
it at the merged file, or concatenate the two, and say which in the report.
