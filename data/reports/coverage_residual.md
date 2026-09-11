# Where the analyst's "missing criteria" actually went

Run 2026-09-11. Closes the investigation that ran through v5 (AD-16), v6
(AD-17), the retry re-ask (AD-18) and the metric correction (AD-20). Every
number here is over the eval cohorts at 20 patients x top-10.

## The shortfall was mostly not the model

Decomposed against the cached corpus, TREC 2021's 18.9% shortfall:

| | TREC | SIGIR |
|---|---|---|
| exact echo match | 81.1% | 95.5% |
| **answered, echoed differently** | **7.2%** | 1.4% |
| **not a criterion at all** | **8.4%** | 0.7% |
| **genuinely unanswered** | **3.3%** | 2.5% |

Four prompt and retry interventions had been aimed at a 13% model-compliance
problem. The model's actual omission rate is **~3%**, which is why none of them
moved it much: v5's numbering worked by accident of format, v6's instruction did
nothing, and the retry's gains were half measurement artifact.

## Two deterministic fixes, neither involving the model

**The parser emitted things that are not criteria.** "inclusion criteria:"
survived as its own criterion 91 times on TREC, alongside stranded labels
("Performance status:", whose content is on the following lines) and
placeholders ("Not specified"). 340 of 1,472 TREC entries and 203 of 1,013
SIGIR entries. The rule that removes them is structural, never length: short
does not mean junk, and "Karnofsky 60-100%", "Contrast allergy" and "Age below
18" are all real criteria of three words or fewer.

Removed 119 entries on TREC (6.4%) and 15 on SIGIR (1.0%), and pulled **74 and
28 real criteria into the assessed window**, because junk was occupying slots
under the MAX_CRITERIA cap.

An earlier claim that this was crowding real criteria past the cap at scale was
overstated and is corrected here: trials that hit the cap mostly still hit it
(28 -> 27 on TREC). The gain is in *which* criteria occupy the window, not in
how many trials are truncated.

**Exact text matching called answered criteria unanswered.** `align_assessments`
pairs by containment, one-to-one, longest criterion first. Criteria never
answered fall 167 -> 147 on TREC and 14 -> 6 on SIGIR; entries matching no
criterion 81 -> 54 on TREC.

## The residual, and why it stops here

~3% on both cohorts, and it is not one thing:

- **Genuine omission.** The model skips a criterion for no visible reason.
  Bounded retries recover some; nothing tried recovers it reliably.
- **Criteria the parser never emitted.** `_parse_block` drops any line of ten
  characters or fewer, which removes "Pregnancy", "Prisoners", "Children" and
  "CALGB 0-2" alongside "Age:" and "Other:" -- 100 lines on TREC and 187 on
  SIGIR, roughly a quarter of them real. These are worse than unanswered: they
  are never asked and never counted, so they do not appear in any rate on this
  page. Relaxing the threshold also admits wrapped-line fragments ("and over",
  "RELATIVE"), so it needs its own measurement.
- **Compound lines.** "Men and women, ages 30 to 69. Documented myocardial
  infarction." is one parsed entry; the model correctly answers it as two, and
  neither answer matches the whole. Splitting on sentence boundaries is
  dangerous in text full of "1.5 mg", "i.e." and "No. 3".

Each remaining route is an ingestion change with a real chance of destroying
correct criteria, measurable only by re-parsing the corpus and re-running both
cohorts. The honest position is that **~3% of criteria are not assessed, the
roll-up reports the affected trials as `needs_review` rather than claiming a
verdict over them, and that is stated rather than closed.**

## What changed in how this is measured

`criterion_unanswered` is counted against the criteria list, not computed as
asked minus answered (AD-20). `criterion_unmatched` is reported beside it. Both
use the same alignment the retry uses, so "answered" has one definition in the
codebase rather than two that disagree.

## Caveat

Fix 1 was measured with `TG_STRICT_CRITERIA=0` so the analyst cache stayed
valid and the comparison cost nothing. The two fixes have therefore not been
measured *together* end to end: the strict parser changes the cache namespace,
so that costs a full fresh run on both cohorts. The structural effects above
are independent and additive; the trial-level interaction is unmeasured.
