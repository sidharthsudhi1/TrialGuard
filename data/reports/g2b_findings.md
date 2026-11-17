# G2b: served analyst on strict synthetic gold

Run 2026-11-17/18. Seed 20261117, digest `15c8eb910c446397`, two builds match. 200 patient-criterion pairs, 0 call errors, $0.0042 on an isolated ledger. Served configuration: prompt v4, full graph (grounding, up to 2 retries), DeepInfra `meta-llama/Llama-3.3-70B-Instruct-Turbo`. Each criterion is passed as `inclusion`, so `met` means the sentence holds for the patient. Raw: `g2b_gold.json`, `g2b_calls.jsonl`, `g2b_results.json`.

## What this measures, and what it does not

This is criterion accuracy on machine-checkable criteria (age, ECOG, serum creatinine, total bilirubin, hemoglobin, platelets, ANC) in templated seven-line notes. It is easier than real notes: every fact is present, stated once, with a unit. It is not the retired TrialGPT 87% target and should not be quoted against it.

## Why G2b and not the 2026-11-10 rows

A read of the v1 rows before any analyst call found their truth labels unreliable. Decimals were cut at the period (`<= 1.5` read as `<= 1`). Creatinine clearance was read as creatinine and HbA1c as hemoglobin. Units were ignored, and fragments were kept. The strict builder keeps only whole bullet lines that fully match a per-attribute pattern with an explicit unit, converts units to the note's, and computes truth with its own comparator. I (Claude) read all 200 kept rows before the run and dropped none. Rows sitting exactly on an inclusive range edge are kept and scored as inclusive. This audit is a model's, not a clinician's.

## Result (n=200, 95% bootstrap CI over rows)

| Metric | Value |
|---|---|
| Accuracy (abstention counts wrong) | **0.865** [0.815, 0.910] |
| Decisive coverage | 0.980 [0.960, 0.995] |
| Accuracy among decisive verdicts | 0.883 [0.835, 0.925] |
| False exclusion (holds, verdict `not_met`) | 0.104 [0.052, 0.164] |
| False inclusion (fails, verdict `met`) | 0.129 [0.061, 0.205] |

Verdicts: 112 met, 84 not_met, 4 cannot_determine, 0 unverifiable. Two rows used a retry.

| Attribute | n | Accuracy [CI] | False excl. | False incl. |
|---|---|---|---|---|
| creatinine | 25 | 1.000 [1.00, 1.00] | 0.00 | 0.00 |
| ecog | 24 | 1.000 [1.00, 1.00] | 0.00 | 0.00 |
| age | 40 | 0.925 [0.83, 1.00] | 0.14 | 0.00 |
| hemoglobin | 23 | 0.870 [0.74, 1.00] | 0.00 | 0.14 |
| platelets | 40 | 0.825 [0.70, 0.93] | 0.22 | 0.06 |
| bilirubin | 15 | 0.800 [0.60, 1.00] | 0.10 | 0.40 |
| anc | 33 | 0.667 [0.52, 0.82] | 0.15 | 0.46 |

## The finding that matters: every wrong verdict was grounded

All 23 wrong decisive verdicts carry a quote that appears verbatim in the note ("ANC 1402/uL", "Bilirubin 1.91 mg/dL"). Grounding did its job. The quote is the right fact, and the comparison on top of it is wrong. This is the entailment gap (weak point 3) measured directly for the first time: on these notes, about 12% of decisive verdicts are faithful to the source and still wrong. A verified citation shows where the fact is. It does not show the verdict follows.

## Exploratory splits (chosen after seeing errors, not registered)

- **Unit conversion.** Rows where the criterion's unit differs from the note's (platelets per uL vs 10^9/L, ANC 10^9/L vs per uL, hemoglobin g/L vs g/dL): 28/42 right (0.667). Same unit: 145/158 (0.918).
- **Exact edge.** Value exactly on a bound: 25/28 (0.893). Range edges failed for age 75 in 18-75 and age 80 in 40-80.
- Some same-unit errors are not boundary or unit cases (ANC 1947 vs `>= 1,500/uL` and ANC 7893 vs `>= 1500/mm3`, both `not_met`; platelets 237 vs `>=150 x 10^9/L`, `not_met`). The cause is not established.

## Decision

No served default changes, and there is no kill bar. G2 stays adopted as an instrument and now has an analyst number: **0.865 criterion accuracy on machine-checkable criteria in templated notes**. The concrete lead is the 23 grounded-but-wrong verdicts. E2 (deterministic numeric evaluation) found 0 flags on the 50 human items. On these notes, a numeric check with unit conversion would have something to catch. Any such check needs its own registered item and must be judged on real notes, not on the set that suggested it.
