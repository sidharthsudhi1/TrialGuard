# R3: synthetic training groups. The cross-encoder crosses the bar; the synthetic data doesn't explain why

Run 2026-10-01 on EC2 (c7i.8xlarge), pre-registered in
`docs/weakpoints_fix_plan.md` §4 before any note was generated. Compute about
$7, generation about $1 of DeepInfra. Raw: `r2_crossencoder_r3.json`;
`data/cache/r3/` (notes, picks and groups, gitignored).

## What was generated

- **Trials:** 5,453 TREC trials sit outside both cohorts' candidate pools with
  ≥3 inclusion and ≥1 exclusion criteria. 2,000 were sampled (seed 0).
- **Notes:** 2,000 written, 0 generation errors. 10 dropped by `detect_phi`, 3 for
  failed keyword extraction.
- **Negatives:** 1,117 notes were dropped for having fewer than 7 negatives in
  their retrieval top-30 after excluding every pool trial. Pools cover most of
  the corpus, so a synthetic note's neighbours are usually pool trials.
- **870 synthetic groups survived.** That's about 20% more training groups in
  each direction (real: 4,452 / 3,152), a smaller dose than the 2,000 planned.

## Result

Recall@100. Δ vs deep fusion is the adopt test; Δ vs R2 is the kill test.

| arm | fold | R3 vs deep | R2 vs deep (for reference) | **R3 vs R2** |
|---|---|---|---|---|
| ce | 2021→2022 | +0.047 (p_bh 0.002) | +0.040 | +0.007 (ns) |
| ce | 2022→2021 | +0.053 | +0.053 | +0.000 (ns) |
| **ce_rrf_deep** | 2021→2022 | **+0.056** (p_bh <0.001) | +0.0495 | **+0.006** (ns) |
| **ce_rrf_deep** | 2022→2021 | **+0.054** (p_bh <0.001) | +0.0499 | **+0.004** (ns) |

Every R3-vs-deep cell is significant, at both @100 and @200. No R3-vs-R2 cell is
significant (all p_bh ≥ 0.51), and the 95% CIs all include 0.

## Against the pre-registration: both rules fire

- **Adopt:** an arm ≥ +0.05 @100 in both directions vs deep. `ce_rrf_deep`
  gives +0.056 and +0.054. **Met.**
- **Kill:** for the arm with the larger R3 mean (`ce_rrf_deep`), R3 − R2 < +0.02
  in either direction. It's +0.006 and +0.004. **Met.**

The pre-registration didn't say what happens when both fire. That is a gap in
how it was written, and it's stated here rather than resolved quietly. Read
together, the two rules say:

1. The synthetic data contributed nothing measurable: +0.004 to +0.006,
   nowhere near significance. **R3 as a technique is rejected**, as the kill rule
   says.
2. R3's model crossing the bar is therefore not evidence that the synthetic
   data works. R2 sat 0.0005 below the bar and R3 sits 0.004–0.006 above it.
   That move is inside the noise of a second training run, so it says more about
   where a seed lands than about the method.
3. What the two runs do establish together is that a fine-tuned cross-encoder
   fused with deep fusion is worth about **+0.05 recall@100 over deep fusion**:
   four independent train/test runs all land between +0.0495 and +0.056, every
   one significant. The adopt bar was set at +0.05, and the effect sits at it,
   not clearly above it.

## Caveats

- **The dose was small:** 870 groups, not 2,000, because negative mining was
  restricted to non-pool trials and the pools cover most of the corpus. A
  larger or differently mined synthetic set could behave differently. That
  would be a new experiment, not a reading of this one.
- **Each condition ran once (seed 0).** The R2-vs-R3 gap is the size you'd expect
  between seeds.

## Decision needed (not made here)

Whether to wire the cross-encoder (`ce_rrf_deep`) into the deep path, where it
costs about 7–8 s per patient for 500 candidates on CPU, rests on reading point
3 above against a bar it meets only at its edge. If adopted, the simpler R2
recipe (no synthetic data) is the one to ship, since R3 shows the extra step
buys nothing. The end-to-end check (H1 harness, top-100) comes before any
claim about surfaced recall.
