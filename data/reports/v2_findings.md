# V2: the end-to-end headline on every patient

Run 2026-09-28 to 2026-09-30 on EC2 (m7i.large, Sydney), top-100, every patient
with an in-corpus eligible trial. Served config: current defaults plus prompt v4,
DeepInfra Llama-3.3-70B. TREC 2022 uses `TG_INDEX_EXCLUSION=0`, because only that
index is cached; it's the same variant as every prior 2022 number.

17,628 assessments. **Every cohort finished at `completion` 1.0 with 0 errors.**
DeepInfra spend was $9.52. Raw: `v2_e2e_{trec_2021,trec_2022,sigir}_top100.json`.
CIs are patient-bootstrap 95% (V1). Lift counts labelled trials on both sides (AD-34).
Measured at per-keyword list depth 50; since AD-35 the default is 200, so rerun
with `TG_LIST_POOL=50` to reproduce.

## Headline

| cohort | n | retrieval ceiling | **surfaced recall** | precision | base rate | **lift** | unverifiable |
|---|---|---|---|---|---|---|---|
| TREC 2021 | 75 | 0.339 [0.301, 0.382] | **0.207** [0.181, 0.235] | 0.590 | 0.386 | **1.53x** [1.45, 1.62] | 0.0288 |
| TREC 2022 | 50 | 0.392 [0.331, 0.462] | **0.275** [0.231, 0.325] | 0.535 | 0.406 | **1.32x** [1.24, 1.40] | 0.0178 |
| SIGIR | 52 | 0.756 [0.683, 0.821] | **0.503** [0.439, 0.558] | 0.270 | 0.184 | **1.47x** [1.36, 1.60] | 0.0203 |

Against the published n=20 table (TREC top-100, older flags):

| cohort | published surfaced recall | full-cohort | published lift | full-cohort |
|---|---|---|---|---|
| TREC 2021 | 0.1770 | **0.2065** | 1.61x | **1.53x** |
| TREC 2022 | 0.2580 | **0.2749** | 1.45x | **1.32x** |

## Reading

- **Recall was understated, and lift was overstated.** Both moves come from
  the patient mix and the retrieval changes since H1 (keyword decay adds about 4
  points to the TREC 2021 retrieval ceiling: 0.298 → 0.339 on the full cohort).
  They are not a change in the agent.
- **The tier beats its pool on every cohort.** No lift CI comes near 1.0. The
  smallest lower bound is 1.24x (TREC 2022).
- **TREC 2022's published 1.45x sits outside its full-cohort CI** [1.24, 1.40].
  That number should not be quoted any more.
- **SIGIR at top-100 is new.** Pools there are gold-sparse (base rate 0.184),
  so precision is low while lift is comparable to TREC. That's the reason this
  repo quotes lift rather than precision.
- **Faithfulness holds at scale.** Criterion unverifiable rate is 1.8–2.9% on
  every cohort, inside the 2.8–4.0% previously published at top-10.

## Headline patients vs the rest, same run

| cohort | patients | surfaced recall | lift |
|---|---|---|---|
| TREC 2021 | H1's 20 | 0.177 [0.143, 0.212] | 1.65x [1.48, 1.87] |
| TREC 2021 | other 55 | 0.218 [0.185, 0.255] | 1.48x [1.39, 1.59] |
| TREC 2022 | H1's 20 | 0.280 [0.212, 0.366] | 1.37x [1.26, 1.50] |
| TREC 2022 | other 30 | 0.272 [0.217, 0.339] | 1.28x [1.20, 1.39] |

This replicates V1's top-10 finding at top-100. The 20 headline patients give a
higher lift on both cohorts, and TREC 2021's are also the lower-recall ones. On
2022 the same 20 patients give 1.37x under current flags against the published
1.45x, so there the config change matters as well as the sample.

## Operational notes

- **Pass 1 exposed a harness bug.** `Executor.map` ends the run on the first
  worker exception; fixed in 66161d2. The first TREC 2021 pass stopped after 279
  of about 7.5k pairs while reporting coverage 1.0. All figures above come from
  pass 3, run with the fix.
- **Throughput** was 7–13 pairs/min at 10 workers and `TG_LLM_TIMEOUT=300`, with
  zero timeouts. Cost was about $0.0005 per pair.
- **The analyst cache grew 13,889 → 39,399 entries.** Every cell above now reruns
  cached-only at $0.

## Top-10 and the widening, full cohorts ($0)

Top-10 pools are a prefix of the top-100 run, so these reran cached-only at
completion 1.0: `v2_e2e_{cohort}_top10.json`.

| cohort | top-10 surfaced recall | top-10 lift | top-100 surfaced recall | ratio |
|---|---|---|---|---|
| TREC 2021 | 0.034 [0.028, 0.042] | 1.37x [1.26, 1.49] | 0.207 | 6.1x |
| TREC 2022 | 0.042 [0.031, 0.055] | 1.17x [1.08, 1.26] | 0.275 | 6.6x |
| SIGIR | 0.163 [0.127, 0.198] | 1.48x [1.29, 1.70] | 0.503 | 3.1x |

Paired per patient, 10 → 100 (`v2_widening.json`): 75/75, 49/50 and 40/52
patients improve, none gets worse, and BH-adjusted p < 0.0001 on each cohort.
H1's conclusion holds on the full cohorts. On TREC, lift *rises* with depth
(1.37x → 1.53x, 1.17x → 1.32x): the agent adds more over its pool as the pool
gets poorer.

The README headline table now carries these six rows. The n=20 table is kept
under a History fold.
