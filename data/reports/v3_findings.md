# V3: NIH-TrialBench target recovery

Run 2026-11-10. The TrialGPT 2.0 repo now carries an NLM public-domain notice (checked the same day; the plan's license block was an empty LICENSE). 126 synthetic vignettes, 1,373 candidate trials, every target file present. Each case is ranked inside its own candidate list with list depth equal to that corpus, MedCPT hybrid, exclusion text included. Notes were not sent to the public API: several summaries contain calendar dates, which `detect_phi` refuses. The clone stays in `data/cache/trialbench` and is not committed. Raw: `v3_trialbench_recovery.json`.

| arm | n | hit@1 | hit@5 | hit@10 | hit@50 |
|---|---|---|---|---|---|
| raw note | 126 | 0.214 | 0.492 | 0.667 | 0.897 |
| keywords | 126 | 0.278 | 0.635 | 0.770 | 0.976 |

Hit@k is the fraction of vignettes whose target trial is in the top k of that case's candidate set.

## Decision

DONE-ADOPTED as a third cohort for retrieval. It is not on the served API.
