# E1b — replicating A1/A5 on a cohort with real exclusion criteria

Run 2026-08-31. Two arms, prompt v4, DeepInfra, 20 patients, top-10, both
clean (cache coverage 1.0, no budget stops, no incomplete patients).
TREC on EC2, $0.084; SIGIR locally, $0.011.

E1b existed because every trial-level number before it came from SIGIR, whose
exclusion criteria were reconstructed on 2026-08-28 from a stripped header
(`c8d9779`). TREC 2021 carries 152,542 typed exclusion criteria natively.

## Headline

| | SIGIR v4 | TREC 2021 v4 |
|---|---|---|
| retrieval recall (ceiling) | 0.1736 | 0.0480 |
| end-to-end recall (eligible tier) | 0.0139 | 0.0020 |
| surfaced recall (eligible + review) | 0.1111 | 0.0276 |
| surfaced precision | 0.3077 | 0.6885 |
| eligible-tier precision | 0.3333 | 0.4286 |
| excluded / cannot_determine / eligible | 111 / 83 / 6 | 131 / 62 / 7 |
| criterion unverifiable rate | 0.0276 | 0.0395 |

## Most of TREC's precision advantage is the pool, not the agent

|  | retrieved+labelled | gold-eligible | base rate | surfaced precision | lift |
|---|---|---|---|---|---|
| sigir | 113 | 25 | 22.1% | 30.8% | **1.39x** |
| trec_2021 | 170 | 73 | 42.9% | 68.8% | **1.60x** |

TREC labels roughly 76 eligible trials per patient, so its retrieved pool is
almost twice as rich before the agent sees anything. Read as lift over base
rate, the tier is worth 1.60x on TREC and 1.39x on SIGIR — real, better on
TREC, and far short of the 2.2x the raw precision ratio suggests. Quote the
lift, not the precision.

## A1 does not replicate

A1 (SIGIR, prompt v1) concluded the agent's loss was abstention rather than
false exclusion: 16 of 22 lost trials were `cannot_determine`, and 85 of 88
unresolved criteria were honest abstention because the note never stated the
fact. On TREC that inverts:

| | SIGIR v4 | TREC v4 |
|---|---|---|
| excluded | 55.5% | **65.5%** |
| cannot_determine | 41.5% | 31.0% |

Given real exclusion criteria the agent becomes decisive, not abstinent. A1's
diagnosis was a property of a cohort with zero native exclusion criteria, and
any plan built on "attack the abstentions" was aimed at an artifact.

## Prompt v4 is a small, two-sided change

SIGIR v1 -> v4 on identical retrieval: surfaced precision 0.2794 -> 0.3077,
surfaced recall 0.1319 -> 0.1111, `cannot_determine` 108 -> 83. Typed criteria
make the agent more decisive and slightly more precise, at some recall. Both
arms here are v4, so the SIGIR/TREC gap above is the cohort, not the prompt.

## end_to_end_recall is not a system-quality number on TREC

TREC averages 76 eligible trials per patient, so at top-10 the ceiling is
200/1520 = 0.1316. Retrieval reaches 0.048, 36% of its own ceiling, and the
agent works inside that. Reading 0.0020 as quality repeats the mistake CLAUDE.md
already records for the retired recall@10 target.

## What to trust

- Tier contract (A5): holds on both cohorts, at 1.4-1.6x lift.
- Faithfulness: unverifiable stays under 4% on both. Not the bottleneck.
- Retrieval: 0.048 of a 0.1316 ceiling on TREC is the largest single gap.
- A1's abstention diagnosis: SIGIR-only. Do not build on it.

n is 20 patients per arm. Directions are solid; second decimals are not.
