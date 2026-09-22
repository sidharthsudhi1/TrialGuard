# E3 — is the system actually oncology-specific?

Measured 2026-09-23 on TREC 2021, 75 patients, prompt v4. Raw:
[`e3_specialty_trec_2021.json`](e3_specialty_trec_2021.json) (retrieval, top-100)
and [`e3_specialty_trec_2021_top10.json`](e3_specialty_trec_2021_top10.json)
(end to end, top-10). **Cost $0**: cache coverage was 1.000, so all 749
assessments replayed from disk.

No data was added. The same 75 patients are partitioned by whether the note
mentions a cancer term, and each group is scored with `end_to_end.score`
unchanged, so the groups compose to the published aggregate by construction.

## Why this needed doing

CLAUDE.md scoped production to oncology, and the standing end-to-end numbers were
read as oncology numbers because of it. **They never were.** 61 of the 75 scored
TREC 2021 topics never mention a cancer term, gold coverage is 1.000 for both
groups, and the published recall@100 of 0.440 is 81% non-oncology by patient count
and by gold volume. The headline was always a mixed number that nobody labelled.

## Retrieval: oncology is the weaker specialty

| group | patients | gold eligible | recall@100 | ceiling @10 |
|---|---|---|---|---|
| all | 75 | 5,569 | **0.4393** | 0.0868 |
| oncology | 14 | 1,069 | **0.3052** | 0.0437 |
| non-oncology | 61 | 4,500 | **0.4701** | 0.0967 |

`all` reproduces the published 0.440, which is the check that the partition is
sound rather than a new measurement.

**Non-oncology retrieves 54% better than oncology.** That inverts what the scope
lock implied. The likely cause is not model quality but corpus density: 26k
recruiting oncology trials compete for the same top-100 slots with heavily
overlapping eligibility language, while a cardiology or neurology patient's gold
sits in a thinner neighbourhood. MedCPT is a general biomedical encoder, not a
cancer-specific one, so it has no compensating advantage in the crowded region.

## End to end: the lift holds in both, and is larger on oncology

Top-10 pools, tiered roll-up, against each group's own pool base rate.

| group | pool | base rate | tier precision | **lift** | e2e recall |
|---|---|---|---|---|---|
| all | 749 | 0.3993 | 0.6598 | **1.65x** | 0.0095 |
| oncology | 140 | 0.3070 | 0.5833 | **1.90x** | 0.0047 |
| non-oncology | 609 | 0.4204 | 0.6749 | **1.61x** | 0.0107 |

**The tiered contract is worth more on oncology (1.90x) than on everything else
(1.61x)**, despite oncology retrieving worse, because oncology's pool arrives
poorer (base rate 0.307 vs 0.420) and therefore has more for the agent to filter.
End-to-end recall runs the other way (non-oncology 2.3x higher) because that is
dominated by retrieval, not by the agent.

So the two halves of the system have **opposite** specialty profiles: retrieval is
worse on oncology, the agent is better on it. The aggregate hid both.

## Faithfulness: also better on oncology

| group | criterion unverifiable | grounded rate | self-referential | weak absence |
|---|---|---|---|---|
| all | 0.0310 | 0.6375 | 11 (0.36%) | 17.2% |
| oncology | **0.0138** | 0.6136 | **0** | **13.2%** |
| non-oncology | 0.0355 | 0.6438 | 11 (0.45%) | 18.2% |

The faithfulness floor holds in both groups and is 2.6x cleaner on oncology. Zero
self-referential citations on 621 grounded oncology criteria. The entailment-gap
proxies (AD-15, AD-19) are both smaller there too.

## What this does not claim

**14 oncology patients is a small cell and the agent figures rest on it.** Tier
precision of 0.5833 comes from 54 surfaced trials, so its 95% interval is roughly
±0.13 and the 1.90x lift is about ±0.4. The *direction* is supported by three
independent metrics moving together (lift, unverifiable rate, weak-absence rate);
the magnitudes are not tight. Retrieval is on firmer ground: 1,069 gold trials.

**The classifier is a regex over disease stems** in the patient note, chosen for
recall over treatment terms that misfile non-oncology notes. It puts 14 patients
in the oncology cell against 12 by a narrower earlier pattern; borderline notes
exist and were not adjudicated.

**Top-10 for the agent half, not top-100.** The specialty contrast does not need
pool depth since both groups face the same depth, and top-100 would have been
7,500 assessments against a $2 daily cap. Absolute end-to-end recall here is
therefore far below the published top-100 figures and is not comparable to them.

**One cohort.** TREC 2022 and SIGIR unrun.

## Consequence

The oncology scope lock was not protecting a specialty advantage. On this cohort
the system retrieves *better* outside oncology and reasons *better* inside it, and
the tiered-contract lift, which is what the thesis actually claims, holds in both
cells at 1.61x and 1.90x against a published headline of 1.60x.

The honest generalisation claim is therefore stronger than the one the repo was
making, and it is now measured rather than assumed: **nothing in the system's
value depends on the patient having cancer.**
