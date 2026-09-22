# E1c — does retrieval survive leaving oncology?

Measured 2026-09-22. TREC 2021 gold held fixed while the haystack grows with real
all-conditions ClinicalTrials.gov records. Raw rows:
[`e1c_distractor_recall.json`](e1c_distractor_recall.json).

Every recall number in this repo was measured on a corpus already narrowed to
oncology (CLAUDE.md locks the scope). A buyer's corpus is not pre-filtered, so the
question is what the embedder does when the distractors stop being on-topic.

## Setup

99,098 distractors fetched from CT.gov with the same recruiting-status filter and
**no condition filter**, deduped against the eval corpus by NCT id (829 overlapped
and were dropped). They are genuinely off-topic — the first record is robot-assisted
plication for diaphragmatic paralysis.

Gold, queries and the base corpus are exactly what the committed eval uses, so the
zero-distractor row reproduces the standing number and every later row is
comparable to it by construction. Distractors are embedded through production's own
`eligibility_text_for_embedding`, not a reimplementation.

## Result

| corpus | multiple | recall@50 | recall@100 | recall@200 | retention@100 | chance |
|---|---|---|---|---|---|---|
| 26,149 | 1.00x | 0.1039 | 0.1759 | 0.2641 | 1.000 | 1.000 |
| 36,149 | 1.38x | 0.0968 | 0.1620 | 0.2484 | 0.921 | 0.723 |
| 51,149 | 1.96x | 0.0898 | 0.1504 | 0.2287 | 0.855 | 0.511 |
| 76,149 | 2.91x | 0.0780 | 0.1323 | 0.2024 | 0.752 | 0.343 |
| 125,247 | **4.79x** | 0.0651 | **0.1102** | 0.1713 | **0.626** | 0.209 |

**Recall degrades, and it degrades far slower than the corpus grows.** A 4.79x
haystack costs 37.4% of recall@100. If ranking carried no signal, recall would fall
with 1/N to 20.9% of baseline; it holds 62.6%, which is **3.0x better than
chance**. The same shape holds at every depth (retention 0.626 at k=50, 0.626 at
k=100, 0.649 at k=200), so this is not an artifact of one cut-off.

The reading: MedCPT is doing real work separating an oncology patient from
cardiology, psychiatry and surgical trials. It is not, however, doing enough of it
for the pool depth to stay constant — a corpus 5x wider needs a deeper pool to
surface the same trials, and `docs/structural_recall_plan.md` already establishes
that pool depth is the binding constraint rather than the embedder.

## What this does not claim

**The absolute level is not production's.** This is dense-only, raw-note query —
no keyword extraction, no lexical arm, no RRF. Production runs all three and
recall@100 on this cohort is 0.440, not 0.1759. Lexical and RRF are held out on
purpose: mixing them in would confound "does the embedder still separate" with
"does FTS still separate". **Do not quote 0.1102 as a system number.** The finding
is the shape of the curve, not its height.

Production would likely degrade *less*, because keyword queries and a lexical arm
add signal that is independent of the embedder — but that is a hypothesis and the
measurement to settle it is the same harness with `--use-keywords`, unrun.

**Distractors are recruiting-status only**, matching production. A full-history
corpus (~500k, all statuses) is roughly 4x larger again and unmeasured.

**One cohort, 75 queries.** TREC 2021 only.

## Consequence for E1a

The all-conditions corpus is **125,247 trials**, and [E1a](e1a_findings.md) puts the
in-process matrix / HNSW crossover near 50k rows. So leaving oncology crosses that
threshold twice over: at 125k the matrix would need 0.38 GB resident and serve at
roughly 7 ms against HNSW's flat ~2.5 ms, with the gap widening as the corpus
grows. **Widening the condition scope and keeping the matrix are not independent
decisions** — the first invalidates the second.
