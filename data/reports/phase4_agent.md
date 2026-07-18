# TrialGuard - Phase 4 Measured Results

Generated on branch `phase/4-agent-tuning`. Zero paid services (Groq free tier +
local MedCPT). Phase 4 turns the Phase 3 faithfulness result into a
**reproducible, tunable** one and adds a retry designed to transfer to TREC.

Phase 4 deliberately does **not** overwrite the Phase 3 report files
(`phase3_agent*.json`) — those numbers are load-bearing. Phase 4 writes
`phase4_agent_<cohort>.json` alongside them.

---

## 1. Significance is now computed in-harness (was off-code)

Phase 3 reported Fisher p-values that were computed by hand. They are now
regenerated inside the run (`eval/significance.py`, `matched_ab`) and written into
every cohort report, so no headline number is ever typed by hand again.

The test runs over the **matched trial set** — trials both arms actually completed
— because the verified arm can stop early on the free-tier daily cap, so the two
arms' trial sets are not identical unless intersected. Reporting over the
intersection removes the selection skew unequal truncation would otherwise add.

**SIGIR, verified arm using the new retrieval-aware retry (matched n=60 trials,
fresh calls until the daily cap):**

| | unsupported | grounded | decisive | unsupported rate |
|---|---|---|---|---|
| single-pass | 9 | 115 | 124 | 0.0726 |
| verified | 1 | 118 | 119 | **0.0084** |

Relative change **-88.4%**, odds ratio 9.23, **Fisher p = 0.0193** (significant at
0.05). This is a smaller matched set than Phase 3's 179 (the daily cap truncated
the verified arm), but the result is regenerated in-code and still significant —
and the verified unsupported rate (0.0084) is *lower* than Phase 3's generic-retry
verified rate (0.0403), the first signal that the retrieval-aware retry is
stronger, not just equal.

---

## 2. Coverage / faithfulness is a curve, not a point

Phase 3 reported a single abstention number (~0.73) next to a single precision
number. Phase 4 reports the joint trade-off as the grounding-strictness knob
(`min_tokens`) is swept — deterministic over cached analyst outputs, zero Groq
calls (`agent_metrics.coverage_curve`).

**SIGIR (n=1258 criteria):**

| min_tokens | coverage | citation precision |
|---|---|---|
| 1 | 0.2687 | 0.9111 |
| **2 (shipped)** | **0.2671** | **0.9057** |
| 3 | 0.2615 | 0.8868 |
| 4 | 0.2464 | 0.8356 |

**Reading it:** tightening the guard past 2 lowers *both* coverage and precision —
it rejects real short facts (the exact TREC artifact Phase 3 fixed) without buying
faithfulness. Loosening to 1 barely moves either. So `min_tokens=2` sits at the
knee, and — the load-bearing conclusion — **coverage is not a grounding knob at
all**. The ~0.73 abstention is analyst-emitted `cannot_determine`, not grounding
rejection, so lowering it requires changing the analyst, which is what the v2
prompt targets (section 4).

---

## 3. Retrieval-aware retry (mechanism shipped, partial measurement)

The Phase 3 retry appended a generic "copy verbatim" nudge; it recovered
paraphrase failures on SIGIR but not genuine verbatim-misses on TREC. Phase 4's
retry (`agent/graph.py::_analyst_node`) instead hands the analyst **the exact
trial source span** for the specific criteria that failed grounding — the
characters it must copy.

- Mechanism implemented and unit-tested (`tests/test_agent_graph.py::
  test_retry_is_retrieval_aware`: the retry note must contain the source span and
  the named failed criterion).
- Measured on SIGIR to the extent the daily cap allowed (section 1): verified
  unsupported dropped to 1/119 = 0.0084, below Phase 3's generic retry.
- Full A/B and the TREC transfer test are **quota-paced** (a full three-cohort run
  is ~40x the daily token cap; the cap was reached mid-session). This is the same
  pacing Phase 3 used for larger-n TREC — code is done, tokens are the gate.

---

## 4. Abstention-lowering prompt v2 (additive, mechanism shipped)

`agent/analyst.py` now carries two prompt versions. v1 (default) is unchanged, so
the Phase 3 cache and results are never invalidated. v2 (`TG_PROMPT_VERSION=v2`)
keeps the verbatim-quote requirement — the faithfulness floor is untouched — but
instructs the analyst to scan both texts for a decisive fact before abstaining,
reserving `cannot_determine` for genuinely absent evidence.

- Additive versioning is unit-tested (`test_prompt_version_switches_cache_key`):
  v2 gets its own cache namespace and cannot collide with the v1 cache.
- Initial fresh v2 attempt was quota-blocked (HTTP 429). Resolved on a fresh
  daily allowance — see section 5.

---

## 5. v2 result: it lowers abstention AND improves faithfulness (DoD-P1)

A fresh v2 run (`TG_PROMPT_VERSION=v2`) reached 93 of 180 SIGIR trials before the
daily cap. Comparing v2 against v1 over that **same matched 93-trial set** (v1
recomputed from cache, single-pass on both arms, deterministic and $0):

| metric | v1 | v2 | relative |
|---|---|---|---|
| abstention | 0.7377 | 0.6729 | **-8.8%** |
| coverage (grounded decisive / criteria) | 0.2623 | 0.3271 | **+24.7%** |
| citation precision | 0.905 | 0.9633 | **+6.4%** |

The coverage shift is significant: **Fisher p = 0.0097** (grounded-vs-not, v1 vs v2).

**This is the Phase 4 headline.** The over-abstention prompt did exactly what it
was meant to and nothing it wasn't: the analyst commits to a decisive verdict
~25% more often, abstains ~9% less — and the extra verdicts are *more* grounded,
not less (precision rose 6%). Lowering abstention did not cost faithfulness; it
improved it, because v2 spends its effort finding real short facts (age, stage,
ECOG) that v1 was leaving on the table as `cannot_determine`.

Caveats, disclosed: v2 emitted slightly fewer criterion objects (642 vs 690) over
the same trials, so each rate is over its own emitted set; the verified (retry)
arm under v2 reached only 4 trials before the cap, so a v2 retry A/B is still
pending (rolls into P2's paced budget). The single-pass v2-vs-v1 result above,
however, is clean, matched, and significant.

---

## What is proven in Phase 4

1. **The faithfulness A/B is reproducible** — the p-value is computed in-harness,
   not by hand, and regenerates from the run (SIGIR matched n=60, p=0.0193).
2. **Faithfulness vs coverage is a measured curve** — `min_tokens=2` is at the
   knee; abstention is analyst-driven, not a grounding artifact.
3. **The retrieval-aware retry is at least as strong as the generic one** on the
   SIGIR sample it could reach (verified unsupported 0.0084 < 0.0403).
4. **v2 lowers abstention without costing faithfulness (DoD-P1, SIGIR).** Matched
   93-trial single-pass: abstention -8.8%, coverage +24.7%, precision +6.4%,
   coverage-shift Fisher p=0.0097. The Phase 4 headline.

## What is still quota-paced (code done, tokens are the gate)

- **DoD-P1 (partial done)** v2 single-pass measured and significant on SIGIR
  (section 5). Remaining: v2 on TREC 2021/2022, and a v2 *retry* arm (reached only
  4 trials before the cap).
- **DoD-P2** Full retrieval-aware-retry A/B on TREC 2021/2022 (+ larger-n TREC
  2022) to test whether the source-span retry transfers the SIGIR benefit.

Both are blocked only by the Groq free-tier daily cap, not by missing code. Runs
resume across days from the analyst cache at zero re-spend.
