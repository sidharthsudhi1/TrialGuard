# v6 — Can the analyst be told to answer every criterion? Measured: no

Run 2026-09-10. TREC 2021, DeepInfra, Llama-3.3-70B-Instruct-Turbo. Two paired
screens, 90 trials and 1,071 criteria, both arms called fresh inside one worker
so provider drift hits them equally. Total spend **$0.065**. Raw:
`l1_prompt_cost_v4_v6_trec_2021.json`, `l1_prompt_cost_v4_v6_trec_2021_n60.json`.

## The question

Under v4 the analyst silently returns fewer assessment objects than it was
handed criteria — 13.0% fewer on TREC (`production_readiness.md` §2.4). It is
not truncation: `max_tokens` is 4096 against a 1,732-token longest response. An
unanswered criterion produces no assessment, so it cannot fail grounding, cannot
trigger the retry edge, and disappears into `needs_review` — while `rollup_trial`
still calls a trial `eligible` over the subset it did receive, which CLAUDE.md
already names unsound.

v5 fixed it (13.0% → 0.5%) but changed **two** things at once: index addressing
*and* an instruction to answer every criterion. It cost 26% of TREC surfaced
recall and raised self-referential quotes 20x, so it was rejected (AD-16) and the
defect stayed open.

v6 isolates the second change. It is v4 with exactly one rule inserted and
nothing else:

```
- Return exactly one object per criterion, in the order given. Every
  criterion must appear once. Never invent a criterion that was not given.
```

Generated from v4 in code rather than copied, so "identical apart from one line"
is enforced by `test_v6_is_v4_plus_exactly_one_rule` rather than maintained by
hand. If the instruction alone recovered coverage, the fix was one sentence with
none of v5's costs.

## Result

| objects returned / criteria asked | v4 | v6 |
|---|---|---|
| screen 1 (30 trials) | 308 / 333 (0.9249) | 298 / 333 (0.8949) |
| screen 2 (60 trials) | 670 / 738 (0.9079) | 673 / 738 (0.9119) |
| **pooled** | **978 / 1071 (0.9132)** | **971 / 1071 (0.9066)** |

Difference −0.65 pp, z = −0.53. Nothing. The two screens disagree on the sign,
which is what noise looks like.

Against v5 on the same cohort and sample size:

| | coverage |
|---|---|
| v4 | 0.9249 (309/333, replicated at 308/333) |
| **v6 — told to answer everything** | **0.8949** |
| **v5 — criteria numbered** | **0.9970 (332/333)** |

Both v5 and v6 carry the completeness instruction. Only v5 recovers coverage.

**The gain was a property of numbering, not of being told.** Plausibly because a
numbered list is checkable: the model can see that it has emitted 1 through 17
and that 18 through 24 are missing, where a bulleted list of long clinical
sentences gives it nothing to count against. That is a hypothesis; what is
measured is that the instruction on its own does nothing.

## Verdict

Rejected. v6 stays registered and unfrozen behind `TG_PROMPT_VERSION=v6`, v1–v4
untouched, the same treatment R4's reranker, L4's partial retry and v5 received:
the measurement is reproducible and the default path is unchanged.

The full quality A/B was deliberately **not** run. v6 exists to answer one
question and the cheap screen answered it; spending 400 more calls on verdict
distributions for a prompt that fails its own premise would buy nothing. That
split — screen first on the mechanism, pay for quality only if the mechanism
holds — is why this cost $0.065 instead of $0.23.

## What this leaves

The prompt cannot be talked into completeness, and the one prompt that achieves
it costs more than the defect does. So the remaining candidate is not a prompt at
all: **detect the shortfall and re-ask for the missing criteria.** The graph
already has a retry edge and already re-asks for specific criteria by name — that
is exactly what the retrieval-aware grounding retry does. Extending its trigger
from "these criteria failed grounding" to "these criteria were never answered"
reuses machinery that exists, is independent of prompt version, and keeps v4.

Its cost is a second call on the ~9% of trials that come back short, against
L4's finding that a re-ask under pressure tends to produce `cannot_determine`
rather than recovery — so it needs the same criterion-level treatment before
adoption, and it may well fail the same way. Not started.

## Caveats

- One cohort. SIGIR cannot discriminate: v4 already returns 239 objects for 238
  criteria there, so there is no shortfall to recover.
- One model. The shortfall is a property of Llama-3.3-70B's behaviour on long
  criteria lists, not a general fact about the task.
- Coverage is counted off raw responses before validation, so it measures what
  the model emitted rather than what survived parsing.
