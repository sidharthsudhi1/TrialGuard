# Phase 8 WS-3 — Provider parity gate: Groq bf16 vs DeepInfra FP8

**Verdict: PASS.** Quantization did not degrade verbatim quoting. Recommend
flipping the default provider to DeepInfra.

Date: 2026-08-02 · Cohort: SIGIR, 30 patients × 6 trials = 180 trials · Prompt v1

---

## What was being tested

DeepInfra serves **only** `meta-llama/Llama-3.3-70B-Instruct-Turbo`, an FP8
build; requesting the plain `-Instruct` ID is silently aliased, and no
unquantized variant exists on the platform. So the migration is not "same
weights, different host" — it changes numerical precision on the model that
produces verbatim quotes.

This matters because the failure would be **silent**. `verify/grounding.py` does
a verbatim substring match; a quote that comes back paraphrased just fails
grounding and is downgraded to `unverifiable`, which is a legitimate output the
UI displays as a feature. A degraded provider looks like a slightly more cautious
system, not like an error.

---

## The comparison is asymmetric, and that has to be said first

The committed Groq run (`phase4_agent_sigir.json`) has **`rate_limited: true` on
its verified arm** — the free-tier daily cap killed it after **60 of 180
trials**. Its baseline arm completed all 180.

So the two runs are not symmetric, and the honest reading splits in two:

- **Baseline arm (max_retries=0): directly comparable.** Both hosts, 180 trials,
  no retry, no truncation. This is the clean measurement of whether FP8 changed
  the model's quoting behavior.
- **Verified arm (max_retries=2): not directly comparable.** 60 trials against
  180, different trial mix. Judged against the committed gate floors instead of
  against the Groq figure.

---

## Baseline arm — the clean comparison

| Metric | Groq (bf16) | DeepInfra (FP8) | Δ |
|---|---|---|---|
| `citation_precision` | 0.9057 | **0.9086** | +0.0029 |
| `unsupported_verdict_rate` | 0.0943 | **0.0914** | −0.0029 |
| `coverage` | 0.2671 | **0.2958** | +0.0287 |
| `abstention_rate` | 0.7329 | **0.7042** | −0.0287 |
| `n_criteria` | 1258 | 1244 | −14 |
| `n_trials` | 180 | 180 | — |

**Citation precision is unchanged** (+0.3pp, well inside run-to-run noise). That
is the metric that would move if FP8 had degraded verbatim reproduction, and it
did not.

Coverage is ~2.9pp **higher** on FP8, with abstention correspondingly lower — the
model committed to slightly more decisive verdicts while grounding them at the
same rate. Not claimed as an improvement: it is one run at n=180, and the
direction could reverse. The claim is only that there is no degradation.

---

## Verified arm — against the committed floors

Judged against `data/reports/baselines.json`, the thresholds the CI gate enforces:

| Gate | DeepInfra value | Op | Threshold | |
|---|---|---|---|---|
| `verified.unsupported_verdict_rate` | 0.0287 | ≤ | 0.05 | PASS |
| `significance.fisher_p` | 0.0004 | ≤ | 0.05 | PASS |
| `significance.relative_change` | −0.6855 | ≤ | −0.3 | PASS |
| `baseline.coverage` | 0.2958 | ≥ | 0.20 | PASS |

Matched-set Fisher exact, both hosts:

| | Groq (60 trials) | DeepInfra (177 trials) |
|---|---|---|
| matched trials | 60 | **177** |
| baseline unsupported | 9 / 124 (0.0726) | 37 / 398 (0.0930) |
| verified unsupported | 1 / 119 (0.0084) | 10 / 342 (0.0292) |
| relative change | −0.8842 | −0.6855 |
| odds ratio | 9.2348 | 3.4028 |
| Fisher p | 0.0193 | **0.0004** |

**On the relative-change difference (−0.88 vs −0.69):** the Groq figure comes from
9 → 1 unsupported verdicts on 60 trials. A single-count difference at that scale
swings the ratio hard, and the truncated 60 are not a random subsample — they are
whichever trials ran before the cap hit. The DeepInfra figure rests on 37 → 10
across 177 matched trials.

Stating the limit plainly: **a like-for-like verified-arm comparison does not
exist**, because Groq's verified arm at n=180 was never measured. This run cannot
rule out a modest real difference in retry efficacy. What it does establish is
that FP8 baseline quoting matches full precision, and that the verified arm clears
every committed floor with roughly an order of magnitude more statistical power
than the number it is being compared against.

This is also the **first complete SIGIR verified arm** the project has. The Groq
one has been truncated since Phase 4.

---

## Minor behavioral note

`n_criteria` differs by 14 (1244 vs 1258, ~1%) across the same 180 trials. The
count reflects how many assessment objects the analyst emitted, so the two builds
enumerate criteria very slightly differently. Too small to affect any rate metric
here, but worth watching if it grows — it would point at truncation or salvage
behavior (`analyst._salvage`) rather than at grounding.

---

## Grounding strictness curve (DeepInfra)

| `min_tokens` | coverage | citation precision |
|---|---|---|
| 1 | 0.2966 | 0.9111 |
| **2** | **0.2958** | **0.9086** |
| 3 | 0.2886 | 0.8864 |
| 4 | 0.2685 | 0.8247 |

`min_tokens=2` still sits at the knee, matching the Phase 4 finding on Groq. No
retune needed.

---

## Cost and provenance

| | |
|---|---|
| Model served | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (confirmed per-call) |
| Fresh calls, this run | 129 |
| Total DeepInfra calls | 221 (includes 92 from an interrupted first attempt) |
| Tokens | 206,819 |
| **Spend** | **$0.0422** |
| Wall clock | 47.5 min |
| Cost source | `provider` (DeepInfra's own `estimated_cost`) on every call |

The first attempt was killed externally at 92 calls. It resumed from the analyst
disk cache at zero cost — the provider-discriminated cache key (WS-1) is what
made resuming safe, since the DeepInfra namespace cannot touch the Groq entries
backing Phase 3/4.

For scale: the full run cost **4.2 cents**. The Phase 4 work this unblocks was
stalled for weeks on a free-tier quota.

---

## Recommendation

1. Flip `settings.llm_provider` default to `deepinfra`.
2. Proceed to WS-4 (Phase 4 carryover: prompt v2 A/B, TREC retry A/B).
3. Leave the Groq arm runnable — it reproduces Phase 3/4 from cache, and the
   committed reports remain the reference.
4. Do **not** update `baselines.json` to the DeepInfra numbers yet. The gate
   should keep enforcing the Groq-derived floors until WS-4 establishes what the
   post-migration baseline actually is.
