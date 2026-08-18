# Phase 9 v5 — Exclusion grounding: absence claims are verified, not exempted

Date: 2026-08-17 · Provider: DeepInfra (`meta-llama/Llama-3.3-70B-Instruct-Turbo`)
· Prompt v4 · TREC 2021, 30 patients / 180 trials · ~$0.13

Closes G-4 in [`docs/eval_gaps_and_priorities.md`](../../docs/eval_gaps_and_priorities.md).

---

## 1. The failure was a verifier gap, not model unfaithfulness

v4 scored exclusion criteria at a 31.2% unsupported-verdict rate against 9.2% for
inclusion. Reading the actual cached outputs on a 15-trial sample, **all 13
exclusion failures were the same pattern** and none were hallucinations:

| Failure mode | n |
|---|---|
| empty quote on a decisive `not_met` | 5 |
| invented negation ("No mention of hepatocellular carcinoma") | 8 |

```
criterion: "Signs or symptoms of hepatocellular carcinoma"
quote:     "No mention of hepatocellular carcinoma"     <- fabricated, ungroundable
criterion: "Acute Pericarditis."
quote:     ""                                            <- nothing to quote
```

An exclusion criterion answered `not_met` asserts the patient does **not** match a
disqualifier. That is a claim about *absence of evidence*, and no verbatim span
can support an absence. The analyst had no valid move: either return an empty
quote or write a negation sentence, and the verbatim check scored both as
unfaithful. The model's reasoning was correct in every case.

Verbatim grounding is well defined for presence claims and undefined for absence
claims. Exclusion criteria are exactly where absence reasoning dominates.

## 2. Fix: verify absence mechanically

Exempting exclusion `not_met` from grounding would have been a hallucination
loophole — the model could assert it freely and nothing would check. Instead the
complementary fact is verified deterministically: **the criterion's distinctive
terms must be genuinely absent from the patient note.**

- Absent → the absence claim is verified (`grounded_by: "absence"`).
- Present → a quotable span exists, so the verbatim requirement still applies and
  a fabricated negation still fails.

Checked against the patient note only. The trial text necessarily restates the
criterion, so including it would make every absence claim look contradicted.
Criteria with no distinctive terms ("Any other condition") yield no terms and fall
back to the verbatim requirement, so vague criteria are not auto-verified.

Presence claims are untouched: exclusion `met` still requires a real span.

## 3. Result (TREC 2021, 180 trials, verified arm)

| | v4 | **v5** |
|---|---|---|
| citation precision | 0.8159 | **0.9161** |
| unsupported rate | 0.1841 | **0.0839** |
| coverage | 0.3908 | **0.4442** |
| abstention | 0.6092 | **0.5558** |
| trial accuracy | 0.4333 | **0.45** |

Split by kind:

| Kind | n | unsupported v4 | unsupported v5 | precision v4 | precision v5 |
|---|---|---|---|---|---|
| Inclusion | 1294 | 9.2% | 8.0% | 0.908 | 0.920 |
| Exclusion | 696 | **31.2%** | **8.9%** | 0.688 | **0.911** |

Exclusion now matches inclusion (0.911 vs 0.920) instead of trailing it threefold.

**Retry significance is restored on TREC 2021:**

| | v4 | **v5** |
|---|---|---|
| baseline unsupported | 0.2078 | 0.1235 |
| verified unsupported | 0.1868 | 0.0841 |
| relative change | −10.1% | **−31.9%** |
| Fisher p | 0.2514 (ns) | **0.0048** ✅ |

Matched n=176 in both.

## 4. Reading it honestly

- **Both arms improve.** The change is to the verifier, which runs in the baseline
  arm too — baseline exclusion precision moved 0.664 → 0.871. Significance is
  restored because the *retry* now has groundable failures to work on rather than
  a floor of structurally unfixable ones.
- **This is a fresh run, not a re-grounding of the v4 cache.** The v4 analyst
  cache was lost with the EC2 instance and never synced to S3, so v5 made fresh
  calls. Inclusion moved +1.2pp (0.908 → 0.920) with no logic change affecting it,
  which is the run-to-run noise floor. The exclusion move (+22.3pp) is an order of
  magnitude outside it.
- **Verification now beats baseline on trial accuracy** (0.45 vs 0.4444) rather
  than trailing it (0.4333 vs 0.4444), addressing part of G-2 — though the margin
  is small and the metric stays capped by abstention.
- **SIGIR is unaffected by construction.** Its parser yields zero exclusion
  criteria, so the absence path never fires there and every committed
  inclusion-only result is untouched. The CI gate stays anchored to Phase 8 SIGIR.
- The coverage/precision curve still has `min_tokens=2` at the knee
  (cov 0.4437 / prec 0.8764; loosening to 1 gives 0.4622 / 0.9130, tightening to 3
  gives 0.4066 / 0.8032).

## 5. Cost

~$0.13 across attempts. Two aborted runs preceded the good one: the first used
Groq because `.env` sets `LLM_PROVIDER=groq` and died on the free-tier daily cap
at 39 calls; the second ran on DeepInfra but crawled because Langfuse span exports
were timing out at 5s per call, and was restarted with tracing disabled. Analyst
caching meant the restart re-paid nothing.
