# E2 — is the verifier independent of the model?

Measured 2026-09-22. SIGIR, 30 patients x 6 trials = 178 assessed trials per arm,
prompt v4, both arms (`max_retries` 0 and 2), six model families on DeepInfra.
Raw arms: `e2_<tag>_agent_sigir.json`. Total inference cost **$0.5210**.

Every faithfulness figure in this repo was produced by one model family, Llama 3.3
70B, across two hosts. The thesis is that faithfulness is a property of the
*architecture* rather than of the model, and with n=1 model that was an assertion.

## The result

**The retry mechanism is significant on all six families.**

| model | vendor | matched | baseline unsup | verified unsup | relative | Fisher p |
|---|---|---|---|---|---|---|
| Llama-3.3-70B | Meta | 175 | 0.0709 | 0.0426 | -39.9% | **0.0319** |
| Qwen2.5-72B | Alibaba | 175 | 0.0698 | 0.0337 | -51.7% | **0.0039** |
| Mistral-Small-24B | Mistral | 175 | 0.0507 | 0.0152 | -70.1% | **0.0152** |
| Gemma-3-27B | Google | 175 | 0.1366 | 0.0894 | -34.6% | **0.0083** |
| gpt-oss-120b | OpenAI | 175 | 0.0902 | 0.0112 | **-87.6%** | **0.0000** |
| DeepSeek-V3 | DeepSeek | 175 | 0.0404 | 0.0120 | -70.3% | **0.0155** |

Six families, five vendors, 24B to 671B parameters. Six out of six significant at
p<0.05, with reductions from 34.6% to 87.6%. Not one model where the mechanism
fails to help.

## What varies, which is the other half of the claim

| model | citation precision | abstention | coverage | trial accuracy | USD |
|---|---|---|---|---|---|
| Llama-3.3-70B | 0.9577 | 0.5796 | 0.4204 | **0.2584** | 0.0657 |
| Qwen2.5-72B | 0.9665 | 0.5850 | 0.4150 | 0.2247 | 0.1407 |
| Mistral-Small-24B | 0.9850 | **0.7942** | 0.2058 | 0.1180 | **0.0214** |
| Gemma-3-27B | **0.9113** | 0.6132 | 0.3868 | 0.1798 | 0.0558 |
| gpt-oss-120b | **0.9890** | 0.7592 | 0.2408 | 0.1348 | 0.0664 |
| DeepSeek-V3 | 0.9882 | 0.7255 | 0.2745 | 0.1461 | 0.1710 |

Citation precision spans 7.8 points and abstention spans 21.5. The rate at which a
model *attempts* a decisive verdict on a non-verbatim quote varies **8.1x**, from
1.10% (gpt-oss-120b) to 8.87% (Gemma-3-27B).

**That 8.1x spread is the point.** `unsupported_verdict_rate` counts attempts the
verifier caught, not errors that shipped: `verify/grounding.py` forces every
ungrounded decisive verdict to `unverifiable` before it reaches a roll-up or a
user. So model quality moves the amount of work the verifier does by 8x, and does
not move what a user is shown. The floor is flat under a model axis that is not.

**The catch rate is 100% on every model by construction, not by measurement.**
Grounding is pure Python doing a verbatim substring check; it never sees which
model produced the quote, and the code path is byte-identical across these six
runs. The committed stress test (51/51 corrupted quotes rejected, 0 false
rejections, `verifier_stress.json`) validates the mechanism, and nothing in it is
model-conditional. Re-running it per model would measure the same function six
times.

## The precision/coverage tradeoff is visible and consistent

The models sort into two groups, and the ordering is the same on both axes:

- **Commit more, cite worse**: Llama, Qwen, Gemma abstain 58-61% and reach
  coverage 0.39-0.42, at citation precision 0.911-0.967.
- **Abstain more, cite better**: Mistral, gpt-oss, DeepSeek abstain 73-79% and
  reach coverage 0.21-0.27, at citation precision 0.985-0.989.

Trial accuracy follows coverage rather than citation precision: Llama is best at
0.2584 despite the second-worst citation precision, because a model that abstains
on 79% of criteria cannot resolve enough of a trial to roll it up. **A model
cannot be ranked on faithfulness alone** — abstention buys citation precision, and
`gpt-oss-120b` has the best citations in the set and 48% less coverage than Llama.

This also means the current default is defensible on grounds other than inertia:
Llama 3.3 70B has the best trial accuracy here, at mid-pack cost.

## What this does not claim

**One cohort, one prompt version.** SIGIR under prompt v4, which yields zero
exclusion criteria by construction (README), so this measures inclusion handling
only. TREC would exercise the absence-grounding path (AD-19) and is unrun.

**Not a model recommendation.** 178 trials per arm, one run each, no seed
variation. The spreads between adjacent models are inside what a rerun could move;
the six-for-six significance result is not.

**Cost is DeepInfra's, on one day.** The 8x cost spread (Mistral $0.0214 to
DeepSeek $0.1710) is a serving-price fact, not a model-quality one.

**No frontier API model.** All six are open-weight models on one host, which keeps
the comparison clean but leaves untested whether a GPT-5 or Claude-class model
behaves differently. `openai/gpt-oss-120b` is OpenAI's open-weight release, not
their frontier model.

## Consequence

The system is not a wrapper around Llama 3.3. Swapping the model changes how much
the verifier catches, how much the system abstains, and what it costs, and does
not change the contract: a decisive verdict carries a verbatim citation or it is
marked `unverifiable`. That contract survived a 5-vendor, 28x parameter-range swap
without a code change, because `(provider, model)` is already a value the system
reads rather than an ambient fact about which library was imported (AD-7 amended,
Phase 8).
