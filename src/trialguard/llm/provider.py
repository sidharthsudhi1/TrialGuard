"""Provider-agnostic chat model construction (Phase 8 WS-0).

Two call sites need an LLM: the analyst (agent/analyst.py) and keyword extraction
(retrieval/query_transform.py). Both used to construct ChatGroq inline, which made
the inference host an ambient fact about which library was imported rather than a
recorded input to the result.

Routing both through here makes (provider, model) a value the rest of the system
can read: it discriminates the analyst cache key so two hosts never share an entry,
it lands in eval report headers, and it lets the WS-3 parity gate run both arms
from one code path.

DeepInfra speaks the OpenAI wire format, so ChatOpenAI with a base_url override is
the client. That also covers Together, Fireworks, OpenRouter, and vLLM without a
new dependency — which matters because if the parity gate fails on FP8, the fix is
a different host, not a different library.
"""

from __future__ import annotations

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

_PURPOSES = ("analyst", "keywords")

# Analyst output is a JSON array of per-criterion assessments; 4096 is the cap the
# _salvage() truncation path in analyst.py is written against. Keyword extraction
# returns at most 12 short phrases.
_MAX_TOKENS = {"analyst": 4096, "keywords": 1024}


def active_provider() -> str:
    from trialguard.config import settings

    provider = settings.llm_provider.lower()
    if provider not in ("groq", "deepinfra"):
        raise ValueError(f"unknown llm_provider {provider!r}; expected groq or deepinfra")
    return provider


def active_model() -> str:
    """The model ID for the active provider, as recorded in cache keys and reports."""
    from trialguard.config import settings

    return settings.groq_model if active_provider() == "groq" else settings.deepinfra_model


def get_chat_model(purpose: str = "analyst"):
    """Return a configured chat model. `purpose` is "analyst" or "keywords"."""
    if purpose not in _PURPOSES:
        raise ValueError(f"unknown purpose {purpose!r}; expected one of {_PURPOSES}")

    from trialguard.agent.ratelimit import MAX_RETRIES
    from trialguard.config import settings

    if active_provider() == "groq":
        from langchain_groq import ChatGroq

        # These reproduce the pre-abstraction call sites exactly. The keywords path
        # deliberately keeps ChatGroq's default sampling rather than pinning
        # temperature=0: the committed keyword caches (data/cache/keywords/) were
        # generated under the defaults and back the Phase 2/7 retrieval numbers, so
        # changing sampling here would make a cache regeneration silently produce
        # different retrieval results.
        if purpose == "analyst":
            return ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                temperature=0,
                max_tokens=_MAX_TOKENS["analyst"],
                max_retries=MAX_RETRIES,
            )
        return ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model)

    import os

    from langchain_openai import ChatOpenAI

    # temperature=0 on both purposes. The Groq keywords path can't adopt this
    # without invalidating its committed cache; DeepInfra starts with a fresh
    # cache namespace, so it starts deterministic.
    #
    # Timeout is per HTTP attempt. v4 analyst output (typed inclusion+exclusion,
    # up to 24 criteria, max_tokens=4096) routinely exceeds 60s on Llama 3.3 70B;
    # 60s then raised APITimeoutError and killed the TREC eval arm. Groq's
    # MAX_RETRIES=8 would stack to tens of minutes on a dead connection, so the
    # DeepInfra client retries twice and the eval harness retries the trial.
    timeout = float(os.environ.get("TG_LLM_TIMEOUT", "180"))
    return ChatOpenAI(
        api_key=settings.deepinfra_api_key,
        base_url=DEEPINFRA_BASE_URL,
        model=settings.deepinfra_model,
        temperature=0,
        max_tokens=_MAX_TOKENS[purpose],
        max_retries=2,
        timeout=timeout,
    )


def extract_usage(response) -> dict:
    """Normalise token counts and provider-reported cost across providers.

    LangChain's `usage_metadata` is normalised and therefore lossy: it drops
    DeepInfra's `estimated_cost` extension, which survives only on the raw
    `response_metadata["token_usage"]`. The cost ledger (WS-2) prefers
    `provider_usd` when present and falls back to a local price table when it is
    None (Groq reports no cost).

    `served_model` is the model the provider says it actually ran, which is not
    necessarily the one requested — DeepInfra aliases "...-Instruct" to the FP8
    "...-Instruct-Turbo" build with no warning. Recording it is how a silent
    substitution becomes visible.
    """
    um = getattr(response, "usage_metadata", None) or {}
    rm = getattr(response, "response_metadata", None) or {}
    tu = rm.get("token_usage") or {}

    return {
        "input_tokens": um.get("input_tokens") or tu.get("prompt_tokens") or 0,
        "output_tokens": um.get("output_tokens") or tu.get("completion_tokens") or 0,
        "total_tokens": um.get("total_tokens") or tu.get("total_tokens") or 0,
        "provider_usd": tu.get("estimated_cost"),
        "served_model": rm.get("model_name") or rm.get("model"),
    }
