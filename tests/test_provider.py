"""Provider abstraction (Phase 8 WS-0). No network: construction and shape only."""

import pytest

from trialguard.config import settings
from trialguard.llm.provider import active_model, active_provider, extract_usage, get_chat_model


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "test-key")


@pytest.fixture
def deepinfra(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "deepinfra")
    monkeypatch.setattr(settings, "deepinfra_api_key", "test-key")


def test_default_provider_is_groq():
    """DeepInfra since the WS-3 parity gate passed (phase8_provider_parity.md).
    Pinned because changing the default silently changes which model produced
    every fresh result, and that must be a deliberate act."""
    from trialguard.config import Settings

    assert Settings(_env_file=None).llm_provider == "deepinfra"


def test_active_model_tracks_provider(groq):
    assert active_model() == settings.groq_model
    assert active_provider() == "groq"


def test_deepinfra_model_is_the_served_turbo_id(deepinfra):
    """DeepInfra aliases "...-Instruct" to the FP8 "...-Instruct-Turbo" build with
    no warning. The configured ID must be the one actually served, or the cache
    key claims two different builds are the same model."""
    assert active_model().endswith("-Turbo")


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    with pytest.raises(ValueError, match="unknown llm_provider"):
        active_provider()


def test_unknown_purpose_rejected(groq):
    with pytest.raises(ValueError, match="unknown purpose"):
        get_chat_model("summariser")


def test_groq_analyst_params_unchanged(groq):
    """The pre-abstraction analyst call site pinned temperature=0 / max_tokens=4096.
    Drift here would change every fresh Phase 3/4 assessment.

    ChatGroq coerces temperature=0 to 1e-08 because Groq's API rejects a literal
    zero; DeepInfra accepts 0. Both are effectively greedy, but they are not the
    same request — one more reason the parity gate (WS-3) measures rather than
    assumes."""
    llm = get_chat_model("analyst")
    assert llm.temperature == pytest.approx(0, abs=1e-6)
    assert llm.max_tokens == 4096


def test_groq_keywords_keeps_default_sampling(groq):
    """Committed keyword caches were generated under ChatGroq's default sampling
    and back the Phase 2/7 retrieval numbers. Pinning temperature here would make
    a cache regeneration silently produce different retrieval results."""
    llm = get_chat_model("keywords")
    assert llm.temperature != 0


def test_deepinfra_points_at_openai_compat_endpoint(deepinfra):
    llm = get_chat_model("analyst")
    assert "deepinfra.com" in str(llm.openai_api_base)
    assert llm.temperature == 0


def test_extract_usage_reads_deepinfra_cost_from_response_metadata():
    """usage_metadata is normalised and lossy: it drops DeepInfra's estimated_cost.
    The ledger must read the raw token_usage block instead."""

    class Resp:
        usage_metadata = {"input_tokens": 25, "output_tokens": 6, "total_tokens": 31}
        response_metadata = {
            "token_usage": {"prompt_tokens": 25, "completion_tokens": 6, "estimated_cost": 4.42e-06},
            "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        }

    usage = extract_usage(Resp())
    assert usage["input_tokens"] == 25
    assert usage["output_tokens"] == 6
    assert usage["provider_usd"] == pytest.approx(4.42e-06)
    assert usage["served_model"].endswith("-Turbo")


def test_extract_usage_tolerates_provider_without_cost():
    """Groq reports no cost. provider_usd is None, and the ledger falls back to
    the price table rather than silently billing zero."""

    class Resp:
        usage_metadata = {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}
        response_metadata = {"model_name": "llama-3.3-70b-versatile"}

    usage = extract_usage(Resp())
    assert usage["provider_usd"] is None
    assert usage["input_tokens"] == 10


def test_extract_usage_survives_empty_response():
    class Resp:
        pass

    usage = extract_usage(Resp())
    assert usage["input_tokens"] == 0
    assert usage["provider_usd"] is None


# --- Cache-key discrimination (WS-1) -----------------------------------------
# The 700 committed analyst entries and 178 keyword entries back Phase 3/4 and
# Phase 2/7. A key change that orphans them is a reproducibility failure, and it
# is invisible at runtime: an orphaned cache just looks like a cold one.

NOTE = "58-year-old woman with stage IV HER2-positive breast cancer, ECOG 1."
NCT = "NCT01234567"


def test_legacy_pair_reproduces_the_original_key_format(groq, monkeypatch):
    """Recomputed independently here rather than asserted against the
    implementation, so a change to the format fails this test instead of being
    silently mirrored by it."""
    import hashlib

    from trialguard.agent.analyst import _cache_key, prompt_version

    monkeypatch.setattr(settings, "groq_model", "llama-3.3-70b-versatile")
    expected = hashlib.sha256(
        f"{prompt_version()}|{NCT}|{NOTE}".encode()
    ).hexdigest()[:20]
    assert _cache_key(NOTE, NCT) == expected


def test_deepinfra_gets_a_separate_cache_namespace(monkeypatch):
    """FP8 and full-precision builds must never share an entry: one host's
    results would be reported as the other's."""
    from trialguard.agent.analyst import _cache_key

    monkeypatch.setattr(settings, "groq_model", "llama-3.3-70b-versatile")
    monkeypatch.setattr(settings, "llm_provider", "groq")
    legacy = _cache_key(NOTE, NCT)

    monkeypatch.setattr(settings, "llm_provider", "deepinfra")
    assert _cache_key(NOTE, NCT) != legacy


def test_keyword_cache_legacy_hash_is_the_bare_note(groq, monkeypatch):
    import hashlib

    from trialguard.retrieval.query_transform import _note_hash

    monkeypatch.setattr(settings, "groq_model", "llama-3.3-70b-versatile")
    assert _note_hash(NOTE) == hashlib.sha256(NOTE.encode()).hexdigest()[:16]


def test_keyword_cache_separates_providers(deepinfra):
    import hashlib

    from trialguard.retrieval.query_transform import _note_hash

    bare = hashlib.sha256(NOTE.encode()).hexdigest()[:16]
    assert _note_hash(NOTE) != bare


def test_a_different_groq_model_also_gets_its_own_namespace(groq, monkeypatch):
    """The carve-out keys on the exact (provider, model) pair, not on provider
    alone — swapping Groq's model must not silently reuse llama-3.3-70b entries."""
    from trialguard.agent.analyst import _cache_key

    monkeypatch.setattr(settings, "groq_model", "llama-3.3-70b-versatile")
    legacy = _cache_key(NOTE, NCT)
    monkeypatch.setattr(settings, "groq_model", "llama-3.1-8b-instant")
    assert _cache_key(NOTE, NCT) != legacy
