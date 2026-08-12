from unittest.mock import MagicMock, patch

from trialguard import tracing
from trialguard.eval.agent_metrics import _observability


def test_emit_scores_noop_without_client():
    with patch.object(tracing, "get_client", return_value=None):
        assert tracing.emit_scores({"faithfulness": 0.99}, session_id="s") is False


def test_emit_scores_pushes_numeric_and_categorical():
    client = MagicMock()
    with patch.object(tracing, "get_client", return_value=client):
        ok = tracing.emit_scores(
            {"faithfulness": 0.99, "dominant_rejection": "ungrounded_quote"},
            session_id="agent-eval-sigir",
        )
    assert ok is True
    kinds = {c.kwargs["name"]: c.kwargs["data_type"] for c in client.create_score.call_args_list}
    assert kinds == {"faithfulness": "NUMERIC", "dominant_rejection": "CATEGORICAL"}
    for c in client.create_score.call_args_list:
        assert c.kwargs["session_id"] == "agent-eval-sigir"
    client.flush.assert_called_once()


# --- Trace config (Phase 8 WS-5) ---------------------------------------------
# Every traced invocation goes through trace_config(), so there is one definition
# of what a traced call carries instead of three ad-hoc `{"callbacks": [...]}`
# literals that drift apart.


def test_trace_config_is_empty_without_tracing():
    """Call sites pass the result straight to .invoke(config=...); an empty dict
    is what they already handled when the handler was None."""
    assert tracing.trace_config(None) == {}
    assert tracing.trace_config(None, provider="deepinfra") == {}


def test_trace_config_carries_handler_and_session_metadata():
    handler = MagicMock()
    handler.trialguard_metadata = {"langfuse_session_id": "agent-eval-sigir"}
    cfg = tracing.trace_config(handler)
    assert cfg["callbacks"] == [handler]
    assert cfg["metadata"]["langfuse_session_id"] == "agent-eval-sigir"


def test_trace_config_records_provider_and_prompt_version():
    """With two hosts and three prompt versions live, a trace that does not say
    which produced it cannot be attributed afterwards."""
    handler = MagicMock()
    handler.trialguard_metadata = {"langfuse_tags": ["agent-eval"]}
    cfg = tracing.trace_config(
        handler, provider="deepinfra", model="Llama-3.3-70B-Instruct-Turbo", prompt_version="v2"
    )
    md = cfg["metadata"]
    assert md["provider"] == "deepinfra"
    assert md["prompt_version"] == "v2"
    assert md["langfuse_tags"] == ["agent-eval"]  # session metadata not clobbered


def test_trace_config_drops_none_extras():
    handler = MagicMock()
    handler.trialguard_metadata = {}
    cfg = tracing.trace_config(handler, provider="groq", nct_id=None)
    assert "nct_id" not in cfg["metadata"]
    assert cfg["metadata"]["provider"] == "groq"


def test_trace_config_tolerates_handler_without_metadata():
    """A handler built by an SDK version that accepted constructor metadata still
    has the attribute stashed, but never assume it."""
    handler = object()
    cfg = tracing.trace_config(handler, provider="groq")
    assert cfg["callbacks"] == [handler]
    assert cfg["metadata"] == {"provider": "groq"}


def test_handler_stashes_metadata_on_both_sdk_majors():
    """v4 dropped constructor metadata. Both paths must leave it reachable, or
    trace_config silently emits traces with no session grouping."""
    import sys
    import types

    for accepts_kwarg in (True, False):
        class _CB:
            def __init__(self, **kw):
                if kw and not accepts_kwarg:
                    raise TypeError("unexpected keyword argument 'metadata'")

        mod = types.ModuleType("langfuse.langchain")
        mod.CallbackHandler = _CB
        with patch.dict(sys.modules, {"langfuse.langchain": mod}), \
             patch.object(tracing, "_credentials_present", return_value=True):
            h = tracing.get_langchain_handler(session_id="s1", tags=["t"])
        assert h.trialguard_metadata["langfuse_session_id"] == "s1"
        assert h.trialguard_metadata["langfuse_tags"] == ["t"]


def test_observability_reports_run_cost():
    """Quality without spend is half the tradeoff a tuning decision turns on."""
    verified = {
        "citation_precision": 0.99,
        "unsupported_verdict_rate": 0.01,
        "abstention_rate": 0.65,
        "coverage": 0.35,
        "mean_retries": 0.3,
    }
    assert "run_usd" not in _observability(verified)
    assert _observability(verified, run_usd=0.0389)["run_usd"] == 0.0389


def test_observability_maps_verified_arm():
    verified = {
        "citation_precision": 0.9916,
        "unsupported_verdict_rate": 0.0084,
        "abstention_rate": 0.7505,
        "coverage": 0.2495,
        "mean_retries": 0.35,
    }
    scores = _observability(verified)
    assert scores["faithfulness"] == 0.9916
    assert scores["mean_retries"] == 0.35
    assert set(scores) == {
        "faithfulness",
        "unsupported_verdict_rate",
        "abstention_rate",
        "coverage",
        "mean_retries",
    }
