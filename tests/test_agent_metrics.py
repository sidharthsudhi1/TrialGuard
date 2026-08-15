"""Harness helpers. No LLM calls."""

from trialguard.eval.agent_metrics import _is_transient


def test_timeout_is_transient():
    class APITimeoutError(Exception):
        pass

    assert _is_transient(APITimeoutError("Request timed out."))
    assert _is_transient(TimeoutError("timed out"))


def test_budget_and_429_are_not_transient():
    assert not _is_transient(RuntimeError("429 rate_limit"))
    assert not _is_transient(RuntimeError("daily USD cap"))
