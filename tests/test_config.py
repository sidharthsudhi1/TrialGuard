"""Smoke tests: config defaults sane, tracing no-ops without credentials."""

from trialguard.config import settings
from trialguard.tracing import flush, get_langchain_handler


def test_defaults():
    assert settings.ctgov_api_base == "https://clinicaltrials.gov/api/v2"
    assert settings.condition_class == "oncology"
    assert settings.ctgov_request_delay == 1.5


def test_tracing_noop_without_credentials():
    # No LANGFUSE keys in test env → handler should be None (no-op)
    handler = get_langchain_handler(session_id="test", tags=["smoke"])
    assert handler is None


def test_flush_noop_without_credentials():
    # Should not raise even without credentials
    flush()
