"""Smoke tests: config defaults sane, tracing no-ops without credentials."""

import trialguard.config as config_module
from trialguard.config import Settings, settings
from trialguard.tracing import flush, get_langchain_handler


def test_defaults():
    assert settings.ctgov_api_base == "https://clinicaltrials.gov/api/v2"
    assert settings.condition_class == "oncology"
    assert settings.ctgov_request_delay == 1.5


def test_tracing_noop_without_credentials(monkeypatch):
    # Assert on explicitly credential-less settings rather than on an empty
    # ambient environment: the repo's own .env would otherwise mask the case,
    # which is how tracing stayed silently disabled for every CLI run.
    blank = Settings(_env_file=None, langfuse_public_key="", langfuse_secret_key="")
    monkeypatch.setattr(config_module, "settings", blank)
    assert get_langchain_handler(session_id="test", tags=["smoke"]) is None


def test_tracing_disabled_flag_wins(monkeypatch):
    off = Settings(
        _env_file=None,
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        tracing_enabled=False,
    )
    monkeypatch.setattr(config_module, "settings", off)
    assert get_langchain_handler(session_id="test") is None


def test_flush_noop_without_credentials(monkeypatch):
    blank = Settings(_env_file=None, langfuse_public_key="", langfuse_secret_key="")
    monkeypatch.setattr(config_module, "settings", blank)
    flush()
