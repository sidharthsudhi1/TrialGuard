"""Smoke test: settings load without error and defaults are sane."""

from trialguard.config import settings


def test_defaults():
    assert settings.ctgov_api_base == "https://clinicaltrials.gov/api/v2"
    assert settings.condition_class == "oncology"
    assert settings.ctgov_request_delay == 1.5
