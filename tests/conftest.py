"""Test isolation from live infrastructure.

`.env` carries a real DATABASE_URL, and pydantic-settings loads it whenever
`trialguard.config` is imported — including under pytest. Any code path that
checks `settings.database_url` to decide whether to use Postgres will therefore
take the live branch during a unit test. That is how three fixture rows
("keyword 0", "keyword 1", ...) reached the production `cache_entries` table.

Blanking it for every test makes the no-database path the default, which is also
the path CI runs. A test that wants database behaviour opts in explicitly with
`monkeypatch.setattr("trialguard.config.settings.database_url", ...)`, and should
stub the connection rather than open a real one.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_live_database(monkeypatch):
    monkeypatch.setattr("trialguard.config.settings.database_url", "", raising=False)
