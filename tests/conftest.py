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

import os
import uuid
from urllib.parse import quote

import psycopg2
import pytest

TEST_DB = os.environ.get("TG_TEST_DATABASE_URL", "")


@pytest.fixture(autouse=True)
def _no_live_database(monkeypatch):
    monkeypatch.setattr("trialguard.config.settings.database_url", "", raising=False)


@pytest.fixture
def pg_db(monkeypatch):
    """A throwaway schema on TG_TEST_DATABASE_URL holding the full production DDL.

    Needs the pgvector extension on that server (CI runs pgvector/pgvector:pg16).
    Skips locally without it; in CI a missing extension fails instead.
    """
    from trialguard.db import schema

    if not TEST_DB:
        pytest.skip("TG_TEST_DATABASE_URL not set")
    name = f"t_{uuid.uuid4().hex[:10]}"
    admin = psycopg2.connect(TEST_DB)
    admin.autocommit = True
    with admin.cursor() as cur:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg2.Error:
            if os.environ.get("CI"):
                raise
            pytest.skip("pgvector not installed on the test server")
        cur.execute(f"CREATE SCHEMA {name}")

    sep = "&" if "?" in TEST_DB else "?"
    dsn = f"{TEST_DB}{sep}options={quote(f'-csearch_path={name},public')}"
    schema.close_pool()
    monkeypatch.setattr("trialguard.config.settings.database_url", dsn)
    schema.init_schema()
    yield dsn
    schema.close_pool()
    with admin.cursor() as cur:
        cur.execute(f"DROP SCHEMA {name} CASCADE")
    admin.close()
