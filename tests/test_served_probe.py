"""WS-4: the probe alerts on the failures traces cannot show."""

from __future__ import annotations

import datetime as dt

from trialguard.eval.served_probe import check, probe


class _Resp:
    def __init__(self, status: int, body: dict | None = None):
        self.status_code = status
        self._body = body if body is not None else {}

    def json(self):
        return self._body


class _FakeClient:
    """Answers the four probe calls; raise_on lets a test kill one endpoint."""

    def __init__(self, health=None, budget=None, search=None, raise_on=()):
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self.health = health if health is not None else {
            "ok": True,
            "pool_ok": True,
            "corpus_refresh": {"at": now, "corpus": 25965},
        }
        self.budget = budget if budget is not None else {
            "exhausted": False,
            "remaining_usd": 1.5,
        }
        self.search = search if search is not None else {
            "trials": [{"nct_id": f"NCT{i}"} for i in range(10)],
            # Measured shape from the deployed API: the server reports its own
            # breakdown, and keyword_ms is the LLM call when the cache is cold.
            "latency_ms": {"keyword_ms": 0.3, "total_ms": 617.7},
        }
        self.raise_on = set(raise_on)
        self.calls: list[str] = []

    def get(self, url):
        if "health" in url:
            self.calls.append("health")
            if "health" in self.raise_on:
                raise ConnectionError("refused")
            return _Resp(200, self.health)
        self.calls.append("budget")
        if "budget" in self.raise_on:
            raise ConnectionError("refused")
        return _Resp(200, self.budget)

    def post(self, url, json=None):
        self.calls.append("search")
        if "search" in self.raise_on:
            raise ConnectionError("refused")
        return _Resp(200, self.search)

    def close(self):
        pass


def _probe(**kw) -> dict:
    return probe("https://example.test", client=_FakeClient(**kw))


def test_a_healthy_deployment_passes_every_check():
    assert check(_probe())["passed"] is True


def test_the_probe_calls_each_endpoint_twice_to_separate_cold_from_warm():
    """fly.toml suspends idle machines; the first call of each absorbs the resume.

    Search is measured twice for a distinct reason: the first search after a
    resume was measured at 20,576 ms server-side against 542-570 ms settled, so
    one sample would be whichever of those the probe happened to catch.
    """
    client = _FakeClient()
    probe("https://example.test", client=client)

    assert client.calls == ["health", "health", "budget", "search", "search"]


def test_the_first_search_is_reported_but_never_gates():
    """An order-of-magnitude slower first call would fail every deploy."""
    result = {**_probe(), "search_server_cold_ms": 20_576.2}

    outcome = check(result)
    row = next(r for r in outcome["results"] if r["check"] == "search_first_call")
    assert row["passed"] is True
    assert row["value"] == 20_576.2
    assert outcome["passed"] is True


def test_an_unreachable_api_fails_rather_than_raising():
    result = _probe(raise_on=("health",))

    assert result["health_status"] == "ConnectionError"
    assert "ConnectionError" in result["errors"]
    outcome = check(result)
    assert outcome["passed"] is False
    assert _failed(outcome) >= {"health_reachable", "health_ok"}


def test_a_broken_database_pool_alerts():
    """/api/health returns 200 with pool_ok false; a status check alone misses it."""
    result = _probe(health={"ok": True, "pool_ok": False})

    outcome = check(result)
    assert outcome["passed"] is False
    assert "pool_ok" in _failed(outcome)


def test_an_exhausted_budget_alerts():
    result = _probe(budget={"exhausted": True, "remaining_usd": 0.0})

    outcome = check(result)
    assert outcome["passed"] is False
    assert "budget_not_exhausted" in _failed(outcome)


def test_a_search_returning_nothing_alerts():
    """A corpus that vanished still answers 200 with an empty list."""
    result = _probe(search={"trials": [], "latency_ms": {"total_ms": 12.0}})

    outcome = check(result)
    assert outcome["passed"] is False
    assert _failed(outcome) == {"search_returns_trials"}


def test_a_cold_keyword_cache_does_not_alert_on_latency():
    """One LLM call inside the request is the documented cost of a cache miss.

    Measured against the deployed API: 26,870 ms wall on a miss against 725.6 ms
    warm. Gating wall time alone would alert every time the cache is cold and
    explain nothing, so the stable server number is what is gated.
    """
    result = {
        **_probe(search={"trials": [{"nct_id": "NCT1"}],
                         "latency_ms": {"keyword_ms": 26000.0, "total_ms": 640.0}}),
        "search_ms": 26870.1,
    }

    outcome = check(result)
    assert outcome["passed"] is True
    assert result["search_keyword_ms"] == 26000.0


def test_a_response_without_server_timing_alerts():
    """The gate reads latency_ms; a shape change must fail loudly, not pass."""
    result = _probe(search={"trials": [{"nct_id": "NCT1"}]})

    outcome = check(result)
    assert outcome["passed"] is False
    assert "search_latency" in _failed(outcome)


def test_a_stalled_corpus_refresh_alerts():
    """WS-3's schedule fails silently; this is what notices."""
    stale = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=5)).isoformat()
    result = _probe(
        health={"ok": True, "pool_ok": True, "corpus_refresh": {"at": stale}}
    )

    assert result["corpus_age_hours"] > 48
    outcome = check(result)
    assert outcome["passed"] is False
    assert "corpus_freshness" in _failed(outcome)


def test_a_never_refreshed_corpus_is_reported_not_gated():
    """A fresh deploy has no refresh stamp yet, which is not a defect."""
    result = _probe(health={"ok": True, "pool_ok": True})

    assert result["corpus_age_hours"] is None
    assert check(result)["passed"] is True


def test_slow_latency_alerts():
    """A deployment that answers correctly but slowly still alerts.

    The thresholds are applied to a result rather than a stubbed clock: an
    in-process fake returns in microseconds, so driving this through the client
    would compare real timings against a zero bound and pass on rounding.
    """
    result = {
        **_probe(),
        "health_cold_ms": 120_000.0,
        "health_warm_ms": 9_000.0,
        "search_server_ms": 30_000.0,
        "search_ms": 90_000.0,
    }

    outcome = check(result)

    assert outcome["passed"] is False
    assert _failed(outcome) == {
        "health_warm_latency",
        "cold_start",
        "search_latency",
        "search_not_hanging",
    }


def test_committed_thresholds_cover_every_key_check_reads():
    """A threshold renamed in the JSON must not KeyError at 21:00 in CI."""
    from trialguard.eval.served_probe import THRESHOLDS

    assert check(_probe(), thresholds_path=THRESHOLDS)["passed"] is True


def _failed(outcome: dict) -> set[str]:
    return {r["check"] for r in outcome["results"] if not r["passed"]}
