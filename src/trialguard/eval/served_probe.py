"""WS-4 — probe the deployed API for the failures traces cannot show.

`served_monitor.py` reads Langfuse traces, so it only sees requests that reached
the model. It cannot see an API returning 500s, a search that got slow, a daily
budget that is exhausted, or a corpus refresh that stopped firing: those produce
*fewer* traces, not different ones. A trace-reading monitor is blind precisely
when the system is most broken, and reports "no divergence" while doing it.

This probes the live endpoints instead and fails loudly. Thresholds live in
`data/reports/served_probe_thresholds.json` beside the monitor's bands.

Cold start is measured rather than warmed away. `fly.toml` suspends idle machines,
so the first request of the day resumes one; that latency is what a user meets
after a quiet period and nothing had ever recorded it.

CLI: `python -m trialguard.eval.served_probe --base-url https://...`
Exits non-zero on any failed check, so a scheduled runner alerts for free.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

THRESHOLDS = Path("data/reports/served_probe_thresholds.json")

# A fixed synthetic note so the keyword cache hits after the first probe: the
# cache is keyed by note text and lives in Postgres, so a stable string costs one
# LLM call ever and nothing thereafter. Varying it would bill a keyword
# extraction on every run to measure the same thing.
PROBE_NOTE = (
    "62-year-old man with metastatic colorectal cancer, ECOG performance status 1, "
    "prior FOLFOX, adequate renal and hepatic function."
)


def _timed(fn) -> tuple[Any, float, str | None]:
    """Run fn, returning (response, elapsed_ms, error). Never raises."""
    start = time.perf_counter()
    try:
        return fn(), round((time.perf_counter() - start) * 1000, 1), None
    except Exception as e:  # noqa: BLE001 — an unreachable API is a result, not a crash
        return None, round((time.perf_counter() - start) * 1000, 1), type(e).__name__


def _corpus_age_hours(health: dict) -> float | None:
    """Hours since the corpus was last reconciled, or None if it never was."""
    refresh = health.get("corpus_refresh")
    if not isinstance(refresh, dict):
        return None
    stamp = refresh.get("at")
    if not stamp:
        return None
    try:
        at = dt.datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=dt.timezone.utc)
    delta = dt.datetime.now(dt.timezone.utc) - at
    return round(delta.total_seconds() / 3600, 2)


def probe(base_url: str, *, client=None, timeout: float = 60.0) -> dict:
    """Hit the live endpoints once each and report what happened."""
    import httpx

    base = base_url.rstrip("/")
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        # First health call absorbs a Fly resume if the machine was suspended, so
        # it is reported separately rather than folded into the warm number.
        r_cold, cold_ms, cold_err = _timed(lambda: client.get(f"{base}/api/health"))
        r_warm, warm_ms, warm_err = _timed(lambda: client.get(f"{base}/api/health"))
        r_budget, budget_ms, budget_err = _timed(lambda: client.get(f"{base}/api/budget"))
        r_search, search_ms, search_err = _timed(
            lambda: client.post(
                f"{base}/api/search", json={"note": PROBE_NOTE, "top_k": 10}
            )
        )
    finally:
        if owns_client:
            client.close()

    health = _json_or_empty(r_warm) or _json_or_empty(r_cold)
    budget = _json_or_empty(r_budget)
    search = _json_or_empty(r_search)

    return {
        "base_url": base,
        "health_cold_ms": cold_ms,
        "health_warm_ms": warm_ms,
        "health_status": _status(r_cold, cold_err),
        "health_ok": bool(health.get("ok")),
        "pool_ok": bool(health.get("pool_ok")),
        "corpus_age_hours": _corpus_age_hours(health),
        "budget_ms": budget_ms,
        "budget_status": _status(r_budget, budget_err),
        "budget_exhausted": bool(budget.get("exhausted")),
        "budget_remaining_usd": budget.get("remaining_usd"),
        # Client wall time and the server's own accounting are both kept. A note
        # that misses the keyword cache costs one LLM call inside the request --
        # measured at ~27 s against 726 ms warm -- so gating on wall time alone
        # would alert every time the cache is cold and explain nothing. The
        # server's total_ms is the stable number; keyword_ms names the cause when
        # the wall time diverges from it.
        "search_ms": search_ms,
        "search_server_ms": _latency(search, "total_ms"),
        "search_keyword_ms": _latency(search, "keyword_ms"),
        "search_status": _status(r_search, search_err),
        "search_results": len(search.get("trials", []) or []),
        "errors": [e for e in (cold_err, warm_err, budget_err, search_err) if e],
    }


def _latency(search: dict, key: str) -> float | None:
    """One field of the server's own latency_ms breakdown, or None if absent."""
    latency = search.get("latency_ms")
    if not isinstance(latency, dict):
        return None
    value = latency.get(key)
    return round(float(value), 1) if isinstance(value, (int, float)) else None


def _json_or_empty(response) -> dict:
    if response is None:
        return {}
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        return {}
    return body if isinstance(body, dict) else {}


def _status(response, error: str | None) -> int | str:
    if error:
        return error
    return getattr(response, "status_code", "no_response")


def check(result: dict, thresholds_path: Path = THRESHOLDS) -> dict:
    """Apply the committed thresholds. Returns per-check rows and an overall pass."""
    t = json.loads(thresholds_path.read_text())
    rows: list[dict] = []

    def _row(name: str, passed: bool, value: Any, expected: str) -> None:
        rows.append(
            {"check": name, "passed": bool(passed), "value": value, "expected": expected}
        )

    _row("health_reachable", result["health_status"] == 200, result["health_status"], "200")
    _row("health_ok", result["health_ok"], result["health_ok"], "true")
    _row("pool_ok", result["pool_ok"], result["pool_ok"], "true")
    _row(
        "health_warm_latency",
        result["health_warm_ms"] <= t["max_health_warm_ms"],
        result["health_warm_ms"],
        f"<= {t['max_health_warm_ms']} ms",
    )
    _row(
        "cold_start",
        result["health_cold_ms"] <= t["max_health_cold_ms"],
        result["health_cold_ms"],
        f"<= {t['max_health_cold_ms']} ms",
    )
    _row("search_reachable", result["search_status"] == 200, result["search_status"], "200")
    # Gated on the server's own number. A cold keyword cache adds an LLM call
    # inside the request that has nothing to do with whether retrieval regressed.
    server_ms = result.get("search_server_ms")
    if server_ms is None:
        _row("search_latency", False, "no latency_ms in response", "server timing")
    else:
        _row(
            "search_latency",
            server_ms <= t["max_search_server_ms"],
            server_ms,
            f"<= {t['max_search_server_ms']} ms",
        )
    # Wall time is gated far more loosely: it exists to catch a request that is
    # hanging outright, not to police the variance the server number already
    # covers.
    _row(
        "search_not_hanging",
        result["search_ms"] <= t["max_search_wall_ms"],
        result["search_ms"],
        f"<= {t['max_search_wall_ms']} ms",
    )
    _row(
        "search_returns_trials",
        result["search_results"] >= t["min_search_results"],
        result["search_results"],
        f">= {t['min_search_results']}",
    )
    _row(
        "budget_not_exhausted",
        not result["budget_exhausted"],
        result["budget_exhausted"],
        "false",
    )

    # A corpus that was never reconciled is not a failure on a fresh deploy, but a
    # stamp that stops advancing means the scheduled machine is not firing -- the
    # exact silent failure WS-3's schedule introduces.
    age = result["corpus_age_hours"]
    if age is None:
        _row("corpus_freshness", True, "never refreshed", "reported, not gated")
    else:
        _row(
            "corpus_freshness",
            age <= t["max_corpus_age_hours"],
            age,
            f"<= {t['max_corpus_age_hours']} h",
        )

    return {"passed": all(r["passed"] for r in rows), "results": rows}


def main() -> None:
    import argparse
    import os
    import sys

    ap = argparse.ArgumentParser(description="Live API probe (WS-4)")
    ap.add_argument(
        "--base-url",
        default=os.environ.get("TG_API_BASE_URL", "https://trialguard-api.fly.dev"),
    )
    ap.add_argument("--out", default=None, help="Optional JSON report path")
    args = ap.parse_args()

    result = probe(args.base_url)
    outcome = check(result)

    print(f"Served probe — {result['base_url']}")
    for r in outcome["results"]:
        mark = "OK" if r["passed"] else "FAIL"
        print(f"{mark:>6} {r['check']:<22} {str(r['value']):<18} expected {r['expected']}")
    print(
        f"{'':>6} {'search_keyword_ms':<22} {str(result.get('search_keyword_ms')):<18} "
        "(reported: large means the keyword cache was cold)"
    )
    if result["errors"]:
        print(f"{'':>6} transport errors: {', '.join(result['errors'])}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"probe": result, "outcome": outcome}, indent=2)
        )
    if not outcome["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
