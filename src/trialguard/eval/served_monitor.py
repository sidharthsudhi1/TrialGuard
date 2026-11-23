"""E2 — served-metric monitor: read `served`-tagged traces, compare to baselines.

The eval numbers describe the cohorts; nothing had ever measured whether the
*served* system behaves like them. This reads assess traces from Langfuse,
recomputes abstention / grounding-failure / retry rates from the graph's final
state, and diverges loudly when a rate leaves its committed band
(data/reports/served_baselines.json). Bands are deliberately wide: this is a
drift detector for the metrics nobody watches, not a quality gate — CI already
owns quality (regression_gate.py).

CLI: `python -m trialguard.eval.served_monitor --days 7` — prints the table and
exits non-zero on divergence, so a scheduled runner alerts for free.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

BASELINES = Path("data/reports/served_baselines.json")

_ABSTAIN = ("cannot_determine", "unverifiable")


def _trace_output(trace) -> dict | None:
    """Final graph state from a trace, tolerating string-serialized output."""
    out = getattr(trace, "output", None)
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except (ValueError, TypeError):
            return None
    return out if isinstance(out, dict) and "assessments" in out else None


def served_filter(since: dt.datetime) -> str:
    """Root observations of served assess traces started at or after `since`."""
    return json.dumps(
        [
            {
                "type": "arrayOptions",
                "column": "tags",
                "operator": "all of",
                "value": ["served", "assess"],
            },
            {"type": "boolean", "column": "isRootObservation", "operator": "=", "value": True},
            {
                "type": "datetime",
                "column": "startTime",
                "operator": ">=",
                "value": since.isoformat(),
            },
        ]
    )


def fetch_served_states(days: int) -> list[dict]:
    """Final states of every served assess trace in the window."""
    from trialguard.tracing import get_client

    client = get_client()
    if client is None:
        raise RuntimeError("Langfuse credentials absent — cannot read served traces")

    # Langfuse 5 removed trace listing. A trace's final state is its root
    # observation's output, and `filter` overrides every query-parameter filter,
    # so the tags, root flag and window all have to live in it.
    since = dt.datetime.now(dt.UTC) - dt.timedelta(days=days)
    query = served_filter(since)
    states: list[dict] = []
    cursor = None
    while True:
        page = client.api.observations.get_many(
            fields="core,io", filter=query, limit=50, cursor=cursor
        )
        for obs in page.data:
            state = _trace_output(obs)
            if state is not None:
                states.append(state)
        cursor = page.meta.cursor
        if not cursor:
            break
    return states


def compute(states: list[dict]) -> dict:
    """The three E2 rates plus context, from graph final states."""
    n_criteria = abstain = unverifiable = 0
    retried = 0
    verdicts: dict[str, int] = {}
    for s in states:
        if s.get("retries", 0) > 0:
            retried += 1
        verdicts[s.get("trial_verdict", "cannot_determine")] = (
            verdicts.get(s.get("trial_verdict", "cannot_determine"), 0) + 1
        )
        for a in s.get("assessments", []):
            n_criteria += 1
            v = a.get("verdict")
            if v in _ABSTAIN:
                abstain += 1
            if v == "unverifiable":
                unverifiable += 1

    def _rate(a: int, b: int) -> float:
        return round(a / b, 4) if b else 0.0

    return {
        "n_trials": len(states),
        "n_criteria": n_criteria,
        "abstention_rate": _rate(abstain, n_criteria),
        "unverifiable_rate": _rate(unverifiable, n_criteria),
        "retry_rate": _rate(retried, len(states)),
        "trial_verdicts": verdicts,
    }


def check(metrics: dict, baselines_path: Path = BASELINES) -> dict:
    baselines = json.loads(baselines_path.read_text())
    results = []
    for band in baselines["bands"]:
        value = metrics[band["metric"]]
        lo, hi = band.get("min"), band.get("max")
        ok = (lo is None or value >= lo) and (hi is None or value <= hi)
        results.append({**band, "value": value, "passed": ok})
    enough = metrics["n_trials"] >= baselines["min_trials"]
    return {
        "checked": enough,
        "passed": all(r["passed"] for r in results) if enough else True,
        "results": results,
        "min_trials": baselines["min_trials"],
    }


def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Served-trace metric monitor (E2)")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", default=None, help="Optional JSON report path")
    args = ap.parse_args()

    states = fetch_served_states(args.days)
    metrics = compute(states)
    outcome = check(metrics)

    print(f"Served monitor — last {args.days}d: "
          f"{metrics['n_trials']} trials, {metrics['n_criteria']} criteria")
    for r in outcome["results"]:
        band = f"[{r.get('min', '-')}, {r.get('max', '-')}]"
        mark = "OK" if r["passed"] else "DIVERGED"
        print(f"{mark:>8} {r['metric']:<20} {r['value']:.4f}  band {band}")
    print(f"{'':>8} retry_rate           {metrics['retry_rate']:.4f}  (reported, not gated)")

    if args.out:
        Path(args.out).write_text(json.dumps({"metrics": metrics, "outcome": outcome}, indent=2))
    if not outcome["checked"]:
        print(f"only {metrics['n_trials']} trials < min_trials={outcome['min_trials']}; "
              f"not enough served traffic to judge divergence")
    elif not outcome["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
