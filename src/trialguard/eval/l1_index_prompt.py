"""WS-6c / L1 — does an index-addressed prompt actually cost less, and what breaks?

Two halves, because the claim has two halves and they need different sample sizes.

**Efficiency (here).** 43.4% of analyst output is the model retyping criteria it
was just handed. v5 numbers the criteria and asks for the number back. Whether
that shrinks output is a mechanical property with low variance, so a small paired
sample settles it -- but it has to be *paired* and *fresh*: v4 is fully cached
from earlier phases, and a cache hit has no latency and no token count to compare
against. This module therefore calls the model directly with `build_messages`,
bypassing the cache in both arms, and bills every call to the ledger.

It also reports what the indexing itself costs: how many returned objects carry a
usable index, and how many were dropped as out-of-range, duplicated or missing.
That is v5's own failure mode and nothing downstream can see it -- a dropped
criterion just looks like a criterion the analyst declined to answer.

**Quality (not here).** Verdict distribution, grounded counts and the tier
contract need the full assessed set and are produced by `end_to_end` under each
prompt version, where the v4 arm is free from cache. L4 is the precedent: it
improved the faithfulness proxy 26% while making the system strictly worse, and
only the criterion-level split revealed that the vanished grounding failures had
become abstentions. A latency win here means nothing until that table agrees.

  python -m trialguard.eval.l1_index_prompt --cohort sigir --n-trials 20
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

REPORT_DIR = Path("data/reports")
ARMS = ("v4", "v5")


def _index_health(raw: str, typed: list[dict]) -> dict:
    """How the returned objects addressed their criteria, before validation.

    Counted off the raw response rather than the resolved list, because the whole
    point is to see what `resolve_indices` had to throw away.
    """
    from trialguard.agent.analyst import _salvage

    objs = _salvage(raw)
    out = dict.fromkeys(("returned", "by_index", "by_text", "out_of_range", "duplicate",
                         "unusable"), 0)
    seen: set[int] = set()
    for obj in objs:
        out["returned"] += 1
        if obj.get("criterion"):
            out["by_text"] += 1
            continue
        idx = obj.get("index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            try:
                idx = int(str(idx).strip())
            except (TypeError, ValueError):
                out["unusable"] += 1
                continue
        if not 1 <= idx <= len(typed):
            out["out_of_range"] += 1
        elif idx in seen:
            out["duplicate"] += 1
        else:
            seen.add(idx)
            out["by_index"] += 1
    return out


def _one_call(note: str, nct_id: str, typed: list[dict], version: str) -> dict:
    """One uncached analyst call. Returns tokens, wall time and index health."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from trialguard.agent.analyst import build_messages
    from trialguard.llm.cost import active_ledger
    from trialguard.llm.provider import (
        active_model,
        active_provider,
        extract_usage,
        get_chat_model,
    )

    system, user = build_messages(note, nct_id, typed, version)
    t0 = time.perf_counter()
    resp = get_chat_model("analyst").invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    elapsed = time.perf_counter() - t0
    usage = extract_usage(resp)
    # Billed like any other call: this run spends real money and the ledger is
    # what the phase's cost figure is read from.
    active_ledger().record(usage, active_provider(), active_model())
    raw = str(resp.content)
    out_tok = int(usage.get("output_tokens") or 0)
    return {
        "nct_id": nct_id,
        "version": version,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": out_tok,
        "seconds": round(elapsed, 2),
        "tokens_per_second": round(out_tok / elapsed, 1) if elapsed else 0.0,
        "n_criteria": len(typed),
        "index_health": _index_health(raw, typed),
    }


def _summarise(calls: list[dict]) -> dict:
    def _med(key: str) -> float:
        vals = [c[key] for c in calls]
        return round(statistics.median(vals), 2) if vals else 0.0

    health = dict.fromkeys(("returned", "by_index", "by_text", "out_of_range",
                            "duplicate", "unusable"), 0)
    for c in calls:
        for k, v in c["index_health"].items():
            health[k] += v
    return {
        "n_calls": len(calls),
        "median_output_tokens": _med("output_tokens"),
        "median_input_tokens": _med("input_tokens"),
        "median_seconds": _med("seconds"),
        "median_tokens_per_second": _med("tokens_per_second"),
        "total_output_tokens": sum(c["output_tokens"] for c in calls),
        "criteria_asked": sum(c["n_criteria"] for c in calls),
        "index_health": health,
    }


def run(cohort: str, n_trials: int) -> dict:
    from trialguard.eval.agent_metrics import _build_subset
    from trialguard.llm.cost import active_ledger
    from trialguard.llm.provider import active_model, active_provider

    spend_before = active_ledger().spent_usd()
    subset = _build_subset(cohort, n_patients=50, per_class=2)
    work = [(p["note"], tr) for p in subset for tr in p["trials"]][:n_trials]

    calls: dict[str, list[dict]] = {a: [] for a in ARMS}
    for i, (note, tr) in enumerate(work):
        # Alternate which arm goes first. Provider latency drifts over a run, and
        # a fixed order would hand the whole drift to one arm.
        order = ARMS if i % 2 == 0 else tuple(reversed(ARMS))
        for version in order:
            calls[version].append(_one_call(note, tr["nct_id"], tr["criteria"], version))

    summary = {a: _summarise(calls[a]) for a in ARMS}
    base, new = summary["v4"], summary["v5"]

    def _delta(key: str) -> float:
        return round(new[key] - base[key], 2)

    def _ratio(key: str) -> float | None:
        return round(new[key] / base[key], 4) if base[key] else None

    return {
        "cohort": cohort,
        "provider": active_provider(),
        "model": active_model(),
        "arms": summary,
        "delta": {
            "output_tokens_median": _delta("median_output_tokens"),
            "output_tokens_ratio": _ratio("median_output_tokens"),
            "seconds_median": _delta("median_seconds"),
            "seconds_ratio": _ratio("median_seconds"),
            "total_output_tokens_ratio": _ratio("total_output_tokens"),
        },
        # Kept: a median over 20 paired calls hides a bimodal arm, and the raw
        # rows are what a later reader would otherwise have to re-buy.
        "calls": calls,
        "run_usd": round(max(0.0, active_ledger().spent_usd() - spend_before), 6),
    }


def main() -> None:
    import argparse

    from rich.console import Console
    from rich.table import Table

    ap = argparse.ArgumentParser(description="L1: v4 vs v5 output cost (WS-6c)")
    ap.add_argument("--cohort", default="sigir", choices=["sigir", "trec_2021", "trec_2022"])
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    console = Console()
    console.print(
        f"[bold]L1 paired prompt cost[/bold] {args.cohort} · {args.n_trials} trials × 2 arms "
        f"· uncached, billed"
    )
    result = run(args.cohort, args.n_trials)

    t = Table("metric", "v4", "v5", "ratio")
    a, b, d = result["arms"]["v4"], result["arms"]["v5"], result["delta"]
    t.add_row("median output tokens", str(a["median_output_tokens"]),
              str(b["median_output_tokens"]), str(d["output_tokens_ratio"]))
    t.add_row("median seconds", str(a["median_seconds"]),
              str(b["median_seconds"]), str(d["seconds_ratio"]))
    t.add_row("median input tokens", str(a["median_input_tokens"]),
              str(b["median_input_tokens"]), "")
    t.add_row("median out tok/s", str(a["median_tokens_per_second"]),
              str(b["median_tokens_per_second"]), "")
    console.print(t)

    h = result["arms"]["v5"]["index_health"]
    console.print(
        f"v5 index health · returned {h['returned']} of {result['arms']['v5']['criteria_asked']} "
        f"asked · by index {h['by_index']} · echoed text {h['by_text']} · "
        f"out of range {h['out_of_range']} · duplicate {h['duplicate']} · "
        f"unusable {h['unusable']}"
    )
    console.print(f"run cost ${result['run_usd']:.6f}")

    out = Path(args.out) if args.out else REPORT_DIR / f"l1_prompt_cost_{args.cohort}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    console.print(f"\nReport: [cyan]{out}[/cyan]")


if __name__ == "__main__":
    main()
