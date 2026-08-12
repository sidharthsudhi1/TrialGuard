"""Per-call cost accounting and the daily spend circuit breaker (Phase 8 WS-2).

The Groq free tier was its own cost control: run out of tokens and calls stop.
On a metered provider nothing stops a retry loop or a scripted client except this
module, so the token cap and the USD cap are enforced by the same ledger.

Three ideas the rest of the system depends on:

- **Gate on an estimate, account on the actual.** A call's cost is unknowable
  before it is made, so `check()` gates on a token estimate and `record()` bills
  the usage the provider actually reported. Systems that bill from estimates
  drift away from the invoice.
- **Prefer the provider's own number.** DeepInfra returns `usage.estimated_cost`
  per call, which cannot go stale the way a local price table does. `PRICES` is
  a fallback for providers that report nothing, and a cross-check that warns when
  the two disagree — a silent divergence means the table is stale and everything
  billed from it is wrong.
- **The budget is a circuit breaker, not a report.** It is checked on the write
  path and it fails closed, because its job is to make runaway spend structurally
  impossible rather than merely observable after the fact.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

# Imported as a module, not by value: `_today` is the single date seam the whole
# system rolls over on, and binding it here by value would fork it in two.
from trialguard.agent import ratelimit
from trialguard.agent.ratelimit import BudgetExhausted

log = logging.getLogger(__name__)

LEDGER_PATH = Path("data/cache/cost_ledger.json")

# USD per 1M tokens, (input, output). Fallback only: used when the provider
# reports no per-call cost, and as a cross-check when it does.
#
# DeepInfra Turbo derived from two live calls on 2026-07-29, solving the pair:
#   12 in / 2 out -> $1.84e-6      25 in / 6 out -> $4.42e-6   =>  $0.10 / $0.32
# A single data point underdetermines two variables — the first fit from one call
# was wrong, which is itself the argument for preferring the provider's number.
#
# Groq's free tier bills nothing, so it is priced at zero. That records $0
# honestly rather than inventing a paid-tier rate we are not being charged.
PRICES: dict[tuple[str, str], tuple[float, float]] = {
    ("deepinfra", "meta-llama/Llama-3.3-70B-Instruct-Turbo"): (0.10, 0.32),
    ("groq", "llama-3.3-70b-versatile"): (0.0, 0.0),
}

# Warn when the provider's reported cost and the local table disagree by more
# than this fraction. Divergence means the table has drifted from real pricing.
DIVERGENCE_TOLERANCE = 0.10


def table_usd(provider: str, model: str, in_tokens: int, out_tokens: int) -> float | None:
    """Cost from the local price table, or None if the pair is not priced."""
    rate = PRICES.get((provider, model))
    if rate is None:
        return None
    per_m_in, per_m_out = rate
    return (in_tokens * per_m_in + out_tokens * per_m_out) / 1_000_000


def call_usd(usage: dict, provider: str, model: str) -> tuple[float, str]:
    """Cost of one call as (usd, source), where source is "provider" or "table".

    Cross-checks the two when both exist: a divergence beyond
    DIVERGENCE_TOLERANCE is logged, because it means the table is stale and any
    historical figure billed from it is wrong.
    """
    reported = usage.get("provider_usd")
    fallback = table_usd(
        provider, model, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    )

    if reported is None:
        if fallback is None:
            log.warning(
                "No cost for (%s, %s): provider reported none and the pair is not "
                "in PRICES. Recording $0 — spend is being undercounted.",
                provider,
                model,
            )
            return 0.0, "unknown"
        return fallback, "table"

    if fallback is not None and fallback > 0:
        drift = abs(reported - fallback) / fallback
        if drift > DIVERGENCE_TOLERANCE:
            log.warning(
                "Price table drift for (%s, %s): provider $%.3g vs table $%.3g (%.0f%%). "
                "Update PRICES; figures billed from the table are unreliable.",
                provider,
                model,
                reported,
                fallback,
                drift * 100,
            )
    return float(reported), "provider"


class CostLedger:
    """Disk-persisted daily spend record. Resets on date rollover.

    Enforces two ceilings because they are two different external constraints:
    `usd_cap` is the money the account may spend, and `token_cap` is the Groq
    free tier's hard TPD limit, which no amount of budget makes spendable past.
    """

    def __init__(
        self,
        path: Path = LEDGER_PATH,
        usd_cap: float | None = None,
        token_cap: int | None = None,
    ):
        self.path = path
        if usd_cap is None:
            from trialguard.config import settings

            usd_cap = settings.daily_usd_cap
        self.usd_cap = usd_cap
        self.token_cap = token_cap

    def _load(self) -> dict:
        """Today's counter, carrying the accumulated per-day history forward.

        The daily fields reset on rollover because `usd_cap` means *per day* — a
        counter that never reset would trip permanently after a few days' use.
        But resetting used to destroy the previous day outright, so the ledger
        could answer "how much today" and never "how much has this project
        spent". `history` keeps the per-day record the cost story needs; the
        enforcement path still reads only the daily fields.
        """
        today = ratelimit._today()
        if self.path.exists():
            data = json.loads(self.path.read_text())
            history = data.get("history", {})
            if data.get("date") == today:
                data["history"] = history
                return data
            # Rollover: bank the closing day before zeroing the counter.
            if data.get("calls"):
                history[data["date"]] = {
                    "usd": data.get("usd", 0.0),
                    "tokens": data.get("tokens", 0),
                    "calls": data.get("calls", 0),
                }
        else:
            history = {}
        return {
            "date": today,
            "usd": 0.0,
            "tokens": 0,
            "calls": 0,
            "by_model": {},
            "history": history,
        }

    def spent_usd(self) -> float:
        return self._load()["usd"]

    def spent_tokens(self) -> int:
        return self._load()["tokens"]

    def remaining_usd(self) -> float:
        return max(0.0, self.usd_cap - self.spent_usd())

    def exhausted(self) -> bool:
        if self.remaining_usd() <= 0:
            return True
        return self.token_cap is not None and self.spent_tokens() >= self.token_cap

    def check(self, estimated_tokens: int = 0, provider: str = "", model: str = "") -> None:
        """Raise BudgetExhausted if the next call would cross either ceiling.

        Called before every fresh request. Callers (demo.run, the eval harness)
        catch this and degrade to cached-only rather than continuing to spend.
        """
        if self.token_cap is not None:
            spent = self.spent_tokens()
            if spent + estimated_tokens >= self.token_cap:
                raise BudgetExhausted(
                    f"Daily token cap reached: spent {spent} + est {estimated_tokens} "
                    f">= {self.token_cap}. Falling back to cached-only."
                )

        # Estimate the incoming call at the output rate: the pessimistic side,
        # since output tokens cost more and the split is unknown before the call.
        est_usd = 0.0
        if provider and model:
            est_usd = table_usd(provider, model, 0, estimated_tokens) or 0.0
        spent = self.spent_usd()
        if spent + est_usd >= self.usd_cap:
            # %g, not %.2f: a cap or spend below a cent must not render as $0.00,
            # which would make the message say a limit was crossed at zero.
            raise BudgetExhausted(
                f"Daily spend cap reached: ${spent:.4g} + est ${est_usd:.4g} "
                f">= ${self.usd_cap:.4g}. Falling back to cached-only."
            )

    def record(self, usage: dict, provider: str, model: str) -> float:
        """Bill one completed call from its real usage. Returns the USD charged."""
        usd, source = call_usd(usage, provider, model)
        data = self._load()
        data["usd"] = round(data["usd"] + usd, 8)
        data["tokens"] = data["tokens"] + int(usage.get("total_tokens") or 0)
        data["calls"] = data["calls"] + 1

        # Keyed by the model the provider says it SERVED, not the one requested.
        # DeepInfra aliases "-Instruct" to the FP8 "-Instruct-Turbo" build, so a
        # silent substitution shows up here as an unexpected key.
        served = usage.get("served_model") or model
        bucket = data["by_model"].setdefault(
            f"{provider}|{served}", {"usd": 0.0, "tokens": 0, "calls": 0, "source": source}
        )
        bucket["usd"] = round(bucket["usd"] + usd, 8)
        bucket["tokens"] += int(usage.get("total_tokens") or 0)
        bucket["calls"] += 1
        bucket["source"] = source

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True))
        return usd

    def summary(self) -> dict:
        d = self._load()
        history = d.get("history", {})
        # Today is live and not yet banked into history, so add it explicitly
        # rather than summing history alone.
        lifetime = {
            "usd": round(sum(v["usd"] for v in history.values()) + d["usd"], 6),
            "tokens": sum(v["tokens"] for v in history.values()) + d["tokens"],
            "calls": sum(v["calls"] for v in history.values()) + d["calls"],
            "days": len(history) + (1 if d["calls"] else 0),
        }
        return {
            "date": d["date"],
            "usd": round(d["usd"], 6),
            "usd_cap": self.usd_cap,
            "tokens": d["tokens"],
            "calls": d["calls"],
            "by_model": d["by_model"],
            "history": history,
            "lifetime": lifetime,
        }


def active_ledger() -> CostLedger:
    """Ledger configured for the active provider.

    The Groq free tier's TPD ceiling is a real external limit, so it is enforced
    only on the Groq arm; a metered provider has no such wall and is bounded by
    the USD cap alone.
    """
    from trialguard.llm.provider import active_provider

    token_cap = ratelimit.GROQ_TPD_CAP if active_provider() == "groq" else None
    return CostLedger(token_cap=token_cap)
