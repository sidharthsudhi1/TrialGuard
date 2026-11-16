"""Eval spend that must not touch the production Postgres ledger."""

from __future__ import annotations

from pathlib import Path

from trialguard.llm.cost import CostLedger


def isolated_ledger(path: Path, cap: float) -> CostLedger:
    """A file ledger capped at `cap` dollars. Blanks the database URL for this process."""
    from trialguard.config import settings
    from trialguard.llm import cost

    settings.database_url = ""
    settings.daily_usd_cap = cap
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = CostLedger(path=path, usd_cap=cap)
    cost.active_ledger = lambda: ledger
    return ledger
