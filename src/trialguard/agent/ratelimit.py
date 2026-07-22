"""Groq free-tier budget accounting + backoff policy (Phase 5 WS-3).

One place for every rate-limit knob that used to be a magic number scattered
across the analyst. Two jobs:

- Backoff policy: the client-side retry/spacing constants the analyst reads, so
  a 429 is honored (Retry-After) and fresh calls are spaced under the TPM window.
- Daily token budget: the free tier is TPD-capped (~100k). Once today's recorded
  spend hits the cap, a fresh call raises BudgetExhausted instead of hammering the
  API into 429s — the harness catches it and degrades to cached-only cleanly. The
  count persists to disk so runs that span days (the normal case here) resume
  against the same daily budget. See the groq-freetier-limits note.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

# Free-tier caps for llama-3.3-70b-versatile. Overridable for other tiers/tests.
TPM_CAP = int(os.environ.get("TG_GROQ_TPM", "12000"))
TPD_CAP = int(os.environ.get("TG_GROQ_TPD", "100000"))
MAX_RETRIES = int(os.environ.get("TG_GROQ_MAX_RETRIES", "8"))
ANALYST_DELAY = float(os.environ.get("TG_ANALYST_DELAY", "7"))

BUDGET_PATH = Path("data/cache/groq_budget.json")


class BudgetExhausted(RuntimeError):
    """Today's recorded token spend has reached the daily cap."""


def _today() -> str:
    return datetime.date.today().isoformat()


def estimate_tokens(text: str) -> int:
    """Rough token estimate for pre-call budget gating (~4 chars/token)."""
    return len(text) // 4


class TokenBudget:
    """Disk-persisted daily token counter. Resets on date rollover."""

    def __init__(self, path: Path = BUDGET_PATH, cap: int = TPD_CAP):
        self.path = path
        self.cap = cap

    def _load(self) -> dict:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("date") == _today():
                return data
        return {"date": _today(), "spent": 0}

    def spent(self) -> int:
        return self._load()["spent"]

    def remaining(self) -> int:
        return max(0, self.cap - self.spent())

    def exhausted(self) -> bool:
        return self.remaining() <= 0

    def check(self, estimate: int = 0) -> None:
        """Raise BudgetExhausted if spending `estimate` more would cross the cap."""
        if self.spent() + estimate >= self.cap:
            raise BudgetExhausted(
                f"Groq daily token cap reached: spent {self.spent()} + est {estimate} "
                f">= {self.cap}. Falling back to cached-only."
            )

    def record(self, tokens: int) -> None:
        data = self._load()
        data["spent"] = data["spent"] + max(0, tokens)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data))
