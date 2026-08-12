"""Backoff policy and budget primitives (Phase 5 WS-3; reworked Phase 8 WS-2).

One place for every rate-limit knob that used to be a magic number scattered
across the analyst: the client-side retry/spacing constants, the request-pacing
delay, and the exception the harness catches when a budget is spent.

Spend accounting itself lives in llm/cost.py. The split is deliberate: this
module holds constraints imposed by the *transport* (429s, TPM windows, the free
tier's TPD wall), while cost.py holds constraints imposed by the *account*
(money). BudgetExhausted stays here because demo.py and eval/agent_metrics.py
import it from this path, and it is the contract between the analyst and every
caller that degrades to cached-only.
"""

from __future__ import annotations

import datetime
import os

# Groq free-tier caps for llama-3.3-70b-versatile. Overridable for other tiers.
# These describe an external wall, not a policy choice: past TPD the API refuses,
# so no amount of budget makes those tokens spendable.
TPM_CAP = int(os.environ.get("TG_GROQ_TPM", "12000"))
GROQ_TPD_CAP = int(os.environ.get("TG_GROQ_TPD", "100000"))
MAX_RETRIES = int(os.environ.get("TG_GROQ_MAX_RETRIES", "8"))

# Seconds to sleep after a fresh analyst call, to stay inside the TPM window.
# Only the free tier needs it: a metered provider has no per-minute wall to pace
# against, and 7s per call across a full eval run is hours of pure waiting.
GROQ_ANALYST_DELAY = float(os.environ.get("TG_ANALYST_DELAY", "7"))


class BudgetExhausted(RuntimeError):
    """A daily budget (tokens or spend) is used up. Callers degrade to cached-only."""


def _today() -> str:
    return datetime.date.today().isoformat()


def estimate_tokens(text: str) -> int:
    """Rough token estimate for pre-call budget gating (~4 chars/token).

    Deliberately crude: it exists only to gate a call whose real cost cannot be
    known yet. Billing uses the provider's reported usage, never this.
    """
    return len(text) // 4


def analyst_delay() -> float:
    """Inter-call pacing for the active provider.

    An explicit TG_ANALYST_DELAY always wins, so a caller can still pace a
    metered provider if it starts returning 429s.
    """
    if "TG_ANALYST_DELAY" in os.environ:
        return float(os.environ["TG_ANALYST_DELAY"])

    from trialguard.llm.provider import active_provider

    return GROQ_ANALYST_DELAY if active_provider() == "groq" else 0.0
