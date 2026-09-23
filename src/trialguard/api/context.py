"""Values and helpers shared by the HTTP layer and the job runner.

Extracted so the import graph runs one way: context <- runner <- routes. These
lived in routes.py while it also held the job engine, which made every one of
them look like an HTTP concern. `SOURCE` and the tracing handler are neither --
they describe which corpus is served and how a request is traced, and both
halves need them.
"""

from __future__ import annotations

from typing import Any

SOURCE = "ctgov_live"
MAX_CRITERIA = 24


def _trace_handler(session_id: str, kind: str):
    """Langfuse handler for one served request, or None when tracing is off.

    The serving path built no handler at all, so `trace_config` took its
    `handler is None` no-op branch and every live assessment ran untraced while
    the eval harness traced everything. `session_id` is the job id for assess and
    a per-request id for search, so a user-reported result can be found later.
    """
    from trialguard.tracing import get_langchain_handler

    return get_langchain_handler(session_id=session_id, tags=["served", kind])


def _budget_exhausted_detail(exc: BaseException) -> dict[str, Any]:
    from trialguard.llm.cost import active_ledger

    ledger = active_ledger()
    summary = ledger.summary()
    return {
        "error": "BudgetExhausted",
        "message": str(exc),
        "usd_spent": summary["usd"],
        "usd_cap": summary["usd_cap"],
        "remaining_usd": ledger.remaining_usd(),
    }
