"""Langfuse v3 tracing. LangGraph agents use get_langchain_handler().

Lazy-initialized: Langfuse client created only after env vars are loaded.
No-op fallback when credentials are absent or tracing is disabled.
"""

from __future__ import annotations

import os
from typing import Any


def _credentials_present() -> bool:
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )


def get_client():
    """Return Langfuse v3 client, or None if credentials/tracing absent.

    Import Langfuse here (not at module top) so env vars are already set.
    """
    from trialguard.config import settings  # local import — env loaded by now

    if not settings.tracing_enabled or not _credentials_present():
        return None
    try:
        from langfuse import get_client as _get_client  # type: ignore

        return _get_client()
    except ImportError:
        return None


def get_langchain_handler(
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Return a Langfuse CallbackHandler for LangGraph invocations.

    Usage:
        handler = get_langchain_handler(session_id="run-42", tags=["eval"])
        graph.invoke(input, config={"callbacks": [handler]})

    Returns None when tracing is disabled; LangGraph ignores None callbacks.
    """
    if not _credentials_present():
        return None
    try:
        from langfuse.langchain import CallbackHandler  # type: ignore

        metadata: dict[str, Any] = {}
        if session_id:
            metadata["langfuse_session_id"] = session_id
        if user_id:
            metadata["langfuse_user_id"] = user_id
        if tags:
            metadata["langfuse_tags"] = tags

        return CallbackHandler(metadata=metadata if metadata else None)
    except ImportError:
        return None


def flush() -> None:
    """Flush all queued trace events. Call before process exit in scripts."""
    client = get_client()
    if client is not None:
        client.flush()
