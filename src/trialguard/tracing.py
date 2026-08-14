"""Langfuse tracing (v3 and v4). LangGraph agents use get_langchain_handler().

Lazy-initialized: Langfuse client created only after env vars are loaded.
No-op fallback when credentials are absent or tracing is disabled.
"""

from __future__ import annotations

import os
from typing import Any


def _export_credentials() -> bool:
    """Mirror Langfuse settings into os.environ and report whether they exist.

    The Langfuse SDK reads its credentials from the environment, but this project
    loads config through pydantic-settings, which populates `settings` from .env
    without touching os.environ. Reading os.getenv directly therefore reported "no
    credentials" for every CLI run — tracing silently no-opped everywhere except
    where the vars happened to be exported by hand.
    """
    from trialguard.config import settings  # local import — env loaded by now

    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return False
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_base_url)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
    return True


def _credentials_present() -> bool:
    return _export_credentials()


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
        graph.invoke(input, config=trace_config(handler))

    Returns None when tracing is disabled; LangGraph ignores None callbacks.
    """
    from trialguard.config import settings

    if not settings.tracing_enabled or not _credentials_present():
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

        try:
            handler = CallbackHandler(metadata=metadata if metadata else None)
        except TypeError:
            # Langfuse v4 dropped constructor metadata: session/user/tags ride the
            # per-invocation config instead. This stayed invisible for a release
            # because a credential-less environment returns None above and never
            # reaches the constructor — so CI and the test suite never saw it.
            handler = CallbackHandler()
        # Stashed either way so trace_config() is the single place that knows how
        # metadata reaches Langfuse, regardless of which SDK major is installed.
        handler.trialguard_metadata = metadata
        return handler
    except ImportError:
        return None


def trace_config(handler: Any, **extra: Any) -> dict[str, Any]:
    """Build the LangChain invocation config for a traced call.

    Every LangGraph/LLM invocation in this project goes through here, so there is
    one definition of "what a traced call carries" rather than three ad-hoc
    `{"callbacks": [handler]}` literals that drift apart.

    `extra` becomes trace metadata — provider, model, prompt_version, nct_id. That
    matters more since Phase 8: with two hosts and three prompt versions in play, a
    trace that does not record which produced it cannot be attributed afterwards.

    Returns `{}` when tracing is off, which every call site already handles.
    """
    if handler is None:
        return {}
    metadata = dict(getattr(handler, "trialguard_metadata", {}) or {})
    metadata.update({k: v for k, v in extra.items() if v is not None})
    config: dict[str, Any] = {"callbacks": [handler]}
    if metadata:
        config["metadata"] = metadata
    return config


def emit_scores(
    scores: dict[str, float | str],
    session_id: str | None = None,
    trace_id: str | None = None,
) -> bool:
    """Push run/trace quality scores to Langfuse. No-op when tracing is disabled.

    These are the custom scores the Phase 5 dashboard trends over time
    (faithfulness, abstention, coverage, mean retries). Numeric values become
    NUMERIC scores; strings become CATEGORICAL (e.g. a dominant rejection reason).
    Linked to the eval session so cost/latency/retry, already aggregated by
    Langfuse from the traces, sit next to quality on one board.

    Returns True if the scores were emitted, False if tracing is off.
    """
    client = get_client()
    if client is None:
        return False
    for name, value in scores.items():
        client.create_score(
            name=name,
            value=value,
            session_id=session_id,
            trace_id=trace_id,
            data_type="CATEGORICAL" if isinstance(value, str) else "NUMERIC",
        )
    client.flush()
    return True


def flush() -> None:
    """Flush all queued trace events. Call before process exit in scripts."""
    client = get_client()
    if client is not None:
        client.flush()
