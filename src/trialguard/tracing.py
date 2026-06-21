"""Langfuse tracing skeleton. All agent steps go through get_tracer()."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from trialguard.config import settings


def _langfuse_client():
    """Return Langfuse client if credentials exist, else None."""
    if not settings.tracing_enabled:
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse import Langfuse  # type: ignore

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except ImportError:
        return None


_client = _langfuse_client()


@contextmanager
def trace_span(
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Context manager wrapping a single agent step in a Langfuse span.

    Falls back to a no-op if tracing is disabled or credentials are absent.
    """
    if _client is None:
        yield None
        return

    span = _client.span(name=name, metadata=metadata or {})
    try:
        yield span
    except Exception as exc:
        span.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        span.end()


def flush() -> None:
    """Flush pending traces. Call before process exit."""
    if _client is not None:
        _client.flush()
