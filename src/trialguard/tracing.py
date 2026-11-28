"""Langfuse tracing (v3 and v4). LangGraph agents use get_langchain_handler().

Lazy-initialized: Langfuse client created only after env vars are loaded.
No-op fallback when credentials are absent or tracing is disabled.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Phase 40: under the de-identified input contract, served traces carry
# structure only. These keys hold enums, ids and model names; every other
# string in an exported span is replaced, so no note text reaches Langfuse.
REDACTED = "[redacted]"
_ENUM_KEYS = frozenset({
    "verdict", "trial_verdict", "trial_tier", "kind", "role", "type", "nct_id",
    "provider", "model", "model_name", "prompt_version", "purpose", "finish_reason",
})
_ENUM_VALUE = re.compile(r"[\w.:/-]{1,80}")
# Span attributes that hold serialized payloads: scrubbed, never passed through.
_PAYLOAD = re.compile(r"\.(input|output|status_message|model\.parameters)$|\.metadata(\.|$)")
_MAX_FREE_TOKENS = 3


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


def scrub(value: Any, key: str | None = None) -> Any:
    """Structure of a traced payload: numbers, flags and enums kept, text replaced."""
    if isinstance(value, dict):
        return {k: scrub(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v, key) for v in value]
    if isinstance(value, str):
        return value if key in _ENUM_KEYS and _ENUM_VALUE.fullmatch(value) else REDACTED
    return value


def _scrub_attribute(key: str, value: Any) -> Any:
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        decoded = value
    leaf = key.rsplit(".", 1)[-1]
    cleaned = scrub(decoded, leaf)
    return json.dumps(cleaned) if isinstance(value, str) else cleaned


def _free_text(value: Any) -> bool:
    values = value if isinstance(value, (list, tuple)) else [value]
    return any(isinstance(v, str) and len(v.split()) > _MAX_FREE_TOKENS for v in values)


def mask_served_spans(*, params: Any) -> Any:
    """Export-stage Langfuse hook: payloads scrubbed, any other prose dropped.

    Runs on the final span attributes, so it covers the LangChain handler and
    direct observations alike. If it raises, Langfuse drops the batch: the
    failure mode is a missing trace, not a leaked note.
    """
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for ident, span in params.spans.items():
        set_attrs, drop = {}, []
        for key, value in span.attributes.items():
            if _PAYLOAD.search(key):
                set_attrs[key] = _scrub_attribute(key, value)
            elif key.startswith(("gen_ai.prompt", "gen_ai.completion")) or _free_text(value):
                drop.append(key)
        if set_attrs or drop:
            patches[ident] = OtelSpanPatch(set_attributes=set_attrs, delete_attributes=tuple(drop))
    return MaskOtelSpansResult(span_patches=patches)


def _client():
    """The process Langfuse client, built with the served mask when it is on."""
    from langfuse import Langfuse
    from langfuse import get_client as _get_client

    from trialguard.config import settings

    if settings.trace_structure_only:
        global _MASKED_CLIENT
        if _MASKED_CLIENT is None:
            client = Langfuse(mask_otel_spans=mask_served_spans)
            # Langfuse keeps one client per public key and ignores the mask when
            # one already exists. Tracing off beats exporting note text.
            if client._resources.mask_otel_spans is not mask_served_spans:
                return None
            _MASKED_CLIENT = client
        return _MASKED_CLIENT
    return _get_client()


_MASKED_CLIENT = None


def get_client():
    """Return Langfuse v3 client, or None if credentials/tracing absent.

    Import Langfuse here (not at module top) so env vars are already set.
    """
    from trialguard.config import settings  # local import — env loaded by now

    if not settings.tracing_enabled or not _credentials_present():
        return None
    try:
        return _client()
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
        from langfuse.langchain import CallbackHandler

        # The handler binds to the process client, so build it masked first.
        if _client() is None:
            return None
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
