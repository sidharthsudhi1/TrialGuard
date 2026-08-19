"""Durable key-value cache backed by Postgres.

The disk caches under `data/cache/` are the primary store and stay authoritative:
they back committed eval numbers, and reading them first means no existing result
can change. This is the layer underneath, for the served path, where the
container filesystem is replaced on every deploy.

Degrades to a no-op without DATABASE_URL, so the eval CLI, CI, and the $0 Gradio
demo keep working exactly as before.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def _enabled() -> bool:
    from trialguard.config import settings

    return bool(settings.database_url)


def cache_get(namespace: str, key: str) -> Any | None:
    """Return the cached value, or None on miss, when disabled, or on error.

    A cache read that fails must not fail the request it serves — the caller's
    next step is to recompute, which is correct, only slower.
    """
    if not _enabled():
        return None
    try:
        from trialguard.db.schema import get_conn

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT value FROM cache_entries WHERE namespace = %s AND key = %s",
                (namespace, key),
            )
            row = cur.fetchone()
        return row[0] if row else None
    except Exception as e:  # noqa: BLE001 — degrade to a miss, but say so
        log.warning("cache_get(%s) failed: %s", namespace, type(e).__name__)
        return None


def cache_put(namespace: str, key: str, value: Any) -> bool:
    """Store a value. Returns whether it was written.

    Upsert rather than insert: two workers can compute the same key concurrently,
    and the later write is not an error.
    """
    if not _enabled():
        return False
    try:
        from trialguard.db.schema import get_conn

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cache_entries (namespace, key, value)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (namespace, key) DO UPDATE SET value = EXCLUDED.value
                """,
                (namespace, key, json.dumps(value)),
            )
        return True
    except Exception as e:  # noqa: BLE001 — a failed cache write is not a failed request
        log.warning("cache_put(%s) failed: %s", namespace, type(e).__name__)
        return False
