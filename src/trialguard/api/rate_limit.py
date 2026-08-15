"""Per-IP sliding-window rate limiter for the Stage A API."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Allow at most `limit` events per `window_seconds` per key (usually client IP)."""

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = max(1, int(limit))
        self.window_seconds = float(window_seconds)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            cutoff = now - self.window_seconds
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True
