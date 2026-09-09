"""Per-IP sliding-window rate limiter for the Stage A API."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Allow at most `limit` events per `window_seconds` per key (usually client IP).

    Crude and bypassable by anyone with a pool of addresses — it stops accidents
    and casual abuse, not a determined attacker. Auth is the real fix.
    """

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = max(1, int(limit))
        self.window_seconds = float(window_seconds)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def _sweep(self, now: float) -> None:
        """Drop keys with nothing left inside the window.

        Without this the key dict only ever grows, and its keys come from request
        headers — so the memory an attacker can consume is bounded by how many
        distinct values they care to send. Sweeping once per window keeps the map
        proportional to genuinely active clients.
        """
        if now - self._last_sweep < self.window_seconds:
            return
        self._last_sweep = now
        cutoff = now - self.window_seconds
        for key in [k for k, q in self._hits.items() if not q or q[-1] < cutoff]:
            del self._hits[key]

    def take(self, key: str) -> float | None:
        """Record a hit and return None, or return seconds until a slot frees.

        One method rather than allow() plus a separate retry_after(), because the
        two answers have to come from the same look at the window: between two
        calls the oldest hit can age out and the caller would report a wait it had
        already stopped needing.
        """
        now = time.monotonic()
        with self._lock:
            self._sweep(now)
            q = self._hits[key]
            cutoff = now - self.window_seconds
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.limit:
                # The oldest hit leaving the window is what frees the next slot.
                return max(0.0, q[0] + self.window_seconds - now)
            q.append(now)
            return None

    def allow(self, key: str) -> bool:
        return self.take(key) is None
