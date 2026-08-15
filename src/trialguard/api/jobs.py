"""In-process assess job store with TTL eviction.

Interface is create/get/append/complete so Stage B can swap Postgres later.
No Redis.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    job_id: str
    note: str
    nct_ids: list[str]
    status: str = "queued"  # queued | running | done | error
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    skip_cache_write: bool = True


class JobStore:
    """Thread-safe dict of jobs; expired entries dropped on access."""

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _evict_unlocked(self) -> None:
        now = time.time()
        dead = [jid for jid, j in self._jobs.items() if now - j.created_at > self.ttl_seconds]
        for jid in dead:
            del self._jobs[jid]

    def create(
        self, note: str, nct_ids: list[str], *, skip_cache_write: bool = True
    ) -> Job:
        with self._lock:
            self._evict_unlocked()
            job = Job(
                job_id=uuid.uuid4().hex,
                note=note,
                nct_ids=list(nct_ids),
                skip_cache_write=skip_cache_write,
            )
            self._jobs[job.job_id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._evict_unlocked()
            return self._jobs.get(job_id)

    def append(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.events.append(event)
            if job.status == "queued":
                job.status = "running"

    def complete(self, job_id: str, summary: dict[str, Any] | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if summary is not None:
                job.events.append({"type": "summary", **summary})
            job.status = "done"

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.error = error
            job.events.append({"type": "error", "error": error})
            job.status = "error"
