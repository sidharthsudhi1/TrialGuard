"""Assess job store: in-process by default, Postgres when one is configured.

Interface is create/get/append/complete/fail plus heartbeat and events_since.
Endpoints do not learn where jobs live, the same way the keyword cache layered
disk -> Postgres -> model behind one call.

Without DATABASE_URL this degrades to the original in-process dict, which is the
path CI and the $0 Gradio demo take. Durability is the point of the Postgres
implementation, so unlike db/cache.py it does not swallow write errors: a lost
event is lost paid work, and a job whose events stopped reaching the database is
one the orphan check below is supposed to catch.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# The process that owns a job row, deliberately not FLY_MACHINE_ID: a machine
# keeps its Fly id across a restart, so that id cannot tell a crash-restart from a
# suspend/resume. fly.toml suspends idle machines and suspend snapshots RAM, so a
# resumed process is alive with a heartbeat frozen for the whole suspension. The
# process is what dies, so the process is what is identified -- suspend preserves
# this value, a deploy, crash or OOM starts a new interpreter with a new one.
# See AD-13.
INSTANCE_ID = uuid.uuid4().hex

_STALE_ERROR = "worker_died: no heartbeat from the process that owned this job"


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


class InMemoryJobStore:
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

    def get(self, job_id: str, *, with_events: bool = True) -> Job | None:
        with self._lock:
            self._evict_unlocked()
            return self._jobs.get(job_id)

    def events_since(self, job_id: str, after_seq: int) -> list[tuple[int, dict[str, Any]]]:
        """Events with seq > after_seq, 1-based, in append order."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return []
            return [(i, job.events[i - 1]) for i in range(after_seq + 1, len(job.events) + 1)]

    def heartbeat(self, job_id: str) -> None:
        """No-op: a dict cannot outlive the process that owns it."""

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


class PostgresJobStore:
    """Durable jobs. Survives the restart that took the dict with it."""

    def __init__(self, ttl_seconds: int = 3600, stale_seconds: int = 90):
        self.ttl_seconds = ttl_seconds
        self.stale_seconds = stale_seconds

    @staticmethod
    def _conn():
        from trialguard.db.schema import get_conn

        return get_conn()

    def create(
        self, note: str, nct_ids: list[str], *, skip_cache_write: bool = True
    ) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex,
            note=note,
            nct_ids=list(nct_ids),
            skip_cache_write=skip_cache_write,
        )
        with self._conn() as conn, conn.cursor() as cur:
            # Retention is a delete, not eviction-on-read. The note is stored
            # because the worker needs it, under the same one-hour window the
            # in-process store applied -- a shared durable store is a stronger
            # reason to honour the free-text no-persist policy that
            # skip_cache_write encodes, not a weaker one.
            cur.execute(
                "DELETE FROM jobs WHERE created_at < NOW() - make_interval(secs => %s)",
                (self.ttl_seconds,),
            )
            cur.execute(
                """
                INSERT INTO jobs (job_id, note, nct_ids, status, skip_cache_write,
                                  instance_id)
                VALUES (%s, %s, %s, 'queued', %s, %s)
                """,
                (job.job_id, note, list(nct_ids), skip_cache_write, INSTANCE_ID),
            )
        return job

    def _mark_stale_if_orphaned(self, cur, job_id: str) -> None:
        """Fail a job whose owning process is gone. Both conditions, conjoined.

        A different instance means the reader is not the process that owns the
        row; a stale heartbeat means that owner has stopped proving it is alive.
        Neither is sufficient alone:

        - instance alone would fail a healthy job the moment a second machine
          load-balanced a poll away from its owner, which `auto_start_machines`
          permits under load.
        - heartbeat age alone would fail a *suspended* job. fly.toml snapshots RAM
          rather than stopping, so the owner is alive with a frozen heartbeat, and
          the request that resumes the machine is a blocking read that can run
          before the resumed heartbeat task gets the loop back.

        The case this cannot see is a task that dies inside a process that keeps
        running -- same instance, so the conjunction never fires. That one is not
        left to a timer: `_run_assess_job` gives every job a terminal status in a
        `finally`, which is deterministic where a heartbeat is a guess.
        """
        cur.execute(
            """
            UPDATE jobs
               SET status = 'error', error = %s
             WHERE job_id = %s
               AND status IN ('queued', 'running')
               AND instance_id <> %s
               AND heartbeat_at < NOW() - make_interval(secs => %s)
            """,
            (_STALE_ERROR, job_id, INSTANCE_ID, self.stale_seconds),
        )
        if cur.rowcount:
            self._append_unlocked(cur, job_id, {"type": "error", "error": _STALE_ERROR})

    def get(self, job_id: str, *, with_events: bool = True) -> Job | None:
        with self._conn() as conn, conn.cursor() as cur:
            self._mark_stale_if_orphaned(cur, job_id)
            cur.execute(
                """
                SELECT note, nct_ids, status, error, skip_cache_write,
                       EXTRACT(EPOCH FROM created_at)
                  FROM jobs WHERE job_id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            job = Job(
                job_id=job_id,
                note=row[0],
                nct_ids=list(row[1]),
                status=row[2],
                error=row[3],
                skip_cache_write=row[4],
                created_at=float(row[5]),
            )
            if with_events:
                cur.execute(
                    "SELECT event FROM job_events WHERE job_id = %s ORDER BY seq",
                    (job_id,),
                )
                job.events = [r[0] for r in cur.fetchall()]
            return job

    def events_since(self, job_id: str, after_seq: int) -> list[tuple[int, dict[str, Any]]]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT seq, event FROM job_events "
                "WHERE job_id = %s AND seq > %s ORDER BY seq",
                (job_id, after_seq),
            )
            return [(r[0], r[1]) for r in cur.fetchall()]

    def heartbeat(self, job_id: str) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET heartbeat_at = NOW() WHERE job_id = %s", (job_id,)
            )

    @staticmethod
    def _append_unlocked(cur, job_id: str, event: dict[str, Any]) -> None:
        """Insert one event. Caller holds the FOR UPDATE lock on the parent row.

        seq comes from MAX(seq)+1 under that lock rather than from a sequence: a
        BIGSERIAL is allocated before commit, so a reader can see a later id
        committed while an earlier one is still in flight and skip it forever,
        which is the gap WS-2's cursor must not have.
        """
        cur.execute(
            """
            INSERT INTO job_events (job_id, seq, event)
            VALUES (%s,
                    (SELECT COALESCE(MAX(seq), 0) + 1 FROM job_events WHERE job_id = %s),
                    %s::jsonb)
            """,
            (job_id, job_id, json.dumps(event)),
        )

    def _locked_append(self, job_id: str, event: dict[str, Any], status_sql: str) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM jobs WHERE job_id = %s FOR UPDATE", (job_id,))
            if cur.fetchone() is None:
                return
            self._append_unlocked(cur, job_id, event)
            cur.execute(status_sql, (job_id,))

    def append(self, job_id: str, event: dict[str, Any]) -> None:
        self._locked_append(
            job_id,
            event,
            "UPDATE jobs SET heartbeat_at = NOW(), "
            "status = CASE WHEN status = 'queued' THEN 'running' ELSE status END "
            "WHERE job_id = %s",
        )

    def complete(self, job_id: str, summary: dict[str, Any] | None = None) -> None:
        event = {"type": "summary", **(summary or {})}
        self._locked_append(
            job_id, event, "UPDATE jobs SET status = 'done' WHERE job_id = %s"
        )

    def fail(self, job_id: str, error: str) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM jobs WHERE job_id = %s FOR UPDATE", (job_id,))
            if cur.fetchone() is None:
                return
            self._append_unlocked(cur, job_id, {"type": "error", "error": error})
            cur.execute(
                "UPDATE jobs SET status = 'error', error = %s WHERE job_id = %s",
                (error, job_id),
            )


# Kept so existing imports and type references do not move with the migration.
JobStore = InMemoryJobStore


def make_job_store(ttl_seconds: int, stale_seconds: int):
    """Postgres where one is configured, the in-process dict otherwise."""
    from trialguard.config import settings

    if settings.database_url:
        return PostgresJobStore(ttl_seconds=ttl_seconds, stale_seconds=stale_seconds)
    return InMemoryJobStore(ttl_seconds=ttl_seconds)
