"""WS-1: jobs survive the process that created them, and orphans are detected.

The Postgres tests need a scratch database in TG_TEST_DATABASE_URL and skip
without one. They apply JOBS_DDL rather than the full schema, so no pgvector is
required and CI can run them against a stock postgres service.
"""

from __future__ import annotations

import os
import threading

import pytest

from trialguard.api.jobs import (
    InMemoryJobStore,
    PostgresJobStore,
    make_job_store,
)

TEST_DB = os.environ.get("TG_TEST_DATABASE_URL", "")

# Skipping locally is a convenience; skipping in CI would make a green suite mean
# nothing for WS-1, since durability is the entire claim these tests check. CI
# provides a scratch postgres service, so a missing URL there is a broken
# workflow, not an absent database.
needs_db = pytest.mark.skipif(
    not TEST_DB and not os.environ.get("CI"),
    reason="TG_TEST_DATABASE_URL not set",
)


# --------------------------------------------------------------------------
# Contract, no database
# --------------------------------------------------------------------------


def test_both_stores_expose_the_same_interface():
    """Endpoints must not learn where jobs live, so the surfaces cannot drift."""
    surface = {"create", "get", "events_since", "heartbeat", "append", "complete", "fail"}
    for name in surface:
        assert callable(getattr(InMemoryJobStore, name)), name
        assert callable(getattr(PostgresJobStore, name)), name


def test_events_since_is_one_based_and_exclusive():
    store = InMemoryJobStore()
    job = store.create("note", ["NCT1"])
    for i in range(3):
        store.append(job.job_id, {"type": "trial", "i": i})

    assert [s for s, _ in store.events_since(job.job_id, 0)] == [1, 2, 3]
    assert [s for s, _ in store.events_since(job.job_id, 2)] == [3]
    assert store.events_since(job.job_id, 3) == []
    assert store.events_since(job.job_id, 0)[0][1]["i"] == 0


def test_heartbeat_reports_whether_the_job_is_still_ours():
    """The beat is the cancellation channel: False means stop spending."""
    store = InMemoryJobStore()
    job = store.create("note", ["NCT1"])
    assert store.heartbeat(job.job_id) is True

    store.fail(job.job_id, "taken over elsewhere")
    assert store.heartbeat(job.job_id) is False
    assert store.heartbeat("no-such-job") is False


def test_make_job_store_falls_back_without_a_database():
    assert isinstance(make_job_store(3600, 90), InMemoryJobStore)


def test_make_job_store_uses_postgres_when_configured(monkeypatch):
    monkeypatch.setattr("trialguard.config.settings.database_url", "postgres://x/y")
    assert isinstance(make_job_store(3600, 90), PostgresJobStore)


# --------------------------------------------------------------------------
# Durability and orphan detection, against a real Postgres
# --------------------------------------------------------------------------


@pytest.fixture
def pg(monkeypatch):
    from trialguard.db.schema import JOBS_DDL, close_pool, get_conn

    close_pool()
    monkeypatch.setattr("trialguard.config.settings.database_url", TEST_DB)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(JOBS_DDL)
        cur.execute("TRUNCATE jobs CASCADE")
    yield
    close_pool()


@needs_db
def test_events_written_before_a_kill_are_still_readable(pg):
    """Acceptance: kill the machine mid-assess; the log is intact afterwards."""
    store = PostgresJobStore()
    job = store.create("synthetic note", ["NCT1", "NCT2"])
    store.append(job.job_id, {"type": "criterion", "nct_id": "NCT1"})
    store.append(job.job_id, {"type": "trial", "nct_id": "NCT1"})

    # A new process: same rows, new interpreter, new instance id.
    monkey = "deadbeef" * 4
    import trialguard.api.jobs as jobs_mod

    original, jobs_mod.INSTANCE_ID = jobs_mod.INSTANCE_ID, monkey
    try:
        reread = PostgresJobStore().get(job.job_id)
    finally:
        jobs_mod.INSTANCE_ID = original

    assert reread is not None
    assert reread.note == "synthetic note"
    assert reread.nct_ids == ["NCT1", "NCT2"]
    types = [e["type"] for e in reread.events]
    assert types[:2] == ["criterion", "trial"]


@needs_db
def test_a_job_whose_process_is_gone_reads_as_failed(pg, monkeypatch):
    """Acceptance: a dead worker's job is failed, not left `running`."""
    from trialguard.db.schema import get_conn

    store = PostgresJobStore(stale_seconds=30)
    job = store.create("note", ["NCT1"])
    store.append(job.job_id, {"type": "criterion"})
    assert store.get(job.job_id).status == "running"

    # The owner stopped beating and a new interpreter is reading the row.
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET heartbeat_at = NOW() - interval '5 minutes' "
            "WHERE job_id = %s",
            (job.job_id,),
        )
    monkeypatch.setattr("trialguard.api.jobs.INSTANCE_ID", "a-different-process")
    reread = PostgresJobStore(stale_seconds=30).get(job.job_id)

    assert reread.status == "error"
    assert "worker_died" in reread.error
    assert reread.events[-1]["type"] == "error"


@needs_db
def test_a_suspended_process_is_not_mistaken_for_a_dead_one(pg):
    """AD-13: suspend freezes the heartbeat of a process that is still alive.

    Same instance, heartbeat far older than the window -- which is exactly what a
    resumed machine looks like to the blocking read that resumed it. The instance
    clause is what keeps this job running, and it is why the heartbeat alone
    cannot be the test.
    """
    from trialguard.db.schema import get_conn

    store = PostgresJobStore(stale_seconds=10)
    job = store.create("note", ["NCT1"])
    store.append(job.job_id, {"type": "criterion"})
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET heartbeat_at = NOW() - interval '10 minutes' "
            "WHERE job_id = %s",
            (job.job_id,),
        )

    assert store.get(job.job_id).status == "running"


@needs_db
def test_a_second_live_machine_does_not_fail_another_machines_job(pg, monkeypatch):
    """The other half of the conjunction: a fresh heartbeat means alive.

    `auto_start_machines` can put a second machine behind the same hostname, and
    a poll load-balanced to it must not kill a job the first machine is running.
    """
    store = PostgresJobStore(stale_seconds=30)
    job = store.create("note", ["NCT1"])
    store.append(job.job_id, {"type": "criterion"})

    monkeypatch.setattr("trialguard.api.jobs.INSTANCE_ID", "the-other-machine")
    reread = PostgresJobStore(stale_seconds=30).get(job.job_id)

    assert reread.status == "running"


@needs_db
def test_a_finished_job_is_never_reopened_as_an_orphan(pg, monkeypatch):
    store = PostgresJobStore()
    job = store.create("note", ["NCT1"])
    store.append(job.job_id, {"type": "trial"})
    store.complete(job.job_id, {"n_trials": 1, "status": "done"})

    monkeypatch.setattr("trialguard.api.jobs.INSTANCE_ID", "a-different-process")
    reread = PostgresJobStore().get(job.job_id)

    assert reread.status == "done"
    assert reread.events[-1]["type"] == "summary"


@needs_db
def test_seq_is_gapless_under_concurrent_appends(pg):
    """WS-2's cursor is only safe if seq has no holes and no reordering."""
    store = PostgresJobStore()
    job = store.create("note", ["NCT1"])

    def worker(n: int) -> None:
        for i in range(5):
            store.append(job.job_id, {"type": "criterion", "w": n, "i": i})

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    seqs = [s for s, _ in store.events_since(job.job_id, 0)]
    assert seqs == list(range(1, 31))


@needs_db
def test_retention_deletes_expired_jobs_and_their_events(pg):
    """Retention is a delete, not eviction-on-read (AD-13)."""
    from trialguard.db.schema import get_conn

    store = PostgresJobStore(ttl_seconds=3600)
    old = store.create("note", ["NCT1"])
    store.append(old.job_id, {"type": "trial"})
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET created_at = NOW() - interval '2 hours' WHERE job_id = %s",
            (old.job_id,),
        )

    store.create("fresh note", ["NCT2"])

    assert store.get(old.job_id) is None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM job_events WHERE job_id = %s", (old.job_id,))
        assert cur.fetchone()[0] == 0


@needs_db
def test_a_disowned_job_stops_beating(pg, monkeypatch):
    """A woken machine must not resume spending on a job already failed.

    Suspended-but-unreachable and gone are the same thing to everyone waiting, so
    another reader is right to fail the job. What this guarantees is that the
    original owner finds out on its next beat instead of paying for the rest.
    """
    from trialguard.db.schema import get_conn

    store = PostgresJobStore(stale_seconds=30)
    job = store.create("note", ["NCT1", "NCT2"])
    store.append(job.job_id, {"type": "criterion"})
    assert store.heartbeat(job.job_id) is True

    # Another reader declares it dead while this owner is unreachable.
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET heartbeat_at = NOW() - interval '5 minutes' "
            "WHERE job_id = %s",
            (job.job_id,),
        )
    monkeypatch.setattr("trialguard.api.jobs.INSTANCE_ID", "a-different-process")
    assert PostgresJobStore(stale_seconds=30).get(job.job_id).status == "error"
    monkeypatch.undo()

    assert store.heartbeat(job.job_id) is False


@needs_db
def test_a_disowned_beat_does_not_revive_the_heartbeat(pg):
    """The guard is on the UPDATE, so a terminal row is never touched again."""
    from trialguard.db.schema import get_conn

    store = PostgresJobStore()
    job = store.create("note", ["NCT1"])
    store.fail(job.job_id, "failed elsewhere")

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT heartbeat_at FROM jobs WHERE job_id = %s", (job.job_id,))
        before = cur.fetchone()[0]

    assert store.heartbeat(job.job_id) is False

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT heartbeat_at FROM jobs WHERE job_id = %s", (job.job_id,))
        assert cur.fetchone()[0] == before
