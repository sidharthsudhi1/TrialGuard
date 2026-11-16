"""A full pool makes the next lease wait. It used to raise and the API answered 500."""

import threading
import time

import psycopg2
import pytest
from psycopg2 import pool

from trialguard.db import schema


class _Conn:
    closed = 0

    def commit(self):
        pass

    def rollback(self):
        pass


class _OnePool:
    def __init__(self):
        self.out = 0

    def getconn(self):
        if self.out:
            raise pool.PoolError("connection pool exhausted")
        self.out += 1
        return _Conn()

    def putconn(self, conn, close=False):
        self.out -= 1


@pytest.fixture
def one_slot(monkeypatch):
    fake = _OnePool()
    monkeypatch.setattr(schema, "_get_pool", lambda: fake)
    monkeypatch.setattr(schema, "_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(schema, "_last_used", {})
    monkeypatch.setattr(schema, "_alive", lambda conn: True)
    return fake


def test_second_lease_waits_for_the_first(one_slot):
    held = threading.Event()
    done: list[float] = []

    def second():
        held.wait()
        t0 = time.monotonic()
        with schema.get_conn():
            done.append(time.monotonic() - t0)

    worker = threading.Thread(target=second)
    worker.start()
    with schema.get_conn():
        held.set()
        time.sleep(0.2)
    worker.join(timeout=5)
    assert done and done[0] >= 0.1


def test_a_lease_that_never_frees_times_out(one_slot, monkeypatch):
    monkeypatch.setattr(schema, "_LEASE_WAIT_S", 0.05)
    with schema.get_conn():
        with pytest.raises(psycopg2.OperationalError, match="came free"):
            with schema.get_conn():
                pass
