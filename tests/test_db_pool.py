"""get_conn against a fake pool: a dead pooled connection is never handed out,
and a dead connection's rollback never replaces the error worth reporting."""

from __future__ import annotations

import time

import psycopg2
import pytest

from trialguard.db import schema


class _Cursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if self.conn.dead:
            raise psycopg2.OperationalError("server closed the connection unexpectedly")
        self.conn.pings += sql == "SELECT 1"


class _Conn:
    def __init__(self, name, dead=False, closed=0):
        self.name, self.dead, self.closed, self.pings = name, dead, closed, 0

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        if self.dead:
            self.closed = 2
            raise psycopg2.OperationalError("SSL connection has been closed unexpectedly")

    def rollback(self):
        if self.dead or self.closed:
            raise psycopg2.InterfaceError("connection already closed")


class _Pool:
    def __init__(self, conns):
        self.free, self.closed, self.returned = list(conns), [], []

    def getconn(self):
        return self.free.pop(0)

    def putconn(self, conn, close=False):
        (self.closed if close else self.returned).append(conn.name)


@pytest.fixture
def fake_pool(monkeypatch):
    def install(*conns):
        p = _Pool(conns)
        monkeypatch.setattr(schema, "_get_pool", lambda: p)
        monkeypatch.setattr(schema, "_last_used", {})
        return p

    return install


def test_a_closed_pooled_connection_is_replaced(fake_pool):
    p = fake_pool(_Conn("stale", closed=1), _Conn("fresh"))
    with schema.get_conn() as conn:
        assert conn.name == "fresh"
    assert p.closed == ["stale"] and p.returned == ["fresh"]


def test_an_idle_connection_the_server_dropped_is_replaced(fake_pool):
    p = fake_pool(_Conn("dropped", dead=True), _Conn("fresh"))
    with schema.get_conn() as conn:
        assert conn.name == "fresh"
    assert p.closed == ["dropped"]


def test_a_recently_used_connection_is_not_pinged(fake_pool):
    busy = _Conn("busy")
    fake_pool(busy)
    schema._last_used[id(busy)] = time.monotonic()
    with schema.get_conn() as conn:
        assert conn is busy
    assert busy.pings == 0


def test_the_real_error_survives_a_dead_connection(fake_pool):
    conn = _Conn("c")
    p = fake_pool(conn)
    with pytest.raises(RuntimeError, match="the real failure"):
        with schema.get_conn() as c:
            c.dead = True
            raise RuntimeError("the real failure")
    assert p.closed == [] and p.returned == ["c"]


def test_a_commit_on_a_dropped_connection_reports_the_drop_and_discards_it(fake_pool):
    conn = _Conn("c")
    p = fake_pool(conn)
    with pytest.raises(psycopg2.OperationalError):
        with schema.get_conn() as c:
            c.dead = True
    assert p.closed == ["c"]
