"""Tests for db/queries batch fetch (mocked DB)."""

from unittest.mock import MagicMock, patch

from trialguard.db import queries as Q


def test_get_trials_empty():
    assert Q.get_trials([]) == {}


def test_get_trials_batches_any():
    row = (
        "NCT1",
        "Title",
        "RECRUITING",
        "PHASE2",
        ["melanoma"],
        ["drug"],
        "elig raw",
        ["inc1"],
        ["exc1"],
        "18 Years",
        "N/A",
        "ALL",
        False,
        "ctgov_live",
    )
    cur = MagicMock()
    cur.fetchall.return_value = [row]
    cur.description = [(c,) for c in Q._TRIAL_COLS]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    with patch.object(Q, "get_conn") as gc:
        gc.return_value.__enter__.return_value = conn
        out = Q.get_trials(["NCT1", "NCT2"], source="ctgov_live")

    assert list(out) == ["NCT1"]
    assert out["NCT1"]["title"] == "Title"
    assert out["NCT1"]["exclusion_criteria"] == ["exc1"]
    sql = cur.execute.call_args[0][0]
    assert "ANY(%s)" in sql
    assert "source = %s" in sql


def test_get_trial_missing():
    cur = MagicMock()
    cur.fetchall.return_value = []
    cur.description = [(c,) for c in Q._TRIAL_COLS]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch.object(Q, "get_conn") as gc:
        gc.return_value.__enter__.return_value = conn
        assert Q.get_trial("NCT999") is None
