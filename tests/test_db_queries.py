"""Tests for db/queries batch fetch (mocked DB)."""

from unittest.mock import MagicMock, patch

from trialguard.db import queries as Q


def test_get_trials_empty():
    assert Q.get_trials([]) == {}


# Keyed by column name, not position: the previous fixture was a bare tuple that
# had to be re-counted by hand every time _TRIAL_COLS grew.
_ROW_VALUES = {
    "nct_id": "NCT1",
    "title": "Title",
    "status": "RECRUITING",
    "phase": "PHASE2",
    "conditions": ["melanoma"],
    "interventions": ["drug"],
    "eligibility_raw": "elig raw",
    "inclusion_criteria": ["inc1"],
    "exclusion_criteria": ["exc1"],
    "min_age": "18 Years",
    "max_age": "N/A",
    "sex": "ALL",
    "healthy_volunteers": False,
    "last_updated": "2026-08-14",
    "source": "ctgov_live",
}


def test_row_fixture_covers_every_selected_column():
    """A column added to the SELECT without a value here would fail obscurely."""
    assert set(_ROW_VALUES) == set(Q._TRIAL_COLS)


def test_get_trials_batches_any():
    row = tuple(_ROW_VALUES[c] for c in Q._TRIAL_COLS)
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
    # WS-3: served so a reader can see the age of the record a quote came from.
    assert out["NCT1"]["last_updated"] == "2026-08-14"
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
