"""WS-3: the corpus refresh notices revised records and costs nothing when idle."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from trialguard.scripts.refresh import plan_refresh, refresh


def _t(nct: str, status: str = "RECRUITING", updated: str = "2026-08-01") -> dict:
    return {"nct_id": nct, "status": status, "last_updated": updated}


# --------------------------------------------------------------------------
# The diff, directly
# --------------------------------------------------------------------------


def test_an_unchanged_corpus_plans_no_work():
    fresh = {"NCT1": _t("NCT1"), "NCT2": _t("NCT2")}
    existing = {"NCT1": ("RECRUITING", "2026-08-01"), "NCT2": ("RECRUITING", "2026-08-01")}

    assert plan_refresh(fresh, existing) == (set(), set(), [], [])


def test_a_moved_update_date_is_revised_not_restatused():
    """The case status-only comparison missed: text may have changed under it."""
    fresh = {"NCT1": _t("NCT1", updated="2026-09-01")}
    existing = {"NCT1": ("RECRUITING", "2026-08-01")}

    expired, new, revised, restatused = plan_refresh(fresh, existing)
    assert revised == ["NCT1"]
    assert restatused == []
    assert (expired, new) == (set(), set())


def test_a_status_change_without_a_date_change_stays_cheap():
    fresh = {"NCT1": _t("NCT1", status="ENROLLING_BY_INVITATION")}
    existing = {"NCT1": ("RECRUITING", "2026-08-01")}

    _, _, revised, restatused = plan_refresh(fresh, existing)
    assert revised == []
    assert restatused == ["NCT1"]


def test_a_trial_that_left_the_enrolling_set_expires():
    fresh = {"NCT1": _t("NCT1")}
    existing = {"NCT1": ("RECRUITING", "2026-08-01"), "NCT2": ("RECRUITING", "2026-08-01")}

    expired, new, _, _ = plan_refresh(fresh, existing)
    assert expired == {"NCT2"}
    assert new == set()


def test_a_trial_absent_from_the_corpus_is_new():
    expired, new, revised, restatused = plan_refresh({"NCT9": _t("NCT9")}, {})
    assert new == {"NCT9"}
    assert (expired, revised, restatused) == (set(), [], [])


def test_a_null_stored_date_counts_as_revised():
    """Rows ingested before the column was populated must be reconciled once."""
    _, _, revised, _ = plan_refresh(
        {"NCT1": _t("NCT1")}, {"NCT1": ("RECRUITING", None)}
    )
    assert revised == ["NCT1"]


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


@contextmanager
def _stub_db(rows: list[tuple]):
    """A get_conn whose only query is the corpus SELECT."""
    cur = MagicMock()
    cur.fetchall.return_value = rows
    cur.__enter__ = lambda s: cur
    cur.__exit__ = lambda *a: False
    conn = MagicMock()
    conn.cursor.return_value = cur

    @contextmanager
    def fake_get_conn():
        yield conn

    with patch("trialguard.scripts.refresh.get_conn", fake_get_conn):
        yield cur


def _run(fresh: list[dict], rows: list[tuple]):
    with (
        _stub_db(rows),
        patch("trialguard.scripts.refresh.fetch_oncology_trials", return_value=fresh),
        patch("trialguard.scripts.refresh.normalise_trial", side_effect=lambda t: t),
        patch("trialguard.scripts.refresh.embed_batch") as embed,
        patch("trialguard.scripts.refresh.upsert_trials") as upsert,
        patch("trialguard.scripts.refresh.cache_put") as cache_put,
    ):
        summary = refresh()
    return summary, embed, upsert, cache_put


def test_a_noop_refresh_makes_zero_embedding_calls():
    """WS-3 acceptance: idempotent, and never re-embeds unchanged text."""
    fresh = [_t("NCT1"), _t("NCT2")]
    rows = [("NCT1", "RECRUITING", "2026-08-01"), ("NCT2", "RECRUITING", "2026-08-01")]

    summary, embed, upsert, _ = _run(fresh, rows)

    embed.assert_not_called()
    upsert.assert_not_called()
    assert summary["embedded"] == 0
    assert summary == {
        "new": 0,
        "expired": 0,
        "revised": 0,
        "restatused": 0,
        "corpus": 2,
        "embedded": 0,
    }


def test_a_revised_record_is_re_embedded():
    fresh = [_t("NCT1", updated="2026-09-04"), _t("NCT2")]
    rows = [("NCT1", "RECRUITING", "2026-08-01"), ("NCT2", "RECRUITING", "2026-08-01")]

    summary, embed, upsert, _ = _run(fresh, rows)

    assert embed.call_count == 1
    assert upsert.call_count == 1
    assert [t["nct_id"] for t in upsert.call_args[0][0]] == ["NCT1"]
    assert summary["revised"] == 1


def test_a_partial_pull_aborts_without_deleting_the_corpus():
    """A failed CT.gov pull must not mass-expire what it could not fetch."""
    rows = [(f"NCT{i}", "RECRUITING", "2026-08-01") for i in range(10)]

    summary, embed, upsert, cache_put = _run([_t("NCT0")], rows)

    assert summary is None
    embed.assert_not_called()
    upsert.assert_not_called()
    cache_put.assert_not_called()


def test_the_refresh_records_when_it_ran():
    """Staleness nothing can observe is the state this work stream ends."""
    fresh = [_t("NCT1")]
    rows = [("NCT1", "RECRUITING", "2026-08-01")]

    _, _, _, cache_put = _run(fresh, rows)

    namespace, key, value = cache_put.call_args[0]
    assert (namespace, key) == ("corpus", "last_refresh")
    assert value["corpus"] == 1
    assert "at" in value
