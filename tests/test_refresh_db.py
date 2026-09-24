"""Refresh v2 end to end against a real pgvector Postgres.

CT.gov and the embedding model are stubbed at their boundaries (pull_trials,
fetch_version, lookup_ids, publish.embed); everything between them -- plan,
confirm, audit, ledger, lease, the publish transaction, soft expiry, the
readers -- runs for real.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from trialguard.db import schema
from trialguard.db.cache import cache_get
from trialguard.ingestion import publish as pub
from trialguard.ingestion.ctgov import PullResult
from trialguard.scripts import refresh as R

pytestmark = pytest.mark.skipif(
    not os.environ.get("TG_TEST_DATABASE_URL") and not os.environ.get("CI"),
    reason="TG_TEST_DATABASE_URL not set",
)

TS = "2026-09-23T09:00:05"


def _raw(nct: str, **kw) -> dict:
    t = {
        "nct_id": nct,
        "title": f"Trial {nct}",
        "status": "RECRUITING",
        "phase": "PHASE2",
        "conditions": ["melanoma"],
        "interventions": ["drug"],
        "eligibility_raw": (
            "Inclusion Criteria:\n\n* Adults with measurable melanoma\n\n"
            "Exclusion Criteria:\n\n* Prior anti-PD-1 therapy"
        ),
        "min_age": "18 Years",
        "max_age": "",
        "sex": "ALL",
        "healthy_volunteers": False,
        "last_updated": "2026-09-01",
    }
    t.update(kw)
    return t


class World:
    """The stubbed outside world: what CT.gov says, and what got embedded."""

    def __init__(self, monkeypatch):
        self.trials: dict[str, dict] = {}
        self.ts = TS
        self.status_of: dict[str, str | None] = {}
        self.embedded: list[str] = []
        monkeypatch.setattr(R, "pull_trials", self._pull)
        monkeypatch.setattr(R, "fetch_version", lambda: {"dataTimestamp": self.ts})
        monkeypatch.setattr(R, "lookup_ids", lambda ids: {n: self.status_of.get(n) for n in ids})
        monkeypatch.setattr(pub, "embed", self._embed)

    def _pull(self, statuses=None):
        trials = {n: dict(t) for n, t in self.trials.items()}
        return PullResult(trials=trials, total_count=len(trials), pages=1,
                          data_timestamp=self.ts,
                          field_missing={"title": 0, "status": 0,
                                         "eligibility_raw": sum(1 for t in trials.values()
                                                                if not t["eligibility_raw"]),
                                         "last_updated": 0})

    def _embed(self, trials, on_chunk=None, chunk=500):
        rng = np.random.default_rng(len(self.embedded))
        for t in trials:
            t["embedding"] = rng.random(768, dtype=np.float32)
            self.embedded.append(t["nct_id"])
        if on_chunk:
            on_chunk()

    def set(self, *trials: dict) -> None:
        self.trials = {t["nct_id"]: t for t in trials}


@pytest.fixture
def world(pg_db, monkeypatch):
    monkeypatch.delenv("TG_REFRESH_GATES", raising=False)
    return World(monkeypatch)


def _q(sql: str, params=()):
    with schema.get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _runs():
    return [r[0] for r in _q("SELECT outcome FROM refresh_runs ORDER BY started_at")]


def _active():
    return {r[0] for r in _q("SELECT nct_id FROM trials WHERE expired_at IS NULL")}


def test_a_first_run_bootstraps_and_publishes_a_version(world):
    world.set(_raw("NCT00000001"), _raw("NCT00000002"), _raw("NCT00000003"))

    code, report = R.run()

    assert code == R.EXIT_OK
    assert report["outcome"] == "published"
    assert report["summary"]["bootstrap"] is True
    assert _active() == {"NCT00000001", "NCT00000002", "NCT00000003"}
    assert _runs() == ["published"]
    version = cache_get("corpus", "version")
    assert version["run_id"] == report["run_id"]
    assert cache_get("corpus", "last_refresh")["at"]


def test_an_unpublished_ctgov_costs_one_request(world):
    world.set(_raw("NCT00000001"))
    R.run()
    world.embedded.clear()

    code, report = R.run()

    assert (code, report["outcome"]) == (R.EXIT_OK, "skipped_unchanged")
    assert world.embedded == []
    assert _runs() == ["published", "skipped_unchanged"]


def test_an_unchanged_corpus_is_a_noop_that_embeds_nothing(world):
    world.set(_raw("NCT00000001"), _raw("NCT00000002"))
    R.run()
    world.embedded.clear()
    world.ts = "2026-09-24T09:00:05"

    _, report = R.run()

    assert report["outcome"] == "noop"
    assert world.embedded == []
    # A noop does not move the corpus version, so serving processes do not reload.
    assert cache_get("corpus", "version")["run_id"] != report["run_id"]


def test_only_moved_text_is_re_embedded(world):
    world.set(_raw("NCT00000001"), _raw("NCT00000002"))
    R.run()
    world.embedded.clear()
    world.ts = "2026-09-24T09:00:05"
    world.set(
        _raw("NCT00000001", eligibility_raw="Inclusion Criteria:\n\n* Adults aged 21 or over"),
        _raw("NCT00000002", status="ENROLLING_BY_INVITATION", healthy_volunteers=True),
    )

    _, report = R.run()

    assert world.embedded == ["NCT00000001"]
    assert report["summary"]["changed"] == 1
    assert report["summary"]["meta_only"] == 1
    status, healthy = _q(
        "SELECT status, healthy_volunteers FROM trials WHERE nct_id = 'NCT00000002'"
    )[0]
    assert (status, healthy) == ("ENROLLING_BY_INVITATION", True)


def test_a_missing_trial_still_recruiting_is_kept(world):
    """F3: a crawl that drops an id is not evidence the trial left."""
    world.set(_raw("NCT00000001"), _raw("NCT00000002"))
    R.run()
    world.ts = "2026-09-24T09:00:05"
    world.set(_raw("NCT00000001"))
    world.status_of = {"NCT00000002": "RECRUITING"}

    _, report = R.run()

    assert report["summary"]["kept_missing"] == 1
    assert "NCT00000002" in _active()
    assert _q("SELECT missing_runs FROM trials WHERE nct_id = 'NCT00000002'")[0][0] == 1


def test_a_confirmed_departure_is_soft_expired_and_unretrievable(world):
    from trialguard.db.queries import get_trial
    from trialguard.retrieval.bm25 import bm25_search

    world.set(_raw("NCT00000001"), _raw("NCT00000002"))
    R.run()
    world.ts = "2026-09-24T09:00:05"
    world.set(_raw("NCT00000001"))
    world.status_of = {"NCT00000002": "COMPLETED"}

    _, report = R.run()

    assert report["expired"] == {"NCT00000002": "status:COMPLETED"}
    assert _active() == {"NCT00000001"}
    assert {n for n, _ in bm25_search("melanoma", source="ctgov_live")} == {"NCT00000001"}
    row = get_trial("NCT00000002", source="ctgov_live")
    assert row["expired_at"] is not None
    assert row["expired_reason"] == "status:COMPLETED"


def test_a_returning_trial_comes_back_without_an_embedding_call(world):
    world.set(_raw("NCT00000001"), _raw("NCT00000002"))
    R.run()
    world.ts = "2026-09-24T09:00:05"
    world.set(_raw("NCT00000001"))
    world.status_of = {"NCT00000002": "COMPLETED"}
    R.run()
    world.embedded.clear()
    world.ts = "2026-09-25T09:00:05"
    world.set(_raw("NCT00000001"), _raw("NCT00000002"))

    _, report = R.run()

    assert report["summary"]["resurrected"] == 1
    assert world.embedded == []
    assert "NCT00000002" in _active()


def test_a_failed_publish_leaves_the_corpus_untouched(world, monkeypatch):
    world.set(_raw("NCT00000001"), _raw("NCT00000002"))
    R.run()
    before = _q("SELECT nct_id, status, doc_hash FROM trials ORDER BY nct_id")
    world.ts = "2026-09-24T09:00:05"
    world.set(
        _raw("NCT00000001", eligibility_raw="Inclusion Criteria:\n\n* Changed"),
        _raw("NCT00000002", status="NOT_YET_RECRUITING"),
    )

    def boom(*a, **k):
        raise RuntimeError("connection dropped mid-publish")

    monkeypatch.setattr(pub, "update_metadata", boom)

    assert R.main([]) == R.EXIT_FAILED
    assert _q("SELECT nct_id, status, doc_hash FROM trials ORDER BY nct_id") == before
    assert _runs()[-1] == "failed"


def test_a_hard_gate_aborts_before_any_write(world):
    world.set(_raw("NCT00000001"), _raw("NCT00000002"))
    R.run()
    before = _q("SELECT nct_id, eligibility_raw FROM trials ORDER BY nct_id")
    world.ts = "2026-09-24T09:00:05"
    # A renamed field: every eligibilityCriteria comes back empty.
    world.set(_raw("NCT00000001", eligibility_raw=""), _raw("NCT00000002", eligibility_raw=""))

    code, report = R.run()

    assert code == R.EXIT_ABORTED
    assert "missing_eligibility_raw_hard" in report["gates_blocking"]
    assert _q("SELECT nct_id, eligibility_raw FROM trials ORDER BY nct_id") == before
    assert _runs()[-1] == "aborted_gate"


def test_a_second_run_is_refused_while_one_holds_the_lease(world):
    _q("INSERT INTO refresh_runs (run_id, source) VALUES ('other', 'ctgov_live') RETURNING 1")

    code, _ = R.run()

    assert code == R.EXIT_LOCKED
    assert _runs()[-1] == "skipped_locked"


def test_an_abandoned_lease_is_taken_over(world):
    world.set(_raw("NCT00000001"))
    _q(
        "INSERT INTO refresh_runs (run_id, source, heartbeat_at) "
        "VALUES ('dead', 'ctgov_live', NOW() - interval '2 hours') RETURNING 1"
    )

    code, _ = R.run()

    assert code == R.EXIT_OK
    assert _q("SELECT outcome, reason FROM refresh_runs WHERE run_id = 'dead'")[0] == (
        "failed", "abandoned",
    )


def test_a_dry_run_writes_nothing(world):
    world.set(_raw("NCT00000001"))

    code, report = R.run(dry_run=True)

    assert (code, report["outcome"]) == (R.EXIT_OK, "dry_run")
    assert report["summary"]["new"] == 1
    assert _q("SELECT count(*) FROM trials")[0][0] == 0
    assert _runs() == []
    assert world.embedded == []


def test_health_reports_the_failure_streak(world):
    from trialguard.ingestion import ledger

    world.set(_raw("NCT00000001"))
    R.run()
    world.ts = "2026-09-24T09:00:05"
    world.set(_raw("NCT00000001", eligibility_raw=""))
    R.run()
    world.ts = "2026-09-25T09:00:05"
    R.run()

    state = ledger.health("ctgov_live")
    assert state["consecutive_failures"] == 2
    assert state["last_attempt"]["outcome"] == "aborted_gate"


def test_the_vector_index_waits_for_data(pg_db):
    assert schema.ensure_vector_index() is False
    assert _q("SELECT to_regclass('trials_embedding_idx')")[0][0] is None
