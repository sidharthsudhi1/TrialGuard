"""WS-1 against a real pgvector Postgres: provenance, the source guard, the backfill.

Needs TG_TEST_DATABASE_URL on a server with the pgvector extension available
(CI runs pgvector/pgvector:pg16). Each test gets its own schema, so the full
DDL runs in isolation and is dropped afterwards. Skips locally without one;
in CI a missing database fails, as in test_jobs.py.
"""

from __future__ import annotations

import os
import random

import pytest

from trialguard.db import schema
from trialguard.ingestion import loader
from trialguard.ingestion.provenance import content_hash, doc_hash, parser_version

pytestmark = pytest.mark.skipif(
    not os.environ.get("TG_TEST_DATABASE_URL") and not os.environ.get("CI"),
    reason="TG_TEST_DATABASE_URL not set",
)


def _trial(nct: str, source: str = "ctgov_live", **kw) -> dict:
    t = {
        "nct_id": nct,
        "title": f"Trial {nct}",
        "status": "RECRUITING",
        "inclusion_criteria": ["Adults with measurable disease"],
        "exclusion_criteria": ["Prior anti-PD-1 therapy"],
        "eligibility_raw": "Inclusion Criteria: ...",
        "last_updated": "2026-09-01",
        "healthy_volunteers": False,
        "embedding": [random.random() for _ in range(768)],  # noqa: S311
        "source": source,
    }
    t.update(kw)
    return t


def _row(nct: str) -> dict:
    with schema.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT source, doc_hash, content_hash, embed_tag, parser_version, "
            "first_seen_at, last_seen_at, title FROM trials WHERE nct_id = %s",
            (nct,),
        )
        r = cur.fetchone()
    keys = ("source", "doc_hash", "content_hash", "embed_tag", "parser_version",
            "first_seen_at", "last_seen_at", "title")
    return dict(zip(keys, r, strict=True)) if r else {}


def test_the_ddl_is_idempotent(pg_db):
    schema.init_schema()
    with schema.get_conn() as conn, conn.cursor() as cur:
        cur.execute(schema.PROVENANCE_DDL)
        cur.execute("SELECT count(*) FROM refresh_runs")
        assert cur.fetchone()[0] == 0


def test_an_upsert_stamps_provenance(pg_db):
    t = _trial("NCT00000001")
    loader.upsert_trials([t])

    row = _row("NCT00000001")
    assert row["doc_hash"] == doc_hash(t)
    assert row["content_hash"] == content_hash(t)
    assert row["parser_version"] == parser_version()
    assert row["embed_tag"]
    assert row["first_seen_at"] is not None


def test_a_re_upsert_keeps_first_seen_and_moves_last_seen(pg_db):
    loader.upsert_trials([_trial("NCT00000001")])
    first = _row("NCT00000001")

    loader.upsert_trials([_trial("NCT00000001", title="Revised title")])
    second = _row("NCT00000001")

    assert second["first_seen_at"] == first["first_seen_at"]
    assert second["last_seen_at"] >= first["last_seen_at"]
    assert second["title"] == "Revised title"
    assert second["doc_hash"] != first["doc_hash"]


def test_an_upsert_cannot_steal_another_sources_row(pg_db):
    """The PK is nct_id alone; without the guard an eval load relabels production."""
    loader.upsert_trials([_trial("NCT00000001", source="sigir")])

    with pytest.raises(loader.SourceCollision):
        loader.upsert_trials(
            [_trial("NCT00000001", title="hijack"), _trial("NCT00000002")]
        )

    assert _row("NCT00000001")["source"] == "sigir"
    assert _row("NCT00000001")["title"] == "Trial NCT00000001"
    # The whole batch rolls back, not just the colliding row.
    assert _row("NCT00000002") == {}


def test_a_duplicate_inside_one_batch_does_not_fail_it(pg_db):
    loader.upsert_trials([_trial("NCT00000001"), _trial("NCT00000001", title="later")])
    assert _row("NCT00000001")["title"] == "later"


def test_the_backfill_stamps_legacy_rows_once(pg_db, monkeypatch):
    from trialguard.scripts import migrate_provenance as mp

    t = _trial("NCT00000001")
    loader.upsert_trials([t])
    # Simulate a row written before provenance existed.
    with schema.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE trials SET doc_hash=NULL, content_hash=NULL, embed_tag=NULL, "
            "parser_version=NULL, first_seen_at=NULL, last_seen_at=NULL"
        )
    monkeypatch.setattr(
        mp, "verify_embed_tag", lambda s, th: {"tag": "medcpt_excl", "passed": True}
    )

    report = mp.migrate(dry_run=False, sample=10, threshold=0.999)

    row = _row("NCT00000001")
    assert report["rows"] == 1
    assert row["doc_hash"] == doc_hash(t)
    assert row["embed_tag"] == "medcpt_excl"
    assert row["parser_version"] == "unknown"
    assert row["first_seen_at"] is not None
    assert mp.migrate(dry_run=False, sample=10, threshold=0.999)["rows"] == 0


def test_a_failed_embedding_check_stamps_unknown(pg_db, monkeypatch):
    from trialguard.scripts import migrate_provenance as mp

    loader.upsert_trials([_trial("NCT00000001")])
    with schema.get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE trials SET doc_hash=NULL, embed_tag=NULL")
    monkeypatch.setattr(
        mp, "verify_embed_tag", lambda s, th: {"tag": "medcpt_excl", "passed": False}
    )

    mp.migrate(dry_run=False, sample=10, threshold=0.999)

    assert _row("NCT00000001")["embed_tag"] == "unknown"


def test_a_dry_run_writes_nothing(pg_db, monkeypatch):
    from trialguard.scripts import migrate_provenance as mp

    loader.upsert_trials([_trial("NCT00000001")])
    with schema.get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE trials SET doc_hash=NULL")
    monkeypatch.setattr(
        mp, "verify_embed_tag", lambda s, th: {"tag": "medcpt_excl", "passed": True}
    )

    report = mp.migrate(dry_run=True, sample=10, threshold=0.999)

    assert report["rows"] == 1
    assert _row("NCT00000001")["doc_hash"] is None
