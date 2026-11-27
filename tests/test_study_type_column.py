"""study_type is stored, hashed and migrated like every other CT.gov field."""

from __future__ import annotations


def test_study_type_is_a_stored_column():
    from trialguard.db.schema import DDL, STUDY_TYPE_DDL
    from trialguard.ingestion.loader import COLUMNS

    assert "study_type" in COLUMNS
    assert "ADD COLUMN IF NOT EXISTS study_type TEXT" in STUDY_TYPE_DDL
    assert STUDY_TYPE_DDL in DDL


def test_study_type_change_is_a_content_change():
    from trialguard.ingestion.provenance import content_hash

    base = {"nct_id": "NCT00000001", "title": "t", "study_type": "INTERVENTIONAL"}
    assert content_hash(base) != content_hash({**base, "study_type": "OBSERVATIONAL"})


def test_row_carries_study_type_for_both_upsert_templates():
    from trialguard.ingestion.loader import META_TEMPLATE, TEMPLATE, row_for

    row = row_for({"nct_id": "NCT00000001", "study_type": "OBSERVATIONAL"}, "ctgov_live",
                  with_embedding=False)
    assert row["study_type"] == "OBSERVATIONAL"
    assert "%(study_type)s" in TEMPLATE
    assert "%(study_type)s" in META_TEMPLATE
