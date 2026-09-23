"""A parser change must bump PARSER_VERSION, or the corpus silently mixes parses.

Rows record the parser version that produced them and the refresh re-parses
whatever is behind. That only works if the version actually moves when the
output does, which is what this pins: 30 real CT.gov eligibility texts
(tests/fixtures/ctgov/eligibility_samples.json, 10 chosen for group headers),
parsed and hashed.

If this fails after an intended parser change: bump PARSER_VERSION in
ingestion/normalise.py and add the new version and hash below.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trialguard.ingestion import normalise
from trialguard.ingestion.provenance import parser_version

SAMPLES = Path(__file__).parent / "fixtures/ctgov/eligibility_samples.json"

PINNED = {
    "2026-09-23.1": "67fb1f63ec9fea45",
}


@pytest.fixture(autouse=True)
def default_parse(monkeypatch):
    monkeypatch.delenv("TG_STRICT_CRITERIA", raising=False)


def _output_hash() -> str:
    rows = json.loads(SAMPLES.read_text())
    out = [normalise._split_criteria(r["eligibility_raw"]) for r in rows]
    return hashlib.sha256(json.dumps(out).encode()).hexdigest()[:16]


def test_parser_output_matches_its_version():
    version = normalise.PARSER_VERSION
    assert version in PINNED, f"no pinned hash for PARSER_VERSION {version!r}; add one"
    assert _output_hash() == PINNED[version], (
        "parser output changed but PARSER_VERSION did not: bump it in "
        "ingestion/normalise.py and pin the new hash here"
    )


def test_the_lax_parse_is_a_different_version(monkeypatch):
    """TG_STRICT_CRITERIA=0 changes the parse, so it must not share a version."""
    strict = parser_version()
    monkeypatch.setenv("TG_STRICT_CRITERIA", "0")
    assert parser_version() != strict
