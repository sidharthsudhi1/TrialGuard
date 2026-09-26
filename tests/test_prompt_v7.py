"""v7: retry instructions travel outside the patient-note fence."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from trialguard.agent import analyst as A
from trialguard.agent import graph as G

TYPED = [{"text": "Stage IV disease", "kind": "inclusion"}]
RETRY = "[Retry 1] These criteria need a verbatim quote"


@pytest.fixture(autouse=True)
def _fresh_graph():
    G._GRAPH = None
    yield
    G._GRAPH = None


def test_v7_system_prompt_is_v4():
    """Identical text is what lets attempt one share v4's cache."""
    assert A._PROMPTS["v7"] == A._PROMPTS["v4"]


def test_v7_first_attempt_is_byte_identical_to_v4():
    assert A.build_messages("note", "NCT1", TYPED, "v7") == A.build_messages(
        "note", "NCT1", TYPED, "v4"
    )


def test_v7_retry_sits_after_the_criteria_outside_the_fence():
    _, user = A.build_messages("62M stage IV", "NCT1", TYPED, "v7", RETRY)

    fence_end = user.index("</patient_note>")
    assert user.index(RETRY) > user.index("Stage IV disease") > fence_end


def test_v7_first_attempt_reads_the_v4_cache_and_retries_do_not():
    first_v7 = A.retry_cache_key("note", "NCT1", TYPED, "", "v7")
    first_v4 = A.retry_cache_key("note", "NCT1", TYPED, "", "v4")
    retry_v7 = A.retry_cache_key("note", "NCT1", TYPED, RETRY, "v7")
    retry_v4 = A.retry_cache_key("note", "NCT1", TYPED, RETRY, "v4")

    assert first_v7 == first_v4
    assert retry_v7 != retry_v4


def test_pre_v7_retry_keys_are_unchanged():
    """Committed retry entries were keyed on the note with the retry folded in."""
    folded = A._cache_key("note\n\n" + RETRY, "NCT1", TYPED, version="v4")
    assert A.retry_cache_key("note", "NCT1", TYPED, RETRY, "v4") == folded


@pytest.mark.parametrize(("version", "in_note"), [("v7", False), ("v4", True)])
def test_graph_routes_the_retry_by_version(monkeypatch, version, in_note):
    monkeypatch.setenv("TG_PROMPT_VERSION", version)
    calls = []

    def fails_then_grounds(note, nct_id, criteria, **kw):
        calls.append((note, kw.get("retry_context", "")))
        quote = "invented text" if len(calls) == 1 else "Stage IV disease"
        return [{"criterion": "Stage IV disease", "verdict": "met", "quote": quote}]

    with patch.object(G, "analyze_trial", side_effect=fails_then_grounds):
        G.assess("62M, Stage IV disease.", "NCT1", TYPED, "Stage IV disease", max_retries=2)

    note, retry = calls[1]
    assert ("[Retry 1]" in note) is in_note
    assert ("[Retry 1]" in retry) is not in_note
