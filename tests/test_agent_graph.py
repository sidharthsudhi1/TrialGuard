from unittest.mock import patch

from trialguard.agent import graph as G

SRC = "Inclusion Criteria: Histologically confirmed melanoma. Age 18 or older."


def _fake_analyst(grounded_quote):
    def _fn(note, nct_id, criteria, handler=None):
        # First attempt returns an ungrounded quote; retries return a grounded one.
        if "Retry" in note:
            return [{"criterion": "dx", "verdict": "met", "quote": "Histologically confirmed melanoma"}]
        return [{"criterion": "dx", "verdict": "met", "quote": grounded_quote}]
    return _fn


def test_single_pass_flags_unverifiable(monkeypatch):
    G._GRAPH = None
    with patch.object(G, "analyze_trial", _fake_analyst("fabricated diagnosis text here")):
        state = G.assess("patient", "NCT1", ["dx"], SRC, max_retries=0)
    a = state["assessments"][0]
    assert a["verdict"] == "unverifiable"
    assert state["trial_verdict"] == "cannot_determine"


def test_verified_retry_recovers_grounding(monkeypatch):
    G._GRAPH = None
    with patch.object(G, "analyze_trial", _fake_analyst("fabricated diagnosis text here")):
        state = G.assess("patient", "NCT1", ["dx"], SRC, max_retries=2)
    a = state["assessments"][0]
    assert a["verdict"] == "met" and a["grounded"]
    assert state["trial_verdict"] == "eligible"


def test_retry_capped(monkeypatch):
    G._GRAPH = None
    def always_bad(note, nct, crit, handler=None):
        return [{"criterion": "dx", "verdict": "met", "quote": "never in source at all here"}]
    with patch.object(G, "analyze_trial", always_bad):
        state = G.assess("patient", "NCT1", ["dx"], SRC, max_retries=2)
    # exhausted retries -> unverifiable, never forced to a verdict
    assert state["assessments"][0]["verdict"] == "unverifiable"


def test_retry_is_retrieval_aware(monkeypatch):
    # On retry the analyst must be handed the exact source span and the failed
    # criterion, not a generic nudge.
    G._GRAPH = None
    seen_notes = []

    def _capture(note, nct_id, criteria, handler=None):
        seen_notes.append(note)
        if "Retry" in note:
            return [{"criterion": "dx", "verdict": "met", "quote": "Histologically confirmed melanoma"}]
        return [{"criterion": "dx", "verdict": "met", "quote": "not in the source"}]

    with patch.object(G, "analyze_trial", _capture):
        G.assess("patient", "NCT1", ["dx"], SRC, max_retries=2)

    retry_note = next(n for n in seen_notes if "Retry" in n)
    assert SRC in retry_note            # the retrieved source span is injected
    assert "dx" in retry_note           # the specific failed criterion is named


def test_prompt_version_switches_cache_key(monkeypatch):
    from trialguard.agent import analyst as A
    monkeypatch.delenv("TG_PROMPT_VERSION", raising=False)
    assert A.prompt_version() == "v1"
    k1 = A._cache_key("note", "NCT1")
    monkeypatch.setenv("TG_PROMPT_VERSION", "v2")
    assert A.prompt_version() == "v2"
    assert A._cache_key("note", "NCT1") != k1   # v2 never collides with v1 cache
    assert "v2" in A._PROMPTS


def test_analyst_parse_salvages_truncated_json():
    from trialguard.agent.analyst import _parse
    good = '{"assessments":[{"criterion":"a","verdict":"met","quote":"x"},{"criterion":"b","verdict":"not_met","quote":"y"}]}'
    assert len(_parse(good)) == 2
    trunc = '{"assessments":[{"criterion":"a","verdict":"met","quote":"stage IV"},{"criterion":"b","verdict":"met","quote":"z"},{"criterion":"c","verdict":"met","quote":"cut off her'
    r = _parse(trunc)
    assert [o["criterion"] for o in r] == ["a", "b"]


def test_exclusion_met_excludes_trial():
    from trialguard.agent.schema import rollup_trial_verdict

    assert (
        rollup_trial_verdict(
            [
                {"kind": "inclusion", "verdict": "met"},
                {"kind": "exclusion", "verdict": "met"},
            ]
        )
        == "excluded"
    )


def test_exclusion_not_met_allows_eligible():
    from trialguard.agent.schema import rollup_trial_verdict

    assert (
        rollup_trial_verdict(
            [
                {"kind": "inclusion", "verdict": "met"},
                {"kind": "exclusion", "verdict": "not_met"},
            ]
        )
        == "eligible"
    )


def test_build_typed_criteria_includes_exclusion():
    from trialguard.agent.schema import build_typed_criteria

    trial = {
        "inclusion_criteria": ["Age >= 18"],
        "exclusion_criteria": ["Brain metastases"],
    }
    crit, truncated = build_typed_criteria(trial)
    assert not truncated
    assert crit == [
        {"text": "Age >= 18", "kind": "inclusion"},
        {"text": "Brain metastases", "kind": "exclusion"},
    ]


def test_criteria_truncation_flagged():
    from trialguard.agent.schema import MAX_CRITERIA, build_typed_criteria

    trial = {
        "inclusion_criteria": [f"inc{i}" for i in range(MAX_CRITERIA)],
        "exclusion_criteria": ["exc0"],
    }
    crit, truncated = build_typed_criteria(trial)
    assert truncated
    assert len(crit) == MAX_CRITERIA
    assert all(c["kind"] == "inclusion" for c in crit)
