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
    always_bad = lambda note, nct, crit, handler=None: [
        {"criterion": "dx", "verdict": "met", "quote": "never in source at all here"}
    ]
    with patch.object(G, "analyze_trial", always_bad):
        state = G.assess("patient", "NCT1", ["dx"], SRC, max_retries=2)
    # exhausted retries -> unverifiable, never forced to a verdict
    assert state["assessments"][0]["verdict"] == "unverifiable"
