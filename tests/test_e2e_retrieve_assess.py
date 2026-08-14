"""Cached end-to-end: retrieve → assess without live LLM or Neon.

Exercises the production retrieve() orchestration (dense + FTS fused via RRF)
with the dense/bm25 backends stubbed, then runs the real LangGraph assess path
with a stubbed analyst — so the join between pipeline and agent is covered in CI.
"""

from unittest.mock import patch

from trialguard.agent import graph as G
from trialguard.retrieval import pipeline as P


def test_retrieve_fuses_stubbed_backends():
    with patch.object(P, "dense_search", return_value=[("NCT1", 0.9), ("NCT2", 0.5)]):
        with patch.object(P, "bm25_search", return_value=[("NCT2", 0.8), ("NCT3", 0.4)]):
            hits, lat = P.retrieve("melanoma stage IV", top_k=2, source="ctgov_live")
    assert len(hits) == 2
    assert {h[0] for h in hits} <= {"NCT1", "NCT2", "NCT3"}
    assert "dense_ms" in lat and "bm25_ms" in lat


def test_retrieve_then_assess_end_to_end():
    """Production join: retrieve() hits → typed criteria → assess() roll-up."""
    G._GRAPH = None
    trial_row = {
        "nct_id": "NCT1",
        "title": "Melanoma study",
        "status": "RECRUITING",
        "eligibility_raw": (
            "Inclusion Criteria: Histologically confirmed melanoma. Age 18 or older. "
            "Exclusion Criteria: Active brain metastases."
        ),
        "inclusion_criteria": ["Histologically confirmed melanoma", "Age 18 or older"],
        "exclusion_criteria": ["Active brain metastases"],
    }

    def _analyst(note, nct_id, criteria, handler=None):
        # Return grounded quotes for inclusion met / exclusion not_met → eligible
        return [
            {
                "criterion": "Histologically confirmed melanoma",
                "verdict": "met",
                "quote": "Histologically confirmed melanoma",
            },
            {
                "criterion": "Age 18 or older",
                "verdict": "met",
                "quote": "Age 18 or older",
            },
            {
                "criterion": "Active brain metastases",
                "verdict": "not_met",
                "quote": "no brain metastases",  # from patient note
            },
        ]

    note = "58-year-old with melanoma, no brain metastases."
    with patch.object(P, "dense_search", return_value=[("NCT1", 0.9)]):
        with patch.object(P, "bm25_search", return_value=[("NCT1", 0.8)]):
            hits, _ = P.retrieve(note, top_k=1, source="ctgov_live")
    assert hits[0][0] == "NCT1"

    from trialguard.agent.schema import build_typed_criteria

    criteria, truncated = build_typed_criteria(trial_row)
    assert not truncated
    assert any(c["kind"] == "exclusion" for c in criteria)

    with patch.object(G, "analyze_trial", _analyst):
        state = G.assess(note, "NCT1", criteria, trial_row["eligibility_raw"], max_retries=0)

    assert state["trial_verdict"] == "eligible"
    kinds = {a["criterion"]: a["kind"] for a in state["assessments"]}
    assert kinds["Active brain metastases"] == "exclusion"
