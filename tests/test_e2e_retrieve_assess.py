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


def test_fanout_ranking_is_deterministic():
    """Concurrent per-keyword search must not let thread scheduling pick winners.

    RRF sums over lists so order cannot change a score, but sorted() is stable:
    if rankings were appended as tasks completed, exact ties would resolve by
    whichever query returned first. Backends here return tied scores and finish
    in deliberately jumbled order.
    """
    import time
    from unittest.mock import patch

    import trialguard.retrieval.query_transform as QT

    delays = {"kw-a": 0.03, "kw-b": 0.0, "kw-c": 0.015}

    def _dense(q, top_k=50, source=None):
        time.sleep(delays[q])
        return [(f"NCT-{q}-1", 0.9), ("NCT-TIE", 0.5)]

    def _bm25(q, top_k=50, source=None):
        time.sleep(delays[q] / 2)
        return [("NCT-TIE", 0.8), (f"NCT-{q}-2", 0.4)]

    with (
        patch.object(QT, "generate_keywords", return_value=["kw-a", "kw-b", "kw-c"]),
        patch.object(P, "dense_search", side_effect=_dense),
        patch.object(P, "bm25_search", side_effect=_bm25),
    ):
        runs = [
            P.retrieve("note", top_k=8, source="ctgov_live", use_keywords=True)[0]
            for _ in range(5)
        ]

    assert all(r == runs[0] for r in runs), "fan-out ranking varied across runs"
    assert runs[0][0][0] == "NCT-TIE"  # appears in all six lists
