from unittest.mock import MagicMock, patch

from trialguard import tracing
from trialguard.eval.agent_metrics import _observability


def test_emit_scores_noop_without_client():
    with patch.object(tracing, "get_client", return_value=None):
        assert tracing.emit_scores({"faithfulness": 0.99}, session_id="s") is False


def test_emit_scores_pushes_numeric_and_categorical():
    client = MagicMock()
    with patch.object(tracing, "get_client", return_value=client):
        ok = tracing.emit_scores(
            {"faithfulness": 0.99, "dominant_rejection": "ungrounded_quote"},
            session_id="agent-eval-sigir",
        )
    assert ok is True
    kinds = {c.kwargs["name"]: c.kwargs["data_type"] for c in client.create_score.call_args_list}
    assert kinds == {"faithfulness": "NUMERIC", "dominant_rejection": "CATEGORICAL"}
    for c in client.create_score.call_args_list:
        assert c.kwargs["session_id"] == "agent-eval-sigir"
    client.flush.assert_called_once()


def test_observability_maps_verified_arm():
    verified = {
        "citation_precision": 0.9916,
        "unsupported_verdict_rate": 0.0084,
        "abstention_rate": 0.7505,
        "coverage": 0.2495,
        "mean_retries": 0.35,
    }
    scores = _observability(verified)
    assert scores["faithfulness"] == 0.9916
    assert scores["mean_retries"] == 0.35
    assert set(scores) == {
        "faithfulness",
        "unsupported_verdict_rate",
        "abstention_rate",
        "coverage",
        "mean_retries",
    }
