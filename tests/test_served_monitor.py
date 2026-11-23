import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

from trialguard.eval.served_monitor import check, compute, fetch_served_states


def _state(verdicts: list[str], retries: int = 0) -> dict:
    return {
        "retries": retries,
        "trial_verdict": "cannot_determine",
        "assessments": [{"verdict": v} for v in verdicts],
    }


def test_compute_rates():
    states = [
        _state(["met", "not_met", "cannot_determine", "unverifiable"], retries=1),
        _state(["met", "met"], retries=0),
    ]
    m = compute(states)
    assert m["n_trials"] == 2
    assert m["n_criteria"] == 6
    assert m["abstention_rate"] == round(2 / 6, 4)
    assert m["unverifiable_rate"] == round(1 / 6, 4)
    assert m["retry_rate"] == 0.5


def test_check_passes_inside_bands():
    m = compute([_state(["met", "cannot_determine"]) for _ in range(20)])
    outcome = check(m)
    assert outcome["checked"] is True
    assert outcome["passed"] is True


def test_check_diverges_on_abstain_everything(tmp_path):
    m = compute([_state(["cannot_determine"] * 5) for _ in range(20)])
    outcome = check(m)
    assert outcome["passed"] is False
    failed = [r["metric"] for r in outcome["results"] if not r["passed"]]
    assert "abstention_rate" in failed


def test_check_withholds_judgment_below_min_trials():
    m = compute([_state(["cannot_determine"] * 5)])
    outcome = check(m)
    assert outcome["checked"] is False
    assert outcome["passed"] is True


def _page(outputs: list[str], cursor: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(output=o) for o in outputs],
        meta=SimpleNamespace(cursor=cursor),
    )


def test_fetch_reads_served_root_observations_across_pages(monkeypatch):
    state = _state(["met", "unverifiable"], retries=1)
    pages = {
        None: _page([json.dumps(state), "not json"], "c2"),
        "c2": _page([json.dumps({"no": "assessments"}), json.dumps(state)], None),
    }
    calls: list[dict] = []

    def get_many(**kwargs):
        calls.append(kwargs)
        return pages[kwargs["cursor"]]

    client = SimpleNamespace(
        api=SimpleNamespace(observations=SimpleNamespace(get_many=get_many))
    )
    monkeypatch.setattr("trialguard.tracing.get_client", lambda: client)

    assert fetch_served_states(days=7) == [state, state]
    assert [c["cursor"] for c in calls] == [None, "c2"]
    assert "io" in calls[0]["fields"].split(",")

    query = {f["column"]: f for f in json.loads(calls[0]["filter"])}
    assert query["tags"]["operator"] == "all of"
    assert query["tags"]["value"] == ["served", "assess"]
    assert query["isRootObservation"]["value"] is True
    since = dt.datetime.fromisoformat(query["startTime"]["value"])
    window = dt.datetime.now(dt.UTC) - since
    assert dt.timedelta(days=7) <= window < dt.timedelta(days=7, minutes=1)


def test_bands_cover_committed_eval_anchors():
    # The bands must not flag the system the eval reports describe.
    bands = {b["metric"]: b for b in json.loads(
        Path("data/reports/served_baselines.json").read_text())["bands"]}
    for anchor in (0.5558, 0.6092, 0.593):
        b = bands["abstention_rate"]
        assert b["min"] <= anchor <= b["max"]
    for anchor in (0.0276, 0.0471):
        assert anchor <= bands["unverifiable_rate"]["max"]
