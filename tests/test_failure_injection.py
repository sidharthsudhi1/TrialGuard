"""WS-7 — the phase's acceptance criteria as an executable suite.

Every work stream in Phase 10 claims a behaviour under a specific failure. This
file is where those claims are injected rather than asserted in prose, which is
what separates the phase from a ranked wish list. Nothing here needs a deployed
environment, a database or a provider: the failure is the fixture.

Cross-references the stream each case discharges, so a test that starts failing
names the promise it broke.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.test_api import STUB_ROWS, _parse_sse
from trialguard.api.app import create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TG_PROMPT_VERSION", "v4")
    monkeypatch.setattr("trialguard.config.settings.database_url", "")
    monkeypatch.setattr("trialguard.config.settings.api_cors_origin", "http://localhost:3000")
    monkeypatch.setattr("trialguard.config.settings.api_search_rate_per_min", 1000)
    monkeypatch.setattr("trialguard.config.settings.api_assess_rate_per_min", 1000)
    from trialguard.api.routes import _preset_notes

    _preset_notes.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    _preset_notes.cache_clear()


def _assess(client, nct_ids=("NCT0001",), assess_impl=None, get_trial=None):
    """Run one job to completion and return its parsed SSE log."""
    trial_lookup = get_trial or (lambda nct, source=None: STUB_ROWS.get(nct))
    with (
        patch("trialguard.agent.sanitize.detect_injection", return_value=False),
        patch("trialguard.db.queries.get_trial", side_effect=trial_lookup),
        patch("trialguard.llm.cost.active_ledger") as ledger,
    ):
        ledger.return_value.exhausted.return_value = False
        ledger.return_value.summary.return_value = {
            "usd": 2.0, "usd_cap": 2.0, "calls": 1, "date": "2026-09-07"
        }
        ledger.return_value.remaining_usd.return_value = 0.0
        stack = patch("trialguard.agent.graph.assess", side_effect=assess_impl)
        with stack if assess_impl else _null():
            created = client.post(
                "/api/assess",
                json={"note": "synthetic note", "nct_ids": list(nct_ids)},
            )
            assert created.status_code == 200, created.text
            job_id = created.json()["job_id"]
            with client.stream("GET", f"/api/assess/{job_id}") as stream:
                return job_id, _parse_sse("".join(stream.iter_text()))


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# --- provider returns something that is not the contract ---------------------


def test_malformed_provider_json_yields_no_verdict_rather_than_a_wrong_one():
    """WS-6 / AD-3. Truncation at the token cap is the common case and is
    salvaged; a response that is not JSON at all must produce nothing, never a
    verdict assembled from fragments."""
    from trialguard.agent.analyst import _parse

    assert _parse("I think the patient probably qualifies.") == []
    assert _parse('{"assessments": [{"criterion": "A", "verdict"') == []


def test_a_truncated_array_keeps_the_criteria_that_did_close():
    from trialguard.agent.analyst import _parse

    raw = (
        '{"assessments": [{"criterion": "A", "verdict": "met", "quote": "aa bb"},'
        '{"criterion": "B", "verdi'
    )

    out = _parse(raw)

    assert [a["criterion"] for a in out] == ["A"]


def test_a_verdict_the_enum_does_not_know_is_never_decisive():
    """OWASP LLM05. "eligible" is not a criterion verdict; coercing it to
    cannot_determine is what stops a malformed field reading as a decision."""
    from trialguard.agent.schema import validate_assessments

    out = validate_assessments([{"criterion": "A", "verdict": "eligible", "quote": "x"}])

    assert out[0]["verdict"] == "cannot_determine"


def test_a_provider_that_returns_prose_leaves_the_trial_undecided():
    """The whole path, not just the parser: an empty assessment list must roll up
    to cannot_determine and surface as needs_review.

    Patched at `graph.analyze_trial`, not `analyst.analyze_trial`: graph.py binds
    the name at import, so patching the defining module leaves the real function
    in place and the test makes a live provider call. It then passes for the wrong
    reason wherever credentials happen to exist. The stub's call count is asserted
    so a future rebinding cannot silently restore that.
    """
    from trialguard.agent.graph import assess

    with patch("trialguard.agent.graph.analyze_trial", return_value=[]) as analyst:
        state = assess(
            "62 M",
            "NCT0001",
            [{"text": "Stage IV NSCLC", "kind": "inclusion"}],
            "Inclusion: Stage IV NSCLC.",
            max_retries=0,
        )

    assert analyst.call_count == 1
    assert state["trial_verdict"] == "cannot_determine"
    assert state["trial_tier"] == "needs_review"


# --- the corpus moved under the request --------------------------------------


def test_a_trial_deleted_by_a_refresh_mid_session_is_reported_not_guessed(client):
    """WS-3. The refresh deletes trials that left the enrolling set, so a search
    result can be assessed after its row is gone. That must be an honest
    not_found, not a silent skip that leaves the client waiting."""
    _, events = _assess(client, get_trial=lambda nct, source=None: None)

    trial = next(payload for _, payload in events if payload.get("type") == "trial")
    assert trial["error"] == "not_found"
    assert trial["trial_verdict"] == "cannot_determine"
    assert trial["assessments"] == []
    assert events[-1][1]["status"] == "done"


def test_a_trial_whose_criteria_vanished_still_closes_the_stream(client):
    """A revised record can lose its parsed criteria while keeping its row."""
    empty = {**STUB_ROWS["NCT0001"], "inclusion_criteria": [], "exclusion_criteria": []}
    _, events = _assess(client, get_trial=lambda nct, source=None: empty)

    trial = next(payload for _, payload in events if payload.get("type") == "trial")
    assert trial["error"] == "no_criteria"
    assert events[-1][1]["status"] == "done"


def test_a_stale_corpus_row_is_visible_rather_than_silent():
    """WS-3/WS-4. Staleness nothing can see is the state the stream closed. The
    probe reads the same stamp /api/health reports."""
    import datetime as dt

    from trialguard.eval.served_probe import _corpus_age_hours

    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=3)).isoformat()
    assert _corpus_age_hours({"corpus_refresh": {"at": old}}) > 48
    assert _corpus_age_hours({}) is None


# --- the store is not there ---------------------------------------------------


def test_postgres_unavailable_at_job_create_refuses_instead_of_500(client):
    """WS-1. A store that cannot accept the job must say so: a 500 tells the
    client nothing and an accepted job id would promise work nobody is doing."""
    import psycopg2

    with (
        patch("trialguard.agent.sanitize.detect_injection", return_value=False),
        patch("trialguard.llm.cost.active_ledger") as ledger,
        patch.object(
            client.app.state.jobs,
            "create",
            side_effect=psycopg2.OperationalError(
                'connection to server at "ep-secret-host.neon.tech" failed'
            ),
        ),
    ):
        ledger.return_value.exhausted.return_value = False
        r = client.post(
            "/api/assess", json={"note": "synthetic note", "nct_ids": ["NCT0001"]}
        )

    assert r.status_code == 503
    # The message names the condition, never the connection string.
    assert "neon.tech" not in r.text
    assert "retry" in r.json()["detail"].lower()


def test_a_stream_for_a_job_that_does_not_exist_404s(client):
    r = client.get("/api/assess/does-not-exist")

    assert r.status_code == 404


# --- the money runs out mid-flight -------------------------------------------


def test_budget_exhausted_mid_job_fails_the_job_and_names_the_cap(client):
    """WS-6/E2. A run that stops paying must stop claiming, and the client has
    to be able to tell "we stopped" from "nothing matched"."""
    from trialguard.agent.ratelimit import BudgetExhausted

    calls = {"n": 0}

    def _die(note, nct_id, criteria, source_text, **kw):
        calls["n"] += 1
        if calls["n"] > 1:
            raise BudgetExhausted("Daily spend cap reached.")
        return {"trial_verdict": "eligible", "assessments": []}

    _, events = _assess(client, nct_ids=("NCT0001", "NCT0002"), assess_impl=_die)

    last = events[-1][1]
    assert last.get("type") == "error"
    detail = json.loads(last["error"])
    assert detail["error"] == "BudgetExhausted"
    assert detail["usd_cap"] == 2.0
    assert detail["remaining_usd"] == 0.0


# --- the verifier itself ------------------------------------------------------


def test_a_corrupted_quote_can_never_stand_as_decisive():
    """The faithfulness claim, injected: 100% deterministic catch rate is only
    true if a quote that is not in the source is rejected every time."""
    from trialguard.verify.grounding import ground_assessments

    source = "Inclusion: Stage IV NSCLC with measurable disease."
    out = ground_assessments(
        [
            {"criterion": "Stage IV", "verdict": "met", "quote": "Stage IV NSCLC"},
            {"criterion": "Stage IV", "verdict": "met", "quote": "Stage III NSCLC"},
            {"criterion": "Stage IV", "verdict": "not_met", "quote": "paraphrased away"},
        ],
        source,
    )

    assert out[0]["verdict"] == "met"
    assert [a["verdict"] for a in out[1:]] == ["unverifiable", "unverifiable"]
    assert all(a["grounding_failure"] for a in out[1:])


def test_an_analyst_that_answers_a_criterion_it_was_not_asked_about():
    """attach_kinds refuses to guess once the counts disagree: an unresolved
    criterion costs a needs_review row, a wrong guess inverts an exclusion."""
    from trialguard.agent.schema import attach_kinds, rollup_trial

    typed = [
        {"text": "Age >= 18", "kind": "inclusion"},
        {"text": "Prior chemotherapy", "kind": "exclusion"},
    ]
    out = attach_kinds([{"criterion": "Something else entirely", "verdict": "met"}], typed)

    assert out[0]["kind"] == "unknown"
    assert rollup_trial(out)["verdict"] == "cannot_determine"


# --- the boundary -------------------------------------------------------------


def test_a_note_carrying_an_identifier_is_refused_before_any_call(client):
    """Non-negotiable: detect_phi refuses at the boundary, before the LLM and
    before the trace. A test that only checks the response could pass while the
    note had already been sent."""
    with (
        patch("trialguard.llm.cost.active_ledger") as ledger,
        patch("trialguard.agent.graph.assess") as assessed,
    ):
        ledger.return_value.exhausted.return_value = False
        r = client.post(
            "/api/assess",
            json={
                "note": "John Smith, MRN 4432119, seen 03/14/2024 at 555-231-8890.",
                "nct_ids": ["NCT0001"],
            },
        )

    assert r.status_code == 400
    assessed.assert_not_called()
    # Names the identifier classes, never the matched values.
    assert "4432119" not in r.text
    assert "555-231-8890" not in r.text
