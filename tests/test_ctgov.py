"""The CT.gov client proves a crawl complete or refuses to return it."""

from __future__ import annotations

import httpx
import pytest

from trialguard.ingestion import ctgov
from trialguard.ingestion.ctgov import (
    IncompletePull,
    PullCapExceeded,
    _extract_trial,
    pull_trials,
)


def _study(nct: str, elig: str = "Inclusion Criteria:\n- adult", healthy=False) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": f"Trial {nct}"},
            "statusModule": {
                "overallStatus": "RECRUITING",
                "lastUpdatePostDateStruct": {"date": "2026-09-01"},
            },
            "eligibilityModule": {"eligibilityCriteria": elig, "healthyVolunteers": healthy},
        }
    }


class FakeCtgov:
    """A /studies + /version server whose failures are scripted per request."""

    def __init__(self, pages: list[list[str]], total: int | None = None):
        self.pages = pages
        self.total = sum(len(p) for p in pages) if total is None else total
        self.timestamps = ["2026-09-23T09:00:05"]
        self.version_calls = 0
        self.faults: list = []  # consumed one per /studies request
        self.studies_calls = 0
        self.drop_token_after: int | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/version"):
            ts = self.timestamps[min(self.version_calls, len(self.timestamps) - 1)]
            self.version_calls += 1
            return httpx.Response(200, json={"apiVersion": "2.0.5", "dataTimestamp": ts})

        self.studies_calls += 1
        if self.faults:
            fault = self.faults.pop(0)
            if isinstance(fault, Exception):
                raise fault
            if fault is not None:
                return fault

        token = request.url.params.get("pageToken")
        idx = int(token) if token else 0
        body: dict = {"studies": [_study(n) for n in self.pages[idx]]}
        if request.url.params.get("countTotal") == "true" and idx == 0:
            body["totalCount"] = self.total
        last = idx + 1 >= len(self.pages)
        if self.drop_token_after is not None and idx >= self.drop_token_after:
            last = True
        if not last:
            body["nextPageToken"] = str(idx + 1)
        return httpx.Response(200, json=body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(ctgov, "_sleep", slept.append)
    return slept


def test_a_complete_crawl_is_returned_with_its_evidence():
    fake = FakeCtgov([["NCT00000001", "NCT00000002"], ["NCT00000003"]])

    result = pull_trials(client=fake.client())

    assert sorted(result.trials) == ["NCT00000001", "NCT00000002", "NCT00000003"]
    assert result.total_count == 3
    assert result.pages == 2
    assert result.data_timestamp == "2026-09-23T09:00:05"
    assert result.field_missing["eligibility_raw"] == 0


def test_a_crawl_that_ends_early_is_refused():
    """The old client stopped at a missing nextPageToken and called it done."""
    fake = FakeCtgov([["NCT00000001"], ["NCT00000002"], ["NCT00000003"]])
    fake.drop_token_after = 1

    with pytest.raises(IncompletePull, match="fetched 2 of 3"):
        pull_trials(client=fake.client())


def test_a_short_crawl_is_retried_once_before_giving_up():
    fake = FakeCtgov([["NCT00000001"], ["NCT00000002"]])
    fake.drop_token_after = 0

    with pytest.raises(IncompletePull):
        pull_trials(client=fake.client())
    # Two full attempts: page 1 each time, never page 2.
    assert fake.studies_calls == 2


def test_a_republish_mid_crawl_triggers_a_recrawl():
    fake = FakeCtgov([["NCT00000001"], ["NCT00000002"]])
    # before/after of attempt 1 differ; attempt 2 is stable.
    fake.timestamps = ["2026-09-22T09:00:00", "2026-09-23T09:00:05"]

    result = pull_trials(client=fake.client())

    assert result.data_timestamp == "2026-09-23T09:00:05"
    assert len(result.trials) == 2


def test_duplicates_across_pages_are_counted_not_double_loaded():
    fake = FakeCtgov([["NCT00000001", "NCT00000002"], ["NCT00000002"]], total=2)

    result = pull_trials(client=fake.client())

    assert len(result.trials) == 2
    assert result.duplicates == 1


def test_exceeding_the_cap_raises_instead_of_truncating():
    """A truncated pull diffs as mass expiry, so the cap must never truncate."""
    fake = FakeCtgov([["NCT00000001", "NCT00000002", "NCT00000003"]])

    with pytest.raises(PullCapExceeded):
        pull_trials(max_trials=2, client=fake.client())


def test_429_honours_retry_after(no_sleep):
    fake = FakeCtgov([["NCT00000001"]])
    fake.faults = [httpx.Response(429, headers={"Retry-After": "7"})]

    result = pull_trials(client=fake.client())

    assert len(result.trials) == 1
    assert result.retries == 1
    assert 7.0 in no_sleep


def test_transient_5xx_and_timeouts_are_retried():
    fake = FakeCtgov([["NCT00000001"]])
    fake.faults = [httpx.Response(503), httpx.ReadTimeout("slow"), httpx.Response(502)]

    result = pull_trials(client=fake.client())

    assert len(result.trials) == 1
    assert result.retries == 3


def test_retries_are_bounded():
    """The old 429 loop retried forever on a fixed 10s sleep."""
    fake = FakeCtgov([["NCT00000001"]])
    fake.faults = [httpx.Response(503)] * ctgov.MAX_ATTEMPTS

    with pytest.raises(httpx.HTTPStatusError):
        pull_trials(client=fake.client())
    assert fake.studies_calls == ctgov.MAX_ATTEMPTS


def test_a_client_error_is_not_retried():
    fake = FakeCtgov([["NCT00000001"]])
    fake.faults = [httpx.Response(400)]

    with pytest.raises(httpx.HTTPStatusError):
        pull_trials(client=fake.client())
    assert fake.studies_calls == 1


def test_empty_eligibility_is_counted_for_the_audit():
    fake = FakeCtgov([["NCT00000001"]])

    def handler(request):
        resp = fake.handler(request)
        if request.url.path.endswith("/studies"):
            body = resp.json()
            body["studies"] = [_study("NCT00000001", elig="")]
            return httpx.Response(200, json=body)
        return resp

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = pull_trials(client=client)

    assert result.field_missing["eligibility_raw"] == 1


@pytest.mark.parametrize(("raw", "expected"), [(True, True), (False, False), ("Yes", True), (None, False)])
def test_healthy_volunteers_reads_the_v2_boolean(raw, expected):
    """v2 sends a JSON boolean; comparing it to "Yes" made every trial False."""
    assert _extract_trial(_study("NCT00000001", healthy=raw))["healthy_volunteers"] is expected


def test_a_real_captured_page_parses_to_the_shape_the_corpus_needs():
    """Captured from the live API (2026-09-23), all three accept healthy volunteers.

    Refresh it deliberately if CT.gov changes shape; tests never fetch it.
    """
    import json
    from pathlib import Path

    page = json.loads((Path(__file__).parent / "fixtures/ctgov/studies_page.json").read_text())
    fake = FakeCtgov([[]])

    def handler(request):
        if request.url.path.endswith("/version"):
            return fake.handler(request)
        return httpx.Response(200, json=page)

    result = pull_trials(client=httpx.Client(transport=httpx.MockTransport(handler)))

    assert len(result.trials) == 3
    assert all(n.startswith("NCT") and len(n) == 11 for n in result.trials)
    assert all(v == 0 for v in result.field_missing.values())
    assert all(t["healthy_volunteers"] is True for t in result.trials.values())


def test_lookup_reports_current_status_and_absence():
    """Expiry is confirmed by this: a trial CT.gov no longer returns maps to None."""
    from trialguard.ingestion.ctgov import lookup_ids

    seen: list[str] = []

    def handler(request):
        ids = request.url.params["filter.ids"].split(",")
        seen.append(request.url.params["filter.ids"])
        studies = [
            {"protocolSection": {"identificationModule": {"nctId": n},
                                 "statusModule": {"overallStatus": "COMPLETED"}}}
            for n in ids if n != "NCT00000002"
        ]
        return httpx.Response(200, json={"studies": studies})

    ids = [f"NCT{i:08d}" for i in range(1, 151)]
    result = lookup_ids(ids, client=httpx.Client(transport=httpx.MockTransport(handler)))

    assert len(seen) == 2  # chunks of 100
    assert result["NCT00000001"] == "COMPLETED"
    assert result["NCT00000002"] is None
    assert set(result) == set(ids)
