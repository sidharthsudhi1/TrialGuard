"""ClinicalTrials.gov API v2 client.

Pulls oncology trials with defensive handling of null arrays,
inconsistent dates, and CommonMark eligibility text.
Rate-limited to stay under 50 req/min (1.5s spacing in config).

Two entry points:

- pull_trials: the corpus refresh. Returns a PullResult or raises. A crawl it
  cannot prove complete (fetched != countTotal, or CT.gov republished its data
  mid-crawl) is never returned, because the refresh reads "absent from the
  pull" as "left the enrolling set" and deletes on it.
- fetch_oncology_trials: a streaming sample for ingest and the E1c scale work,
  where stopping at max_trials is the point.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Generator
from dataclasses import dataclass, field

import httpx

from trialguard.config import settings

log = logging.getLogger(__name__)

FIELDS = ",".join([
    "NCTId",
    "BriefTitle",
    "OverallStatus",
    "Phase",
    "Condition",
    "InterventionName",
    "EligibilityCriteria",
    "MinimumAge",
    "MaximumAge",
    "Sex",
    "HealthyVolunteers",
    "LastUpdatePostDate",
])

# Fixed enums — case-sensitive
RECRUITING_STATUSES = ["RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"]
ALL_STATUSES = [
    *RECRUITING_STATUSES,
    "ACTIVE_NOT_RECRUITING", "COMPLETED", "TERMINATED",
    "SUSPENDED", "WITHDRAWN", "UNKNOWN",
]
ONCOLOGY_CONDITION = "cancer OR oncology OR tumor OR neoplasm"
ALL_CONDITIONS = "*"

USER_AGENT = "TrialGuard/0.1 (+https://github.com/sidharthsudhi1/TrialGuard)"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 6
MAX_BACKOFF_S = 60.0

# Fields whose absence across the pull is a schema-drift signal rather than a
# sparse record. Counted per pull so the refresh can refuse to publish a crawl
# where, say, every eligibilityCriteria came back empty.
TRACKED_FIELDS = ("title", "status", "eligibility_raw", "last_updated")

# Indirection so tests can run the retry schedule without waiting it out.
_sleep = time.sleep


class IncompletePull(RuntimeError):
    """The crawl could not be shown to hold every matching trial."""


class PullCapExceeded(RuntimeError):
    """More trials match than the caller allowed; truncating would read as expiry."""


@dataclass
class PullResult:
    trials: dict[str, dict]
    total_count: int
    pages: int = 0
    duplicates: int = 0
    retries: int = 0
    data_timestamp: str | None = None
    field_missing: dict[str, int] = field(default_factory=dict)


def _safe_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def _extract_trial(study: dict) -> dict:
    proto = study.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design_mod = proto.get("designModule", {})
    cond_mod = proto.get("conditionsModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})
    elig_mod = proto.get("eligibilityModule", {})

    phases = _safe_list(design_mod.get("phases"))
    phase = phases[0] if phases else None

    interventions = [
        i.get("name", "")
        for i in (arms_mod.get("interventions") or [])
        if i.get("name")
    ]

    # API v2 sends a JSON boolean. The "Yes" string is the v1 shape, kept so a
    # record in either form reads correctly.
    healthy = elig_mod.get("healthyVolunteers")

    return {
        "nct_id": id_mod.get("nctId", ""),
        "title": id_mod.get("briefTitle", ""),
        "status": status_mod.get("overallStatus", ""),
        "phase": phase,
        "conditions": _safe_list(cond_mod.get("conditions")),
        "interventions": interventions,
        "eligibility_raw": elig_mod.get("eligibilityCriteria", "") or "",
        "min_age": elig_mod.get("minimumAge", ""),
        "max_age": elig_mod.get("maximumAge", ""),
        "sex": elig_mod.get("sex", ""),
        "healthy_volunteers": healthy is True or healthy == "Yes",
        "last_updated": status_mod.get("lastUpdatePostDateStruct", {}).get("date", ""),
    }


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return min(MAX_BACKOFF_S, max(0.0, float(value)))
    except ValueError:
        return None


def _get(client: httpx.Client, url: str, params: dict | None = None) -> tuple[httpx.Response, int]:
    """GET with bounded retry. Returns (response, retries used).

    Retries transport errors and 429/5xx with full-jitter exponential backoff,
    honouring Retry-After. Re-requesting the same pageToken is idempotent, so a
    retried page cannot duplicate or skip records.
    """
    for attempt in range(MAX_ATTEMPTS):
        last = attempt == MAX_ATTEMPTS - 1
        try:
            resp = client.get(url, params=params)
        except httpx.TransportError as e:
            if last:
                raise
            log.warning("ctgov transport error (%s), attempt %d", type(e).__name__, attempt + 1)
            _sleep(random.uniform(0, min(MAX_BACKOFF_S, 2**attempt)))  # noqa: S311 -- backoff jitter
            continue
        if resp.status_code in RETRY_STATUSES and not last:
            wait = _retry_after(resp)
            if wait is None:
                wait = random.uniform(0, min(MAX_BACKOFF_S, 2**attempt))  # noqa: S311 -- backoff jitter
            log.warning("ctgov HTTP %d, retrying in %.1fs", resp.status_code, wait)
            _sleep(wait)
            continue
        resp.raise_for_status()
        return resp, attempt
    raise AssertionError("unreachable")


def _client() -> httpx.Client:
    return httpx.Client(timeout=30, headers={"User-Agent": USER_AGENT})


def _base_params(condition: str | None, statuses: list[str] | None) -> dict:
    params: dict = {
        "filter.overallStatus": ",".join(statuses or RECRUITING_STATUSES),
        "pageSize": settings.ctgov_page_size,
        "format": "json",
        "fields": FIELDS,
    }
    cond = ONCOLOGY_CONDITION if condition is None else condition
    if cond != ALL_CONDITIONS:
        params["query.cond"] = cond
    return params


def fetch_version(client: httpx.Client | None = None) -> dict:
    """CT.gov's /version: apiVersion and the dataTimestamp of its last publish."""
    own = client is None
    client = client or _client()
    try:
        resp, _ = _get(client, f"{settings.ctgov_api_base}/version")
        return resp.json()
    finally:
        if own:
            client.close()


def _crawl(client: httpx.Client, params: dict, max_trials: int | None) -> PullResult:
    url = f"{settings.ctgov_api_base}/studies"
    params = {**params, "countTotal": "true"}
    result = PullResult(trials={}, total_count=-1)
    missing = dict.fromkeys(TRACKED_FIELDS, 0)

    while True:
        resp, retries = _get(client, url, params)
        result.retries += retries
        result.pages += 1
        data = resp.json()

        if result.total_count < 0:
            if "totalCount" not in data:
                raise IncompletePull("first page carried no totalCount")
            result.total_count = int(data["totalCount"])
            if max_trials is not None and result.total_count > max_trials:
                raise PullCapExceeded(
                    f"{result.total_count} trials match, cap is {max_trials}"
                )

        for study in data.get("studies") or []:
            trial = _extract_trial(study)
            nct = trial["nct_id"]
            if not nct:
                continue
            if nct in result.trials:
                result.duplicates += 1
                continue
            for f in TRACKED_FIELDS:
                if not trial.get(f):
                    missing[f] += 1
            result.trials[nct] = trial

        next_token = data.get("nextPageToken")
        if not next_token:
            break
        params["pageToken"] = next_token
        _sleep(settings.ctgov_request_delay)

    result.field_missing = missing
    return result


def pull_trials(
    condition: str | None = None,
    statuses: list[str] | None = None,
    max_trials: int | None = None,
    client: httpx.Client | None = None,
) -> PullResult:
    """Crawl every matching trial, or raise.

    Completeness is proven two ways: the deduplicated count must equal the
    countTotal CT.gov reported, and /version's dataTimestamp must not move
    between the start and end of the crawl (a publish mid-crawl reshuffles
    pages). One re-crawl is attempted on failure, then IncompletePull.
    """
    own = client is None
    client = client or _client()
    params = _base_params(condition, statuses)
    try:
        reason = ""
        for attempt in range(2):
            before = fetch_version(client).get("dataTimestamp")
            result = _crawl(client, params, max_trials)
            after = fetch_version(client).get("dataTimestamp")
            fetched = len(result.trials)
            log.info(
                "ctgov crawl: fetched=%d total=%d pages=%d duplicates=%d retries=%d",
                fetched, result.total_count, result.pages, result.duplicates, result.retries,
            )
            if before != after:
                reason = f"CT.gov republished mid-crawl ({before} -> {after})"
            elif fetched != result.total_count:
                reason = f"fetched {fetched} of {result.total_count} reported"
            else:
                result.data_timestamp = after
                return result
            log.warning("ctgov crawl incomplete (attempt %d): %s", attempt + 1, reason)
        raise IncompletePull(reason)
    finally:
        if own:
            client.close()


def fetch_oncology_trials(
    max_trials: int = 5000,
    condition: str | None = None,
    statuses: list[str] | None = None,
) -> Generator[dict, None, None]:
    """Yield trial dicts from CT.gov v2 API, oncology scope, recruiting only.

    Stops at max_trials by design: this is the sampling path. Anything that
    diffs a pull against the corpus must use pull_trials instead.

    condition=ALL_CONDITIONS drops the condition filter entirely (E1c scale work);
    CLAUDE.md locks production scope to oncology, so widening it is a deliberate
    call at the call site rather than a default.
    """
    params = _base_params(condition, statuses)
    url = f"{settings.ctgov_api_base}/studies"
    fetched = 0

    with _client() as client:
        while fetched < max_trials:
            resp, _ = _get(client, url, params)
            data = resp.json()

            studies = data.get("studies", [])
            if not studies:
                break

            for study in studies:
                trial = _extract_trial(study)
                if trial["nct_id"]:
                    yield trial
                    fetched += 1
                    if fetched >= max_trials:
                        return

            next_token = data.get("nextPageToken")
            if not next_token:
                break

            params["pageToken"] = next_token
            _sleep(settings.ctgov_request_delay)

    log.info("Fetched %d trials.", fetched)


LOOKUP_CHUNK = 100


def lookup_ids(ids: list[str], client: httpx.Client | None = None) -> dict[str, str | None]:
    """Current overallStatus per NCT id, or None where CT.gov returned nothing.

    Used to confirm that a trial missing from a complete crawl really left scope
    before it is expired. filter.ids ignores the status filter, so a trial that
    moved to COMPLETED still comes back and says so.
    """
    own = client is None
    client = client or _client()
    url = f"{settings.ctgov_api_base}/studies"
    found: dict[str, str | None] = dict.fromkeys(ids)
    try:
        for i in range(0, len(ids), LOOKUP_CHUNK):
            chunk = ids[i : i + LOOKUP_CHUNK]
            params = {
                "filter.ids": ",".join(chunk),
                "fields": "NCTId,OverallStatus",
                "pageSize": LOOKUP_CHUNK,
                "format": "json",
            }
            resp, _ = _get(client, url, params)
            for study in resp.json().get("studies") or []:
                t = _extract_trial(study)
                if t["nct_id"] in found:
                    found[t["nct_id"]] = t["status"] or None
            _sleep(settings.ctgov_request_delay)
    finally:
        if own:
            client.close()
    return found
