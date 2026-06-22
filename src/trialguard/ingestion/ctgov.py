"""ClinicalTrials.gov API v2 client.

Pulls oncology trials with defensive handling of null arrays,
inconsistent dates, and CommonMark eligibility text.
Rate-limited to stay under 50 req/min (1.5s spacing in config).
"""

from __future__ import annotations

import time
from typing import Generator

import httpx

from trialguard.config import settings

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
        "healthy_volunteers": elig_mod.get("healthyVolunteers") == "Yes",
        "last_updated": status_mod.get("lastUpdatePostDateStruct", {}).get("date", ""),
    }


def fetch_oncology_trials(
    max_trials: int = 5000,
) -> Generator[dict, None, None]:
    """Yield trial dicts from CT.gov v2 API, oncology scope, recruiting only."""

    params: dict = {
        "query.cond": "cancer OR oncology OR tumor OR neoplasm",
        "filter.overallStatus": ",".join(RECRUITING_STATUSES),
        "pageSize": settings.ctgov_page_size,
        "format": "json",
        "fields": FIELDS,
    }

    url = f"{settings.ctgov_api_base}/studies"
    fetched = 0

    with httpx.Client(timeout=30) as client:
        while fetched < max_trials:
            resp = client.get(url, params=params)

            if resp.status_code == 429:
                print("Rate limited. Sleeping 10s.")
                time.sleep(10)
                continue

            resp.raise_for_status()
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
            time.sleep(settings.ctgov_request_delay)

    print(f"Fetched {fetched} trials.")
