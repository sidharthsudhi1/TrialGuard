"""Structured agent I/O. Criterion-level JSON only — no free-text verdicts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["met", "not_met", "cannot_determine", "unverifiable"]


class CriterionAssessment(BaseModel):
    criterion: str
    verdict: Verdict
    quote: str = Field(
        default="", description="Verbatim span from trial text supporting the verdict."
    )
    rationale: str = ""
    grounded: bool = False
    grounding_failure: bool = False


class TrialAssessment(BaseModel):
    nct_id: str
    assessments: list[CriterionAssessment]
    # trial-level roll-up: eligible only if no inclusion criterion is not_met/unverifiable
    trial_verdict: Literal["eligible", "excluded", "cannot_determine"] = "cannot_determine"


_ALLOWED_VERDICTS = {"met", "not_met", "cannot_determine", "unverifiable"}


def validate_assessments(raw: object) -> list[dict]:
    """Coerce untrusted analyst JSON into safe criterion dicts (OWASP LLM05).

    Runs on the live model-output boundary before caching. Drops anything that is
    not a dict, forces the verdict to a known enum (unknown -> cannot_determine so
    a malformed verdict can never be treated as decisive), coerces quote/criterion
    to strings, and keeps only the fields the pipeline reads — so a model response
    with extra or wrongly typed keys cannot inject unexpected behavior downstream.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        verdict = item.get("verdict")
        if verdict not in _ALLOWED_VERDICTS:
            verdict = "cannot_determine"
        out.append(
            {
                "criterion": str(item.get("criterion", "")),
                "verdict": verdict,
                "quote": str(item.get("quote", "") or ""),
                "rationale": str(item.get("rationale", "")),
            }
        )
    return out
