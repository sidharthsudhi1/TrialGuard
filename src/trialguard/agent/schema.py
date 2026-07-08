"""Structured agent I/O. Criterion-level JSON only — no free-text verdicts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["met", "not_met", "cannot_determine", "unverifiable"]


class CriterionAssessment(BaseModel):
    criterion: str
    verdict: Verdict
    quote: str = Field(default="", description="Verbatim span from trial text supporting the verdict.")
    rationale: str = ""
    grounded: bool = False
    grounding_failure: bool = False


class TrialAssessment(BaseModel):
    nct_id: str
    assessments: list[CriterionAssessment]
    # trial-level roll-up: eligible only if no inclusion criterion is not_met/unverifiable
    trial_verdict: Literal["eligible", "excluded", "cannot_determine"] = "cannot_determine"
