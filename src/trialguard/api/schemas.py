"""Request/response models for the Stage A API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    note: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1)


class AssessRequest(BaseModel):
    note: str = Field(..., min_length=1)
    nct_ids: list[str] = Field(..., min_length=1)


class AssessCreated(BaseModel):
    job_id: str


SYNTHETIC_NOTICE = (
    "TrialGuard accepts synthetic patient notes only. Do not submit real PHI. "
    "This is a research demo, not a clinical decision tool."
)
