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


# Says what the system actually does, not what it wishes it did. The synthetic
# rule is procedural — nothing stops a real note being typed — and the served
# path traces full prompts to Langfuse, so submitted text leaves this
# infrastructure. Claiming otherwise would be the one dishonest string in a
# project about faithfulness.
SYNTHETIC_NOTICE = (
    "TrialGuard accepts synthetic patient notes only. Do not submit real PHI. "
    "Submitted notes and model prompts are logged to Langfuse for debugging. "
    "This is a research demo, not a clinical decision tool."
)
