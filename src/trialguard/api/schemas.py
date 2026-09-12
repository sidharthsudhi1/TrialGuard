"""Request/response models for the Stage A API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    note: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1)


class AssessRequest(BaseModel):
    note: str = Field(..., min_length=1)
    nct_ids: list[str] = Field(..., min_length=1)
    # Opt-in only. Depth is what moves surfaced recall (H1), and it is also what
    # spends money and minutes, so the caller has to ask for it by name rather
    # than reach it by sending a longer list.
    deep: bool = False


class LimitsResponse(BaseModel):
    """What a client needs to quote an honest cost and wait before submitting."""

    max_assess_trials: int
    max_assess_trials_deep: int
    assess_workers: int
    usd_per_trial: float
    seconds_per_trial: float
    # Served rather than hardcoded in the client: a note the UI offers but the
    # allowlist does not know is uncacheable, and that is how the demo ended up
    # paying full price on every run.
    presets: list[dict[str, str]] = []


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
