"""Analyst node: assess every criterion of one trial in a single LLM call.

One Groq call per trial (all criteria batched) — never per-criterion. Responses
are cached by (patient, trial, prompt_version) so re-runs cost zero calls and
stay reproducible. The Analyst is instructed to quote verbatim; the deterministic
grounding check (verify/grounding.py) is what actually enforces it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

CACHE_DIR = Path("data/cache/analyst")
PROMPT_VERSION = "v1"

_SYSTEM_PROMPT = """\
You are a clinical trial eligibility analyst. Given a patient summary and a
trial's eligibility criteria, assess EACH criterion independently.

For each criterion output:
- "criterion": the criterion text, verbatim.
- "verdict": one of "met", "not_met", "cannot_determine".
- "quote": a VERBATIM span copied exactly from the trial or patient text that
  justifies your verdict. Copy characters exactly — do not paraphrase. If you
  cannot support a verdict with a real quote, use verdict "cannot_determine"
  and leave "quote" empty.
- "rationale": one short sentence.

Rules:
- "cannot_determine" is a valid, encouraged answer when evidence is absent.
  Never guess to fill a verdict.
- Output JSON only: {"assessments": [{...}, ...]}. No prose.\
"""


def _cache_key(patient_note: str, nct_id: str) -> str:
    h = hashlib.sha256(f"{PROMPT_VERSION}|{nct_id}|{patient_note}".encode()).hexdigest()[:20]
    return h


def _parse(raw: str) -> list[dict]:
    import re
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
    data = json.loads(raw)
    return data.get("assessments", [])


def _llm():
    from langchain_groq import ChatGroq
    from trialguard.config import settings
    return ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model, temperature=0)


def analyze_trial(
    patient_note: str,
    nct_id: str,
    criteria: list[str],
    handler=None,
) -> list[dict]:
    """Return raw per-criterion assessments (pre-grounding). Cached to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(patient_note, nct_id)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    crit_block = "\n".join(f"- {c}" for c in criteria)
    user = f"Patient summary:\n{patient_note}\n\nTrial {nct_id} criteria:\n{crit_block}"

    config = {"callbacks": [handler]} if handler is not None else {}
    resp = _llm().invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user)],
        config=config,
    )
    assessments = _parse(str(resp.content))
    cache_path.write_text(json.dumps(assessments))
    return assessments
