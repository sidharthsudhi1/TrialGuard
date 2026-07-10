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
    try:
        return json.loads(raw).get("assessments", [])
    except json.JSONDecodeError:
        # LLM output truncated at the token cap mid-array. Salvage every complete
        # assessment object rather than dropping the whole trial.
        return _salvage(raw)


def _salvage(raw: str) -> list[dict]:
    """Extract complete assessment objects from a truncated assessments array."""
    idx = raw.find('"assessments"')
    bracket = raw.find("[", idx) if idx >= 0 else raw.find("[")
    if bracket >= 0:
        raw = raw[bracket + 1 :]  # scan inside the array; skip the wrapper object
    objs: list[dict] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(raw):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(raw[start : i + 1])
                    if "criterion" in obj:
                        objs.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1
    return objs


def _llm():
    from langchain_groq import ChatGroq
    from trialguard.config import settings
    # Groq free tier is TPM-capped (~12k tokens/min). max_retries lets the client
    # honor the 429 Retry-After header and back off instead of crashing the run.
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
        max_tokens=4096,
        max_retries=8,
    )


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
    # Space fresh calls to stay under the free-tier TPM window. Cache hits skip this.
    import os
    import time
    time.sleep(float(os.environ.get("TG_ANALYST_DELAY", "7")))
    return assessments
