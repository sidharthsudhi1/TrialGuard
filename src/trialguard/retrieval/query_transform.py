"""Query transformation: extract clinical search keywords from patient notes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

CACHE_DIR = Path("data/cache/keywords")

_SYSTEM_PROMPT = """\
You are a clinical trial retrieval assistant. Given a patient summary, extract a
ranked list of concise search keywords for finding relevant clinical trials.

Rules:
- Each keyword is a short search phrase of 2-6 words, NOT a full sentence.
- Cover: primary condition, disease stage/subtype, biomarkers/mutations,
  prior treatments, and major comorbidities or eligibility-relevant attributes.
- Order most-to-least important for trial matching.
- Output JSON only, no prose: {"keywords": ["...", "..."]}
- Maximum 12 keywords. Omit anything not useful for search.\
"""


def _note_hash(note: str) -> str:
    return hashlib.sha256(note.encode()).hexdigest()[:16]


def _parse_keywords(raw: str, n_max: int) -> list[str]:
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
    data = json.loads(raw)
    kws = data.get("keywords", [])
    seen: set[str] = set()
    result = []
    for kw in kws:
        normalized = kw.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
        if len(result) >= n_max:
            break
    return result


def generate_keywords(patient_note: str, n_max: int = 12) -> list[str]:
    """LLM-extract search keywords from patient note. Cached to disk by note hash."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{_note_hash(patient_note)}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    try:
        from trialguard.config import settings

        llm = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Patient summary:\n{patient_note}"),
        ])
        keywords = _parse_keywords(str(response.content), n_max)
        if not keywords:
            raise ValueError("empty keyword list")
        cache_path.write_text(json.dumps(keywords))
        return keywords
    except Exception:
        return [patient_note]
