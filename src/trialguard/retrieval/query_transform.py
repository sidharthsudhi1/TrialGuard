"""Query transformation: extract clinical search keywords from patient notes."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/keywords")

# Matches provider._MAX_TOKENS["keywords"]; the budget gate gets the same output
# ceiling the client is configured with.
_MAX_OUTPUT_TOKENS = 1024

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


# The (provider, model) pair that produced every committed keyword cache entry.
# Those files back the Phase 2/7 retrieval numbers, so this pair keeps the
# original key format. Same carve-out as the analyst cache (agent/analyst.py).
LEGACY_PAIR = ("groq", "llama-3.3-70b-versatile")


def _note_hash(note: str) -> str:
    """Keyword cache key, discriminated by (provider, model) off the legacy pair.

    Keywords are model-dependent: they drive per-keyword dense+BM25 retrieval, so
    a different host's phrasing changes recall. Sharing one namespace across hosts
    would make a retrieval number unattributable to the model that produced it.
    """
    from trialguard.llm.provider import active_model, active_provider

    pair = (active_provider(), active_model())
    raw = note if pair == LEGACY_PAIR else f"{pair[0]}|{pair[1]}|{note}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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


def generate_keywords(patient_note: str, n_max: int = 12, handler=None) -> list[str]:
    """LLM-extract search keywords from patient note. Cached to disk by note hash.

    `handler` is optional and defaults to None so every existing caller keeps
    working. Until Phase 8 WS-5 this call was untraced and therefore invisible in
    Langfuse — a real LLM call, on the retrieval path, that no trace accounted
    for. The cost ledger is what surfaced it.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    note_key = _note_hash(patient_note)
    cache_path = CACHE_DIR / f"{note_key}.json"

    # Disk first: those files back the committed Phase 2/7 retrieval numbers, so
    # they stay authoritative and no existing result can shift. Postgres is the
    # layer underneath, for the served path, where the container filesystem is
    # replaced on every deploy — regenerated keywords change the queries that
    # drive ranking, so a lost cache silently changes what users are shown.
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    from trialguard.db.cache import cache_get, cache_put

    stored = cache_get("keywords", note_key)
    if stored:
        return stored

    from trialguard.agent.ratelimit import estimate_tokens
    from trialguard.llm.cost import active_ledger
    from trialguard.llm.provider import (
        active_model,
        active_provider,
        extract_usage,
        get_chat_model,
    )
    from trialguard.tracing import trace_config

    provider, model = active_provider(), active_model()
    ledger = active_ledger()
    # Search is not free. A cache-miss note costs one LLM call, and this path used
    # to make it without consulting the budget or recording what it spent, so the
    # ledger under-reported every search and the cap could not stop a scripted
    # client hitting /api/search. Deliberately outside the try below: a spent
    # budget must propagate to the caller, not degrade silently into worse
    # retrieval. Cache hits return before reaching here, so an exhausted budget
    # still serves every preset and every note already seen.
    ledger.check(
        estimate_tokens(_SYSTEM_PROMPT + patient_note) + _MAX_OUTPUT_TOKENS,
        provider=provider,
        model=model,
    )

    try:
        llm = get_chat_model("keywords")
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Patient summary:\n{patient_note}"),
        ], config=trace_config(
            handler, provider=provider, model=model, purpose="keywords"
        ))
        # Bill before parsing: the tokens were spent whether or not the response
        # is usable. Guarded separately because the money is already gone by this
        # point — losing the keywords over a bookkeeping failure would degrade
        # retrieval for a reason the user cannot act on. Logged at error level
        # because unbilled spend is exactly what this path was added to stop.
        try:
            ledger.record(extract_usage(response), provider, model)
        except Exception as e:  # noqa: BLE001
            log.error(
                "keyword call completed but was not billed (%s); the daily ledger "
                "is now under-reporting", type(e).__name__
            )
        keywords = _parse_keywords(str(response.content), n_max)
        if not keywords:
            raise ValueError("empty keyword list")
        cache_path.write_text(json.dumps(keywords))
        cache_put("keywords", note_key, keywords)
        return keywords
    except Exception as e:  # noqa: BLE001 — degrade, but never silently
        # Falling back to the raw note is a real quality cliff: AD-11 measured
        # keyword queries at +50% recall@10 over the narrative. Logged so the
        # cliff is visible instead of looking like ordinary retrieval.
        log.warning(
            "keyword extraction failed (%s); falling back to the raw note, "
            "which retrieves materially worse", type(e).__name__
        )
        return [patient_note]
