"""H2 — listwise LLM rerank: compress a deep candidate pool before the agent.

H1 measured that the agent converts retrieved trials at a depth-independent
rate, so surfaced recall scales with the pool it is handed (6.4x / 6.0x from
top-10 to top-100 on the two TREC cohorts). Deep pools are therefore a cost
problem, not a recall problem: a full assessment is ~29 s and ~$0.0005 per
trial, so top-500 is not servable.

This is the cheap middle stage. One LLM call reads ~50 candidate one-liners
against the patient note and returns the plausible ones in rank order, at
roughly 1/100th the tokens of assessing them. Unlike the cross-encoders R4
rejected, the judgment is the same kind the analyst already makes — read the
eligibility text, decide whether this patient could plausibly qualify — rather
than a web-search relevance score borrowed from another domain.

Ordering only: this stage never emits a verdict, so nothing it does can reach a
citation. Faithfulness is unaffected by construction.

Not on the production path until measured.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/listwise")

BATCH_SIZE = 50
_MAX_OUTPUT_TOKENS = 1024

_SYSTEM_PROMPT = """\
You are screening clinical trials for a patient. Given a patient summary and a
numbered list of trials, return the trials this patient could plausibly qualify
for, best first.

Rules:
- Judge plausibility only. You are screening, not deciding eligibility.
- Keep a trial if the patient's condition and situation fit its target
  population. Drop it if the trial clearly targets a different disease,
  a different age group, or a population the patient plainly is not in.
- When the listing is too brief to tell, keep the trial.
- Order the kept trials most-to-least plausible.
- Output JSON only, no prose: {"ranked": [12, 3, 47, ...]}
- Use the numbers shown. Never invent a number that is not in the list.\
"""


def _summarize(text: str, max_chars: int = 300) -> str:
    """One line per trial: title plus the head of its eligibility text.

    Deliberately short. The point of this stage is that it reads a pool for the
    price of assessing one trial, and that only holds if each candidate costs a
    line rather than a document.
    """
    flat = re.sub(r"\s+", " ", text).strip()
    return flat[:max_chars]


def _cache_key(note: str, nct_ids: list[str], model_name: str) -> str:
    """Keyed by (model, note, exact candidate list).

    The candidate list is part of the key because the model ranks within the
    batch it is shown: the same trial in a different pool is a different
    question, and reusing a score across pools would silently report one pool's
    ranking as another's. Same lesson as the rerank cache bug in 19c02a9.
    """
    payload = f"{model_name}|{note}|{'|'.join(nct_ids)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _parse_ranked(raw: str, n_candidates: int) -> list[int]:
    """Parse the model's index list, dropping anything out of range."""
    cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
    data = json.loads(cleaned)
    out, seen = [], set()
    for i in data.get("ranked", []):
        idx = int(i)
        if 0 <= idx < n_candidates and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _rank_batch(note: str, batch: list[tuple[str, str]], handler=None) -> list[str]:
    """Rank one batch of (nct_id, summary). Returns kept nct_ids in rank order."""
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

    listing = "\n".join(f"{i}. {summary}" for i, (_, summary) in enumerate(batch))
    user = f"Patient summary:\n{note}\n\nTrials:\n{listing}"

    ledger.check(
        estimate_tokens(_SYSTEM_PROMPT + user) + _MAX_OUTPUT_TOKENS,
        provider=provider,
        model=model,
    )

    llm = get_chat_model("listwise")
    response = llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user)],
        config=trace_config(handler, provider=provider, model=model, purpose="listwise"),
    )
    try:
        ledger.record(extract_usage(response), provider, model)
    except Exception as e:  # noqa: BLE001
        log.error(
            "listwise call completed but was not billed (%s); the daily ledger "
            "is now under-reporting", type(e).__name__
        )
    kept = _parse_ranked(str(response.content), len(batch))
    return [batch[i][0] for i in kept]


def listwise_rerank(
    note: str,
    candidates: list[tuple[str, float]],
    trial_texts: dict[str, str],
    top_k: int = 50,
    batch_size: int = BATCH_SIZE,
    handler=None,
) -> list[tuple[str, float]]:
    """Compress a deep candidate pool to top_k by LLM plausibility screening.

    Batches preserve retrieval order, so a batch is a contiguous slice of the
    fused ranking. Kept trials are re-scored by their position within the batch
    and batches are interleaved by rank, so a trial the model ranked first in
    batch 3 outranks one it ranked fifth in batch 1 — the model's judgment
    orders within a batch, and retrieval's ordering is not re-imposed across
    them beyond that.

    On any failure the original ranking is returned unchanged: this stage exists
    to improve an ordering that already works, so it must never be able to make
    retrieval worse than not running it.
    """
    if not candidates:
        return []

    nct_ids = [n for n, _ in candidates]
    from trialguard.llm.provider import active_model

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(note, nct_ids, active_model())}.json"
    if cache_path.exists():
        ranked = json.loads(cache_path.read_text())
        return [(n, 1.0 / (r + 1)) for r, n in enumerate(ranked)][:top_k]

    try:
        batches = [
            [(n, _summarize(trial_texts.get(n, ""))) for n, _ in candidates[i : i + batch_size]]
            for i in range(0, len(candidates), batch_size)
        ]
        per_batch = [_rank_batch(note, b, handler) for b in batches]

        # Interleave by within-batch rank so no batch is starved by its position.
        ranked: list[str] = []
        for rank in range(max((len(b) for b in per_batch), default=0)):
            for b in per_batch:
                if rank < len(b):
                    ranked.append(b[rank])

        cache_path.write_text(json.dumps(ranked))
        return [(n, 1.0 / (r + 1)) for r, n in enumerate(ranked)][:top_k]
    except Exception as e:  # noqa: BLE001 — degrade to retrieval order, never worse
        log.warning(
            "listwise rerank failed (%s); falling back to the retrieval ordering",
            type(e).__name__,
        )
        return candidates[:top_k]
