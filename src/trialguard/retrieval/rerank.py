"""Cross-encoder reranker for pool compression.

Retrieve wide → rerank → slice small. Decouples retrieval depth from agent cost.
Query: full patient note (not keywords — keywords cast wide, note judges richly).
Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU, general-domain).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("data/cache/rerank")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def _note_hash(note: str) -> str:
    return hashlib.sha256(note.encode()).hexdigest()[:16]


def rerank(
    query_note: str,
    candidates: list[tuple[str, float]],
    trial_texts: dict[str, str],
    top_k: int,
) -> list[tuple[str, float]]:
    """Rerank candidates with cross-encoder. Returns top_k (nct_id, score) sorted desc.

    candidates: (nct_id, retrieval_score) from upstream retrieval — order does not matter.
    trial_texts: nct_id → doc text (title + inclusion_criteria).
    Scores cached per query_note hash; re-runs cost zero model calls.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{_note_hash(query_note)}.json"

    nct_ids = [nct for nct, _ in candidates]

    if cache_path.exists():
        cached: dict[str, float] = json.loads(cache_path.read_text())
        scored = [(nct, cached[nct]) for nct in nct_ids if nct in cached]
        missing = [nct for nct in nct_ids if nct not in cached]
        if missing:
            # Partial cache hit — score missing candidates and merge.
            model = _get_model()
            pairs = [(query_note, trial_texts.get(nct, "")) for nct in missing]
            new_scores = model.predict(pairs)
            new_pairs = list(zip(missing, new_scores.tolist()))
            scored += new_pairs
            cached.update(dict(new_pairs))
            cache_path.write_text(json.dumps(cached))
    else:
        model = _get_model()
        pairs = [(query_note, trial_texts.get(nct, "")) for nct in nct_ids]
        scores = model.predict(pairs)  # single batch call — NOT a loop
        scored = list(zip(nct_ids, scores.tolist()))
        cache_path.write_text(json.dumps(dict(scored)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
