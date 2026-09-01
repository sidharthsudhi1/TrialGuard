"""Cross-encoder reranker for pool compression.

Retrieve wide → rerank → slice small. Decouples retrieval depth from agent cost.

Default model is cross-encoder/ms-marco-MiniLM-L-6-v2, general-domain, which
lost 0.173 recall@50 on SIGIR (phase2_rerank.md). That run also used the full
patient note as the query — the same narrative-vs-eligibility mismatch AD-11
identified as the retrieval ceiling — so the model and the query form were never
separated. `model_name` exists to test a clinical cross-encoder against the same
baseline.

Not on the production path: eval only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("data/cache/rerank")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_models: dict = {}


def _get_model(model_name: str):
    if model_name not in _models:
        from sentence_transformers import CrossEncoder

        _models[model_name] = CrossEncoder(model_name)
    return _models[model_name]


def _cache_key(note: str, model_name: str) -> str:
    """Scores are model-specific, so the model belongs in the key.

    Without it a MedCPT run reads back ms-marco's cached scores and silently
    reports the old model's ranking as the new one's.
    """
    return hashlib.sha256(f"{model_name}|{note}".encode()).hexdigest()[:16]


def rerank(
    query_note: str,
    candidates: list[tuple[str, float]],
    trial_texts: dict[str, str],
    top_k: int,
    model_name: str | None = None,
) -> list[tuple[str, float]]:
    """Rerank candidates with cross-encoder. Returns top_k (nct_id, score) sorted desc.

    candidates: (nct_id, retrieval_score) from upstream retrieval — order does not matter.
    trial_texts: nct_id → doc text (title + inclusion_criteria).
    Scores cached per query_note hash; re-runs cost zero model calls.
    """
    model_name = model_name or RERANK_MODEL
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(query_note, model_name)}.json"

    nct_ids = [nct for nct, _ in candidates]

    if cache_path.exists():
        cached: dict[str, float] = json.loads(cache_path.read_text())
        scored = [(nct, cached[nct]) for nct in nct_ids if nct in cached]
        missing = [nct for nct in nct_ids if nct not in cached]
        if missing:
            # Partial cache hit — score missing candidates and merge.
            model = _get_model(model_name)
            pairs = [(query_note, trial_texts.get(nct, "")) for nct in missing]
            new_scores = model.predict(pairs)
            new_pairs = list(zip(missing, new_scores.tolist()))
            scored += new_pairs
            cached.update(dict(new_pairs))
            cache_path.write_text(json.dumps(cached))
    else:
        model = _get_model(model_name)
        pairs = [(query_note, trial_texts.get(nct, "")) for nct in nct_ids]
        scores = model.predict(pairs)  # single batch call — NOT a loop
        scored = list(zip(nct_ids, scores.tolist()))
        cache_path.write_text(json.dumps(dict(scored)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
