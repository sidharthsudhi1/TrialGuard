"""Embed trial eligibility text using sentence-transformers.

Model: BAAI/bge-base-en-v1.5 (768-dim, retrieval-optimized, CPU-friendly).
AD-5 revision: switched from all-MiniLM-L6-v2 (384-dim) — MiniLM recall@100
on SIGIR was 0.49 ceiling; BGE retrieval benchmarks run 15-25% higher.

BGE query encoding requires a retrieval instruction prefix.
Document encoding uses no prefix (standard for BGE asymmetric retrieval).
"""

from __future__ import annotations

MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

# Required prefix for retrieval queries (NOT for documents being indexed).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def _get_model():
    global _model
    if _model is None:
        import os
        from trialguard.config import settings
        if settings.hf_token:
            os.environ.setdefault("HF_TOKEN", settings.hf_token)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str, is_query: bool = True) -> list[float]:
    """Embed a single text. Set is_query=False when embedding trial documents."""
    model = _get_model()
    input_text = (QUERY_PREFIX + text) if is_query else text
    vec = model.encode(input_text, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(
    texts: list[str],
    batch_size: int = 32,
    is_query: bool = False,
) -> list[list[float]]:
    """Embed a batch. Trial documents use is_query=False (no prefix)."""
    model = _get_model()
    inputs = [(QUERY_PREFIX + t) if is_query else t for t in texts]
    vecs = model.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vecs.tolist()


def eligibility_text_for_embedding(trial: dict) -> str:
    """Build the document string: title + inclusion criteria."""
    parts = [trial.get("title", "")]
    parts += trial.get("inclusion_criteria", [])
    return " | ".join(p for p in parts if p)
