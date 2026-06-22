"""Embed trial eligibility text using sentence-transformers (MiniLM, CPU)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, CPU-friendly, one-time download
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vecs.tolist()


def eligibility_text_for_embedding(trial: dict) -> str:
    """Build the string we embed: title + inclusion criteria."""
    parts = [trial.get("title", "")]
    parts += trial.get("inclusion_criteria", [])
    return " | ".join(p for p in parts if p)
