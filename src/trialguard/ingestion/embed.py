"""Embed trial eligibility text.

Two selectable backends (env var TG_EMBED_BACKEND, default "bge"):

- bge:    BAAI/bge-base-en-v1.5 (768-dim). General-domain, asymmetric: queries
          take QUERY_PREFIX, documents do not.
- medcpt: NCBI MedCPT (768-dim). Domain-specific, trained on PubMed search logs;
          separate query and article encoders. This is TrialGPT's own retriever.

Doc text = title + inclusion (+ exclusion when TG_INDEX_EXCLUSION=1, the default).
Exclusion criteria carry disease/biomarker/prior-treatment signal — indexing them
recovers roughly half the eligibility text that was previously discarded.

embed_tag() encodes backend + exclusion flag so cached embeddings never load stale
against a changed config.
"""

from __future__ import annotations

import os

EMBEDDING_DIM = 768

BGE_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
MEDCPT_QUERY_MAXLEN = 64
MEDCPT_ARTICLE_MAXLEN = 512

_bge_model = None
_medcpt = {}


def _backend() -> str:
    # Default MedCPT: on SIGIR (keyword) it beat BGE recall@10 +34%, MRR +21%.
    return os.environ.get("TG_EMBED_BACKEND", "medcpt").lower()


def _index_exclusion() -> bool:
    return os.environ.get("TG_INDEX_EXCLUSION", "1") == "1"


def embed_tag() -> str:
    """Cache-versioning tag: distinguishes model + doc-text config on disk."""
    incl = "excl" if _index_exclusion() else "noexcl"
    return f"{_backend()}_{incl}"


def _device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---- BGE backend ----

def _get_bge():
    global _bge_model
    if _bge_model is None:
        from trialguard.config import settings
        if settings.hf_token:
            os.environ.setdefault("HF_TOKEN", settings.hf_token)
        from sentence_transformers import SentenceTransformer
        _bge_model = SentenceTransformer(BGE_MODEL, device=_device())
    return _bge_model


def _bge_encode(texts: list[str], is_query: bool, batch_size: int) -> list[list[float]]:
    model = _get_bge()
    inputs = [(QUERY_PREFIX + t) if is_query else t for t in texts]
    vecs = model.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 256,
    )
    return vecs.tolist()


# ---- MedCPT backend ----

def _get_medcpt(is_query: bool):
    key = "query" if is_query else "article"
    if key not in _medcpt:
        from trialguard.config import settings
        if settings.hf_token:
            os.environ.setdefault("HF_TOKEN", settings.hf_token)
        import torch
        from transformers import AutoModel, AutoTokenizer
        name = MEDCPT_QUERY_MODEL if is_query else MEDCPT_ARTICLE_MODEL
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(_device()).eval()
        _medcpt[key] = (tok, model)
    return _medcpt[key]


def _medcpt_encode(texts: list[str], is_query: bool, batch_size: int) -> list[list[float]]:
    import torch
    tok, model = _get_medcpt(is_query)
    maxlen = MEDCPT_QUERY_MAXLEN if is_query else MEDCPT_ARTICLE_MAXLEN
    device = _device()
    out: list[list[float]] = []
    show = len(texts) > 256
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tok(
                batch,
                truncation=True,
                padding=True,
                max_length=maxlen,
                return_tensors="pt",
            ).to(device)
            # MedCPT pools on the [CLS] (first) token.
            embeds = model(**enc).last_hidden_state[:, 0, :]
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            out.extend(embeds.cpu().tolist())
            if show:
                print(f"    embedded {min(start + batch_size, len(texts))}/{len(texts)}")
    return out


# ---- public API ----

def _encode(texts: list[str], is_query: bool, batch_size: int) -> list[list[float]]:
    if _backend() == "medcpt":
        return _medcpt_encode(texts, is_query, batch_size)
    return _bge_encode(texts, is_query, batch_size)


def embed_text(text: str, is_query: bool = True) -> list[float]:
    """Embed a single text. Set is_query=False when embedding trial documents."""
    return _encode([text], is_query=is_query, batch_size=1)[0]


def embed_batch(
    texts: list[str],
    batch_size: int = 32,
    is_query: bool = False,
) -> list[list[float]]:
    """Embed a batch. Trial documents use is_query=False."""
    return _encode(texts, is_query=is_query, batch_size=batch_size)


def eligibility_text_for_embedding(trial: dict) -> str:
    """Build the document string: title + inclusion (+ exclusion when enabled)."""
    parts = [trial.get("title", "")]
    parts += trial.get("inclusion_criteria", [])
    if _index_exclusion():
        parts += trial.get("exclusion_criteria", [])
    return " | ".join(p for p in parts if p)
