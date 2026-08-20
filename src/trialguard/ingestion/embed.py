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
import threading

EMBEDDING_DIM = 768

BGE_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
MEDCPT_QUERY_MAXLEN = 64
MEDCPT_ARTICLE_MAXLEN = 512

_bge_model = None
_medcpt = {}
# retrieve() fans its per-keyword searches out across threads, so first use can be
# concurrent. Without this, every thread that misses the cache loads its own copy
# of a 440 MB encoder — an out-of-memory kill on a 2 GB machine, and duplicated
# weights everywhere else.
_model_lock = threading.Lock()


def _backend() -> str:
    # Default MedCPT: on SIGIR (keyword) it beat BGE recall@10 +34%, MRR +21%.
    return os.environ.get("TG_EMBED_BACKEND", "medcpt").lower()


def _index_exclusion() -> bool:
    return os.environ.get("TG_INDEX_EXCLUSION", "1") == "1"


# Chunking. MedCPT's article encoder truncates at 512 tokens, and 42% of the
# ctgov_live corpus is longer than that — the document is built title | inclusion
# | exclusion, so what gets silently discarded is the exclusion criteria. A trial
# whose distinguishing text sits past the window is invisible to dense retrieval
# however good the query is.
#
# Chunks are packed on criterion boundaries rather than cut blindly at a token
# count: eligibility text is already a list of self-contained statements, and
# splitting one mid-sentence would produce a passage that matches nothing. The
# budget is in characters because it has to hold before the tokenizer runs;
# ~1800 chars is roughly 450 tokens, leaving room for the title and specials.
CHUNK_MAX_CHARS = 1800


def _chunking_enabled() -> bool:
    """Opt-in until the decisive measurement lands.

    Chunking is directionally right — 42% of ctgov_live breaches the window — but
    on SIGIR it bought recall@10 +6.1% at MRR -4.0%, which is not enough to adopt
    on, and SIGIR barely has the problem (1.30x expansion against TREC's 1.51x
    over 27.6% of trials). Defaulting it on would also silently invalidate every
    cached eval index, which is an expensive surprise for a change still being
    argued. Flip to "1" once TREC confirms.
    """
    return os.environ.get("TG_CHUNK_DOCS", "0") == "1"


def _hard_split(text: str, budget: int) -> list[str]:
    """Last resort for a single criterion longer than the whole budget."""
    return [text[i : i + budget] for i in range(0, len(text), budget)] or [text]


def chunk_document(trial: dict, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Split one trial into passages that survive the encoder's window.

    The title is repeated in every chunk: it is short, and without it a passage of
    bare exclusion criteria has no indication of what disease it belongs to.
    """
    title = (trial.get("title") or "").strip()
    parts = [p for p in (trial.get("inclusion_criteria") or []) if p]
    if _index_exclusion():
        parts += [p for p in (trial.get("exclusion_criteria") or []) if p]

    head = f"{title} | " if title else ""
    budget = max(200, max_chars - len(head))
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for part in parts:
        if len(part) > budget:
            if cur:
                chunks.append(head + " | ".join(cur))
                cur, cur_len = [], 0
            chunks.extend(head + piece for piece in _hard_split(part, budget))
            continue
        if cur and cur_len + len(part) + 3 > budget:
            chunks.append(head + " | ".join(cur))
            cur, cur_len = [], 0
        cur.append(part)
        cur_len += len(part) + 3

    if cur:
        chunks.append(head + " | ".join(cur))
    return chunks or [title or ""]


def embed_tag() -> str:
    """Cache-versioning tag: distinguishes model + doc-text config on disk."""
    incl = "excl" if _index_exclusion() else "noexcl"
    chunked = "_chunked" if _chunking_enabled() else ""
    return f"{_backend()}_{incl}{chunked}"


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
        with _model_lock:
            if _bge_model is None:  # re-check: another thread may have loaded it
                from trialguard.config import settings
                if settings.hf_token:
                    os.environ.setdefault("HF_TOKEN", settings.hf_token)
                from sentence_transformers import SentenceTransformer
                _bge_model = SentenceTransformer(BGE_MODEL, device=_device())
    return _bge_model


def _bge_encode_array(texts: list[str], is_query: bool, batch_size: int):
    import numpy as np

    model = _get_bge()
    inputs = [(QUERY_PREFIX + t) if is_query else t for t in texts]
    vecs = model.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 256,
    )
    return np.asarray(vecs, dtype=np.float32)


def _bge_encode(texts: list[str], is_query: bool, batch_size: int) -> list[list[float]]:
    return _bge_encode_array(texts, is_query, batch_size).tolist()


# ---- MedCPT backend ----

def _get_medcpt(is_query: bool):
    key = "query" if is_query else "article"
    if key not in _medcpt:
        with _model_lock:
            if key not in _medcpt:  # re-check: another thread may have loaded it
                from trialguard.config import settings
                if settings.hf_token:
                    os.environ.setdefault("HF_TOKEN", settings.hf_token)
                from transformers import AutoModel, AutoTokenizer
                name = MEDCPT_QUERY_MODEL if is_query else MEDCPT_ARTICLE_MODEL
                tok = AutoTokenizer.from_pretrained(name)
                model = AutoModel.from_pretrained(name).to(_device()).eval()
                _medcpt[key] = (tok, model)
    return _medcpt[key]


def _medcpt_encode_array(texts: list[str], is_query: bool, batch_size: int):
    import numpy as np
    import torch

    tok, model = _get_medcpt(is_query)
    maxlen = MEDCPT_QUERY_MAXLEN if is_query else MEDCPT_ARTICLE_MAXLEN
    device = _device()
    out = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)
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
            out[start : start + len(batch)] = embeds.cpu().numpy()
            if show:
                print(f"    embedded {min(start + batch_size, len(texts))}/{len(texts)}")
    return out


def _medcpt_encode(texts: list[str], is_query: bool, batch_size: int) -> list[list[float]]:
    return _medcpt_encode_array(texts, is_query, batch_size).tolist()


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
    """Embed a batch. Trial documents use is_query=False.

    Returns Python lists, which is what the database ingestion path wants and
    what psycopg2 adapts. Callers embedding a whole corpus should use
    embed_matrix instead — see the note there.
    """
    return _encode(texts, is_query=is_query, batch_size=batch_size)


def embed_matrix(texts: list[str], batch_size: int = 32, is_query: bool = False):
    """Embed a whole corpus into a preallocated float32 array.

    embed_batch materialises one Python float object per dimension. For a 39,455
    passage index that is 30 million objects — roughly 1 GB of interpreter
    overhead before the numpy copy even begins, on top of the corpus itself. It
    is what puts an 8 GB machine over the edge on TREC, and the array it is
    immediately converted into is only 121 MB.

    Same computation, same result, filled in place.
    """
    if _backend() == "medcpt":
        return _medcpt_encode_array(texts, is_query, batch_size)
    return _bge_encode_array(texts, is_query, batch_size)


def eligibility_text_for_embedding(trial: dict) -> str:
    """Build the document string: title + inclusion (+ exclusion when enabled)."""
    parts = [trial.get("title", "")]
    parts += trial.get("inclusion_criteria", [])
    if _index_exclusion():
        parts += trial.get("exclusion_criteria", [])
    return " | ".join(p for p in parts if p)
