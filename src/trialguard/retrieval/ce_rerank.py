"""Fine-tuned cross-encoder rerank of the deep pool (R2), fused with the deep ranking."""

from __future__ import annotations

from functools import lru_cache

from trialguard.retrieval.fusion import DEEP_TOP_K

CANDIDATES = 500  # the fused deep pool R2 was trained and measured on
RRF_K = 60
MAX_LEN = 512
BATCH = 128


def doc_text(t: dict) -> str:
    """Trial text the cross-encoder reads. Must match training (r2_crossencoder)."""
    incl = " ".join(t.get("inclusion_criteria") or [])
    excl = " ".join(t.get("exclusion_criteria") or [])
    return f"{t.get('title', '')}. Inclusion: {incl} Exclusion: {excl}"


def enabled(top_k: int, use_keywords: bool) -> bool:
    """Only the configuration R2 measured: keyword queries over a deep pool."""
    from trialguard.config import settings

    return bool(settings.retrieval_ce_model) and use_keywords and top_k >= DEEP_TOP_K


@lru_cache(maxsize=1)
def _model(name: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    return tok, model


def score(query: str, docs: list[str]) -> list[float]:
    import torch

    from trialguard.config import settings

    tok, model = _model(settings.retrieval_ce_model)
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(docs), BATCH):
            chunk = docs[i:i + BATCH]
            enc = tok([query] * len(chunk), chunk, truncation="only_second",
                      max_length=MAX_LEN, padding=True, return_tensors="pt")
            out += model(**enc).logits.view(-1).float().tolist()
    return out


def rerank(
    keywords: list[str], pool: list[tuple[str, float]], docs: dict[str, str]
) -> list[tuple[str, float]]:
    """RRF of the cross-encoder rank and the deep rank over `pool`, best first.

    Candidates with no text keep their deep rank and get the worst cross-encoder
    rank, so a missing row demotes a trial rather than dropping it.
    """
    ids = [n for n, _ in pool]
    scored = [n for n in ids if n in docs]
    s = score("; ".join(keywords), [docs[n] for n in scored]) if scored else []
    ce_order = [n for _, n in sorted(zip(s, scored, strict=True), key=lambda x: -x[0])]
    ce_rank = {n: r for r, n in enumerate(ce_order, start=1)}
    worst = len(ids) + 1
    fused = {
        n: 1 / (RRF_K + ce_rank.get(n, worst)) + 1 / (RRF_K + r)
        for r, n in enumerate(ids, start=1)
    }
    return sorted(fused.items(), key=lambda x: -x[1])
