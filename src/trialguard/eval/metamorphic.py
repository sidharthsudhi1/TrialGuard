"""Sentence edits for the G1 metamorphic check. No model calls."""

from __future__ import annotations

import re

from trialguard.verify.grounding import normalize


def sentences(note: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", note.strip())
    return [p for p in parts if p.strip()]


def _overlap(sentence: str, quote: str) -> int:
    return len(set(normalize(sentence).split()) & set(normalize(quote).split()))


def directional(note: str, quote: str) -> str | None:
    """Delete the sentence that carries the quote."""
    sents = sentences(note)
    if len(sents) < 2 or not quote.strip():
        return None
    scores = [_overlap(s, quote) for s in sents]
    index = max(range(len(sents)), key=lambda n: scores[n])
    if scores[index] < 2:
        return None
    return " ".join(s for n, s in enumerate(sents) if n != index)


def invariance(note: str, quote: str) -> str | None:
    """Move an unrelated sentence. The quote's sentence stays intact."""
    sents = sentences(note)
    if len(sents) < 2 or not quote.strip():
        return None
    others = [n for n, s in enumerate(sents) if _overlap(s, quote) < 2]
    if not others:
        return None
    index = others[0]
    moved = sents[:index] + sents[index + 1 :] + [sents[index]]
    if moved == sents:
        return None
    return " ".join(moved)
