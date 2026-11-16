"""E4 score mapping. Thresholds and the kill bar are fixed before any model runs."""

from __future__ import annotations

import re

THRESHOLDS = tuple(i / 20 for i in range(1, 20))
_BOUNDARY = re.compile(r"[.!?]\s+|\n+")


def sentence_containing(note: str, quote: str) -> str | None:
    """The note sentence that holds `quote`, punctuation included. None if absent."""
    if not quote or quote not in note:
        return None
    start = note.find(quote)
    end = start + len(quote)
    left = 0
    for boundary in _BOUNDARY.finditer(note, 0, start):
        left = boundary.end()
    boundary_after = _BOUNDARY.search(note, end)
    if boundary_after is None:
        right = len(note)
    elif boundary_after.group(0)[0] in ".!?":
        right = boundary_after.start() + 1
    else:
        right = boundary_after.start()
    sentence = note[left:right].strip()
    return sentence or None


def entail_from_tokens(tokens: list) -> float:
    """1 minus the strongest hallucination-token probability. No such token is 1."""
    hallu = 0.0
    for tok in tokens:
        if not isinstance(tok, dict):
            continue
        pred = tok.get("pred", tok.get("label"))
        text = str(pred).lower()
        is_hallu = pred in (1, "1", True) or "hallu" in text
        if not is_hallu:
            continue
        prob = tok.get("prob", tok.get("confidence"))
        hallu = max(hallu, 1.0 if prob is None else float(prob))
    return round(1.0 - min(hallu, 1.0), 4)


def clears_kill(operating_points: list[dict]) -> bool:
    """Human-set bar: at most 2 sound citations damaged, and at least 25% of unsound caught."""
    for row in operating_points:
        recall = row.get("recall") or 0.0
        if row["sound_citations_destroyed"] <= 2 and recall >= 0.25:
            return True
    return False
