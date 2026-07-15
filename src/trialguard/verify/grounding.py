"""Deterministic citation grounding — the faithfulness mechanism.

An Analyst verdict is only allowed to stand if its supporting quote exists
verbatim (modulo whitespace/case/punctuation) in the source trial text. This
check is pure Python: it cannot hallucinate agreement, cannot be fooled by a
confident model, and costs nothing. It is the floor under every GROUNDED stamp.

The LLM (or a local NLI model) may still be used afterward to judge *entailment*
— does the grounded quote actually support the verdict — but existence is settled
here, mechanically, first.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used on both sides."""
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def is_grounded(quote: str, source_text: str, min_tokens: int = 2) -> bool:
    """True iff the normalized quote is a verbatim substring of the source AND
    carries at least min_tokens words.

    A token guard (not a char-length guard): it rejects vague single-word quotes
    ("ECOG", "cancer") that match spuriously, while accepting specific short
    clinical facts ("48 M", "EF was 25%", "T-L spine"). A char-length guard
    rejected those real atomic facts — the high-value evidence in eligibility
    matching — which inflated the apparent hallucination rate on corpora with
    terse patient text (TREC).
    """
    q = normalize(quote)
    if len(q.split()) < min_tokens:
        return False
    return q in normalize(source_text)


def ground_assessments(
    assessments: list[dict],
    source_text: str,
) -> list[dict]:
    """Stamp each assessment with grounding status.

    Each assessment: {criterion, verdict, quote, ...}. Adds:
      - grounded: bool
      - verdict: forced to "unverifiable" when not grounded and a claim was made
    A verdict of "cannot_determine" with no quote is left as-is (honest abstention).
    """
    out = []
    for a in assessments:
        quote = a.get("quote", "") or ""
        verdict = a.get("verdict", "cannot_determine")
        grounded = is_grounded(quote, source_text)
        result = {**a, "grounded": grounded}
        if verdict in ("met", "not_met") and not grounded:
            result["verdict"] = "unverifiable"
            result["grounding_failure"] = True
        out.append(result)
    return out
