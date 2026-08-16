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

# Boilerplate that appears in almost every criterion and carries no patient-
# specific meaning. Absence checking ignores these: "study"/"patient" missing
# from a note says nothing about whether the patient matches a disqualifier.
_ABSENCE_STOPWORDS = frozenset(
    """
    patient patients subject subjects study studies trial trials protocol
    history evidence known current currently prior previous within other
    another more than with without have has had having been being must will
    that this these those they their them from into during course also any
    all who whom which what when where such same able unable likely
    signs symptoms diagnosis diagnosed treatment treated therapy receiving
    receive received participation participate participating enrollment
    enrolled investigational agent agents drug drugs device devices
    condition conditions disease diseases disorder disorders illness
    abnormality abnormalities problem problems finding findings
    criteria criterion requirement requirements unsuitable ineligible
    investigator judgment opinion considered deemed
    """.split()
)


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


def absence_terms(criterion: str) -> list[str]:
    """Distinctive clinical terms of a criterion, used for absence checking."""
    return [
        t
        for t in normalize(criterion).split()
        if len(t) >= 4 and t not in _ABSENCE_STOPWORDS and not t.isdigit()
    ]


def is_absence_grounded(criterion: str, patient_text: str) -> bool:
    """True iff the criterion's distinctive terms are genuinely absent from the note.

    An exclusion criterion answered "not_met" claims the patient does NOT match a
    disqualifier — a statement about absence of evidence, which no verbatim span
    can support. Rather than exempt it from verification (a hallucination
    loophole), verify the complementary fact mechanically: the terms that would
    have to appear for the patient to match are not in the note.

    Checked against the patient note only. The trial text necessarily restates the
    criterion, so including it would make every absence claim look contradicted.
    A criterion with no distinctive terms is unverifiable this way and returns
    False, falling back to the verbatim requirement.
    """
    terms = absence_terms(criterion)
    if not terms:
        return False
    haystack = normalize(patient_text)
    return not any(t in haystack for t in terms)


def ground_assessments(
    assessments: list[dict],
    source_text: str,
    min_tokens: int = 2,
    patient_text: str | None = None,
) -> list[dict]:
    """Stamp each assessment with grounding status.

    Each assessment: {criterion, verdict, quote, kind?, ...}. Adds:
      - grounded: bool
      - grounded_by: "quote" | "absence" when grounded
      - verdict: forced to "unverifiable" when not grounded and a claim was made
    A verdict of "cannot_determine" with no quote is left as-is (honest abstention).
    min_tokens is the strictness knob swept by the coverage/faithfulness curve.

    Exclusion criteria answered "not_met" assert absence and are verified by
    is_absence_grounded instead of a verbatim span; every other verdict keeps the
    verbatim requirement unchanged. `patient_text` is required for that path —
    without it the behavior is exactly as before, so callers that ground against a
    single combined source are unaffected.
    """
    out = []
    for a in assessments:
        quote = a.get("quote", "") or ""
        verdict = a.get("verdict", "cannot_determine")
        grounded = is_grounded(quote, source_text, min_tokens=min_tokens)
        grounded_by = "quote" if grounded else None
        if (
            not grounded
            and verdict == "not_met"
            and a.get("kind") == "exclusion"
            and patient_text is not None
            and is_absence_grounded(str(a.get("criterion", "")), patient_text)
        ):
            grounded = True
            grounded_by = "absence"
        result = {**a, "grounded": grounded}
        if grounded_by:
            result["grounded_by"] = grounded_by
        if verdict in ("met", "not_met") and not grounded:
            result["verdict"] = "unverifiable"
            result["grounding_failure"] = True
        out.append(result)
    return out
