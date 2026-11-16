"""Deterministic badges for citations that do not establish their verdict.

Grounding already proved the quote is verbatim. These checks ask a narrower
question and never change a verdict: a flag is a badge, the same posture as
weak_absence. E1 is a ConText-style assertion scope, E2 is arithmetic on a
number taken from the note, E3 is conjunct coverage.
"""

from __future__ import annotations

import os
import re

from trialguard.verify.grounding import (
    absence_acronyms,
    absence_extra_terms,
    absence_terms,
    normalize,
)

_TERMINATOR = re.compile(r"\b(?:but|however|except)\b|[.;]")
_NEG = re.compile(
    r"\b(?:no|not|denies|denied|deny|without|negative for|absence of|absent|"
    r"ruled out|no evidence of|free of)\b"
)
_HYP = re.compile(r"\b(?:if|possible|possibly|suspected|suspicion of|rule out|differential)\b")
_HIST = re.compile(r"\b(?:history of|h/o|previously|prior history|in the past)\b")
_FAM = re.compile(
    r"\b(?:mother|father|sister|brother|family history|maternal|paternal|parent)\b"
)
_HISTORY_CRITERION = re.compile(r"\b(?:history|prior|previous)\b")
_FAMILY_CRITERION = re.compile(r"\bfamily\b")

_CMP = {
    ">=": lambda v, b: v >= b,
    ">": lambda v, b: v > b,
    "<=": lambda v, b: v <= b,
    "<": lambda v, b: v < b,
    "at least": lambda v, b: v >= b,
    "older than": lambda v, b: v > b,
    "under": lambda v, b: v < b,
    "younger than": lambda v, b: v < b,
}
_OP = r">=|<=|≥|≤|>|<|at least|older than|under|younger than"
_AGE_CMP = re.compile(rf"\bage\b[^.\n]{{0,40}}?({_OP})\s*(\d{{1,3}})", re.I)
_AGE_RANGE = re.compile(
    r"\bage[sd]?\b[^.\n]{0,40}?(\d{1,3})\s*(?:-|to|–)\s*(\d{1,3})", re.I
)
_LAB_CMP = re.compile(
    rf"\b(creatinine|bilirubin|platelets|anc|hemoglobin|ecog)\b[^.\n]{{0,40}}?"
    rf"({_OP})\s*(\d+(?:\.\d+)?)",
    re.I,
)
_ECOG_RANGE = re.compile(
    r"\becog\b[^.\n]{0,30}?(\d)\s*(?:-|to|or)\s*(\d)", re.I
)
_NOTE_AGE = re.compile(r"\b(\d{1,3})\s*-?\s*year-?old\b", re.I)
_NOTE_LAB = {
    "creatinine": re.compile(r"\bcreatinine(?:\s*(?:is|of|:))?\s*(\d+(?:\.\d+)?)", re.I),
    "bilirubin": re.compile(r"\bbilirubin(?:\s*(?:is|of|:))?\s*(\d+(?:\.\d+)?)", re.I),
    "platelets": re.compile(r"\bplatelets(?:\s*(?:is|of|:))?\s*(\d+(?:\.\d+)?)", re.I),
    "anc": re.compile(r"\banc(?:\s*(?:is|of|:))?\s*(\d+(?:\.\d+)?)", re.I),
    "hemoglobin": re.compile(r"\bhemoglobin(?:\s*(?:is|of|:))?\s*(\d+(?:\.\d+)?)", re.I),
    "ecog": re.compile(r"\becog(?:\s+performance\s+status)?(?:\s*(?:is|of|:))?\s*(\d)", re.I),
}


def badges_enabled() -> bool:
    """Stamp weak_entailment. Off until a check clears the pre-registered bar."""
    return os.environ.get("TG_WEAK_ENTAILMENT", "0") == "1"


def _bounded(term: str, haystack: str) -> bool:
    return f" {term} " in f" {haystack} "


def _norm_op(op: str) -> str:
    return {">=": ">=", "≥": ">=", "<=": "<=", "≤": "<=", ">": ">", "<": "<"}.get(
        op.lower(), op.lower()
    )


def _window(quote_norm: str, term: str) -> str:
    """Text between the concept and the previous clause break.

    The closest trigger is the one that governs the concept. A trigger before
    "but" does not.
    """
    at = quote_norm.find(f" {term} ")
    if at < 0:
        return ""
    head = quote_norm[:at]
    breaks = [m.end() for m in _TERMINATOR.finditer(head)]
    return head[breaks[-1] :] if breaks else head


def _scope(window: str) -> str:
    """Rightmost trigger wins: it is the one nearest the concept."""
    found: list[tuple[int, str]] = []
    for name, pattern in (
        ("negated", _NEG),
        ("hypothetical", _HYP),
        ("historical", _HIST),
        ("family", _FAM),
    ):
        last = None
        for match in pattern.finditer(window):
            last = match
        if last is not None:
            found.append((last.start(), name))
    if not found:
        return "affirmed"
    found.sort()
    return found[-1][1]


def context_conflict(criterion: str, quote: str, verdict: str) -> str | None:
    """E1. A scope that points the other way from the verdict, or None."""
    if verdict not in ("met", "not_met") or not quote.strip():
        return None
    quote_norm = normalize(quote)
    hay = f" {quote_norm} "
    terms = [t for t in absence_terms(criterion) if _bounded(t, hay)]
    if not terms:
        return None
    presence = verdict == "met"
    skip_hist = _HISTORY_CRITERION.search(criterion) is not None
    skip_fam = _FAMILY_CRITERION.search(criterion) is not None
    for term in terms:
        scope = _scope(_window(f" {quote_norm} ", term))
        if scope == "historical" and skip_hist:
            scope = "affirmed"
        if scope == "family" and skip_fam:
            scope = "affirmed"
        if presence and scope in ("negated", "hypothetical", "historical", "family"):
            return f"context_{scope}"
        if not presence and scope == "affirmed":
            return "context_affirmed"
        if not presence and scope == "historical" and not skip_hist:
            return "context_historical"
    return None


def _note_value(attr: str, note: str) -> float | None:
    if attr == "age":
        match = _NOTE_AGE.search(note)
        return float(match.group(1)) if match else None
    pattern = _NOTE_LAB.get(attr)
    if pattern is None:
        return None
    match = pattern.search(note)
    return float(match.group(1)) if match else None


def _criterion_holds(criterion: str, value: float, attr: str) -> bool | None:
    if attr == "age":
        ranged = _AGE_RANGE.search(criterion)
        if ranged:
            low, high = float(ranged.group(1)), float(ranged.group(2))
            return low <= value <= high
        compared = _AGE_CMP.search(criterion)
        if compared:
            return _CMP[_norm_op(compared.group(1))](value, float(compared.group(2)))
        return None
    if attr == "ecog":
        ranged = _ECOG_RANGE.search(criterion)
        if ranged:
            low, high = float(ranged.group(1)), float(ranged.group(2))
            return low <= value <= high
    compared = _LAB_CMP.search(criterion)
    if compared and compared.group(1).lower() == attr:
        return _CMP[_norm_op(compared.group(2))](value, float(compared.group(3)))
    return None


def numeric_conflict(criterion: str, verdict: str, patient_text: str) -> str | None:
    """E2. Arithmetic disagrees with the verdict, or None when it cannot tell."""
    if verdict not in ("met", "not_met"):
        return None
    attrs = ["age"]
    lab = _LAB_CMP.search(criterion)
    if lab:
        attrs.append(lab.group(1).lower())
    elif _ECOG_RANGE.search(criterion):
        attrs.append("ecog")
    elif not _AGE_CMP.search(criterion) and not _AGE_RANGE.search(criterion):
        return None
    for attr in attrs:
        value = _note_value(attr, patient_text)
        if value is None:
            continue
        holds = _criterion_holds(criterion, value, attr)
        if holds is None:
            continue
        if holds != (verdict == "met"):
            return f"numeric_{attr}"
    return None


def _conjuncts(criterion: str) -> list[str]:
    parts = re.split(r"\s*(?:;|\n)\s*|\s+\band\b\s+", criterion, flags=re.I)
    return [p.strip(" .") for p in parts if len(p.strip(" .")) >= 3]


def _overlaps(conjunct: str, quote: str) -> bool:
    hay = normalize(quote)
    terms = absence_terms(conjunct)
    extra = absence_extra_terms(conjunct) if absence_acronyms() else []
    return any(_bounded(t, hay) for t in terms + extra)


def uncovered_conjunct(criterion: str, quote: str, verdict: str) -> str | None:
    """E3. A met verdict whose quote misses a conjunct, or None."""
    if verdict != "met":
        return None
    usable = []
    for part in _conjuncts(criterion):
        terms = absence_terms(part)
        extra = absence_extra_terms(part) if absence_acronyms() else []
        if terms or extra:
            usable.append(part)
    if len(usable) < 2:
        return None
    for part in usable:
        if not _overlaps(part, quote):
            return "conjunct_uncovered"
    return None


def badge_reasons(
    criterion: str,
    quote: str,
    verdict: str,
    patient_text: str | None,
) -> list[str]:
    """Reasons a grounded decisive citation still fails one of E1-E3."""
    reasons = []
    context = context_conflict(criterion, quote, verdict)
    if context:
        reasons.append(context)
    if patient_text:
        numeric = numeric_conflict(criterion, verdict, patient_text)
        if numeric:
            reasons.append(numeric)
    conjunct = uncovered_conjunct(criterion, quote, verdict)
    if conjunct:
        reasons.append(conjunct)
    return reasons
