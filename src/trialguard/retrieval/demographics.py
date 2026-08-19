"""Deterministic demographic gating for retrieved candidates.

Age and sex are structured fields on every trial and are stated in the opening
clause of essentially every clinical note (measured: 98% and 97% of the SIGIR
cohort). Matching a 58-year-old woman against a paediatric or male-only trial and
then spending a 29-second analyst call to discover it is work a comparison
operator can do for nothing.

The rule throughout is that only an explicit contradiction excludes. A missing
age, an unparseable note, a trial with no stated limit — all pass. A null is not
a negative, and ClinicalTrials.gov populates these fields inconsistently, so a
filter that treated absence as disqualifying would drop eligible trials silently.
That failure is worse than the waste it prevents: a wrongly assessed trial is
visible in the output, a wrongly filtered one is not.

No LLM involved. A filter that cost a model call would inherit the model's errors
and its latency, which is the opposite of the point.
"""

from __future__ import annotations

import re

_AGE = re.compile(
    r"\b(\d{1,3})\s*[-–]?\s*(?:year|yr)s?\s*[-–]?\s*old\b"
    r"|\b(\d{1,3})\s*(?:yo|y/o|y\.o\.)\b",
    re.IGNORECASE,
)
_AGE_MONTHS = re.compile(r"\b(\d{1,3})\s*[-–]?\s*months?\s*[-–]?\s*old\b", re.IGNORECASE)
_AGE_WEEKS = re.compile(r"\b(\d{1,3})\s*[-–]?\s*(?:week|day)s?\s*[-–]?\s*old\b", re.IGNORECASE)

_FEMALE = re.compile(r"\b(?:female|woman|women|girl|lady)\b", re.IGNORECASE)
_MALE = re.compile(r"\b(?:male|man|men|boy|gentleman)\b", re.IGNORECASE)

# "1 Year", "6 Months", "18 Years", "N/A"
_LIMIT = re.compile(r"(\d+(?:\.\d+)?)\s*(year|month|week|day|hour|minute)s?", re.IGNORECASE)
_PER_YEAR = {
    "year": 1.0,
    "month": 1 / 12,
    "week": 1 / 52.18,
    "day": 1 / 365.25,
    "hour": 1 / 8766,
    "minute": 1 / 525960,
}


def parse_age(note: str) -> float | None:
    """Age in years, or None when the note does not state one unambiguously."""
    m = _AGE.search(note)
    if m:
        return float(m.group(1) or m.group(2))
    m = _AGE_MONTHS.search(note)
    if m:
        return int(m.group(1)) / 12
    m = _AGE_WEEKS.search(note)
    if m:
        # Neonates: the exact value does not matter, only that it is far below any
        # adult lower bound.
        return 0.0
    return None


def parse_sex(note: str) -> str | None:
    """"male" / "female" / None. Ambiguous notes return None and are not gated.

    Only the first mention counts. A note saying "58-year-old woman ... her father
    had" would otherwise read as both, and both means unknown.
    """
    f, m = _FEMALE.search(note), _MALE.search(note)
    if f and m:
        return "female" if f.start() < m.start() else "male"
    if f:
        return "female"
    if m:
        return "male"
    return None


def parse_patient(note: str) -> dict:
    return {"age": parse_age(note), "sex": parse_sex(note)}


def _limit_years(value: str | None) -> float | None:
    if not value:
        return None
    m = _LIMIT.search(str(value))
    if not m:
        return None
    return float(m.group(1)) * _PER_YEAR[m.group(2).lower()]


def exclusion_reason(trial: dict, patient: dict) -> str | None:
    """Why this trial cannot enrol this patient, or None if it might.

    Returns a reason string rather than a bool so a filter can be audited: a
    dropped candidate should always be able to say what dropped it.
    """
    age = patient.get("age")
    if age is not None:
        lo = _limit_years(trial.get("min_age"))
        hi = _limit_years(trial.get("max_age"))
        if lo is not None and age < lo:
            return f"age {age:g} below min {trial.get('min_age')}"
        if hi is not None and age > hi:
            return f"age {age:g} above max {trial.get('max_age')}"

    sex = patient.get("sex")
    trial_sex = (trial.get("sex") or "").strip().upper()
    if sex and trial_sex in ("MALE", "FEMALE") and trial_sex != sex.upper():
        return f"trial enrols {trial_sex.lower()} only"

    return None


def filter_candidates(
    hits: list[tuple[str, float]],
    trials: dict[str, dict],
    patient: dict,
) -> tuple[list[tuple[str, float]], dict[str, str]]:
    """Drop candidates a hard demographic gate rules out.

    A candidate with no metadata row is kept: the filter is only ever allowed to
    act on evidence it actually has.
    """
    if patient.get("age") is None and not patient.get("sex"):
        return hits, {}
    kept, dropped = [], {}
    for nct, score in hits:
        trial = trials.get(nct)
        reason = exclusion_reason(trial, patient) if trial else None
        if reason:
            dropped[nct] = reason
        else:
            kept.append((nct, score))
    return kept, dropped
