"""Input hardening for untrusted patient notes (OWASP LLM01: prompt injection).

The patient note is attacker-controllable free text that is fed into the analyst
prompt. Two defenses live here:

- `detect_injection`: flags notes that carry instruction-injection signatures, so a
  suspicious note can be quarantined or surfaced rather than trusted silently.
- `fence`: wraps the note in explicit delimiters and a data-only label for the
  hardened prompt (analyst v3), so the model is told to treat it as data.

These are defense-in-depth. The load-bearing guarantee is still deterministic
grounding (verify/grounding.py): a verdict cannot stand without a verbatim quote
from a provided source, so a compromised analyst cannot fabricate a grounded
verdict out of nothing. Note that the patient note is itself a grounding source,
so grounding alone does not stop an attacker who plants fake evidence text in the
note — that is exactly the residual risk `detect_injection` covers.
"""

from __future__ import annotations

import re

_NOTE_OPEN = "<patient_note>"
_NOTE_CLOSE = "</patient_note>"

# Instruction-injection signatures. Deliberately conservative: these phrases have
# no legitimate reason to appear in a clinical narrative, so a match is a strong
# signal without flagging ordinary notes.
_INJECTION_PATTERNS = [
    r"ignore (all |the |your )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |the )?(previous|prior|above) ",
    r"you are now\b",
    r"new (instructions|task|role)\s*:",
    r"system\s*(prompt|message)?\s*:",
    r"</?(patient_note|system|assistant|user)>",  # tag-smuggling / fence breakout
    r"mark (all |every )?(criteria|criterion)\b.*\b(met|eligible)",
    r"output .*(eligible|all met)",
    r"pretend (to be|you are)\b",
    r"reveal (your |the )?(system )?(prompt|instructions)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def detect_injection(text: str) -> bool:
    """True if the text carries a known prompt-injection signature."""
    return any(p.search(text) for p in _COMPILED)


def fence(note: str) -> str:
    """Wrap an untrusted note in delimiters, stripping any smuggled fence tags so
    the model cannot close the block early and escape the data context."""
    cleaned = note.replace(_NOTE_OPEN, "").replace(_NOTE_CLOSE, "")
    return f"{_NOTE_OPEN}\n{cleaned}\n{_NOTE_CLOSE}"


# HIPAA Safe Harbor identifiers, as far as a regex can honestly reach. This is a
# *refusal* gate, not a de-identifier: the project's posture is synthetic notes
# only, and quietly redacting real PHI would mean accepting it, processing it,
# and logging a modified version of it. Refusing says what the system actually
# requires.
#
# Deliberately high-precision. Clinical narrative is full of numbers that look
# like identifiers — ages, ECOG scores, lab values, doses, NCT IDs — and a gate
# that blocks legitimate synthetic notes would be turned off within a day, which
# is worse than no gate. Every pattern below is validated against all 308 cohort
# notes for zero false positives (tests/test_phi.py).
_PHI_PATTERNS: list[tuple[str, str]] = [
    ("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("email", r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    # Labelled record numbers only. A bare integer is a lab value far more often
    # than it is an MRN.
    (
        "record_number",
        r"\b(?:mrn|medical record (?:number|no\.?)|patient id|chart (?:number|no\.?))"
        r"\b\s*[:#]?\s*\w*\d\w*",
    ),
    # A DOB label must be followed by something date-shaped. "born on 39th week"
    # is a gestational age and appears verbatim in TREC 2022; accepting any
    # following token flagged it as a birth date.
    (
        "date_of_birth",
        r"\b(?:dob|date of birth|born on)\b\s*[:\-]?\s*"
        r"(?:\d{1,4}[/-]\d{1,2}[/-]\d{1,4}"
        r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s*(?:19|20)\d{2}"
        r"|(?:19|20)\d{2})",
    ),
    # Full calendar date with a day component. Bare years appear constantly in
    # trial and treatment history and are not identifiers on their own.
    ("full_date", r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b"),
    ("phone", r"(?:\b|\+)\d{0,2}[\s.-]?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"),
    (
        "street_address",
        r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b",
    ),
    ("ip_address", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("url", r"https?://\S+"),
    ("named_patient", r"\b(?:patient(?:'s)? name|full name)\b\s*[:\-]\s*[A-Z]"),
]

_PHI_COMPILED = [(name, re.compile(p, re.IGNORECASE)) for name, p in _PHI_PATTERNS]


def detect_phi(text: str) -> list[str]:
    """Names of the PHI identifier classes present, never the matched text.

    Returning categories rather than spans is the point: the caller needs to tell
    the user what to remove, and an error message that echoed the match would
    write the PHI into exactly the logs this gate exists to keep clean.
    """
    return sorted({name for name, pattern in _PHI_COMPILED if pattern.search(text)})
