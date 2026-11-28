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
    found = {name for name, pattern in _PHI_COMPILED if pattern.search(text)}
    if detect_names(text):
        found.add("person_name")
    return sorted(found)


# Phase 40: unlabelled names, which the identifier regexes above cannot reach.
# A clinical de-id model (obi/deid_bert_i2b2, BERT trained on i2b2 2014) finds
# person spans; context rules (honorific, relative, a note that opens on its
# subject) are a precise backstop. A model span or rule match followed by
# sign/syndrome/ring/stage is an eponym, not a person. Measured in
# data/reports/d1_name_refusal_findings.md: holdout recall 0.971, zero false
# positives on 224 cohort notes.
DEID_MODEL = "obi/deid_bert_i2b2"
# Chosen on holdout 1 (0.3 to 0.5 tie on false positives; 0.3 has best recall).
DEID_THRESHOLD = 0.3
_PERSON_LABELS = frozenset({"PATIENT", "STAFF"})
_NAME = r"[A-Z][a-z]+(?:[-'][A-Za-z]+)?"
_HONORIFIC = re.compile(rf"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof)\.?\s+({_NAME})")
_RELATION = re.compile(
    r"\b(?i:daughter|son|wife|husband|partner|mother|father|sister|brother|niece|nephew"
    r"|aunt|uncle|grand(?:son|daughter|mother|father)|friend|neighbou?r|caregiver|carer"
    rf"|fianc[eé]e?|named|called)\s*,?\s+({_NAME}(?:\s+{_NAME})?)"
)
_SUBJECT = re.compile(
    rf"(?:^|[.!?]\s+)\W*({_NAME}(?:\s+{_NAME})?)"
    r"(?:'s\b|,?\s+(?:is|was)\s+an?\s+\d|,\s+an?\s+\d)"
)
_EPONYM_NEXT = re.compile(
    r"(?:'s\b|’s\b)?['’]?\s*(?:sign|syndrome|disease|catheter|lymphoma|test|maneuver"
    r"|score|scale|criteria|classification|procedure|operation|tube|node|cell|phenomenon"
    r"|reflex|palsy|sarcoma|tumou?r|grade|repair|shunt|ulcer|granulomatosis|ring|stage)\b",
    re.I,
)
_REST_OF_WORD = re.compile(r"[\w'’-]*")
_FUNCTION_WORDS = frozenset({"He", "She", "They", "Who", "Is", "Was", "Has", "Had", "And", "The"})
_COMMON: frozenset[str] | None = None
_PIPELINE = None


def _common_words() -> frozenset[str]:
    from pathlib import Path

    global _COMMON
    if _COMMON is None:
        _COMMON = frozenset(Path(__file__).with_name("common_words.txt").read_text().split())
    return _COMMON


def _deid():
    """The de-id pipeline, loaded once. Missing is an error: the refusal must not degrade."""
    global _PIPELINE
    if _PIPELINE is None:
        from transformers import AutoTokenizer, pipeline

        # The model ships no max length; without one a long note overflows 512 positions.
        tokenizer = AutoTokenizer.from_pretrained(DEID_MODEL, model_max_length=512)
        _PIPELINE = pipeline("token-classification", model=DEID_MODEL, tokenizer=tokenizer,
                             aggregation_strategy="simple", stride=64)
    return _PIPELINE


def _uncommon(name: str) -> bool:
    common = _common_words()
    return any(t.lower() not in common for t in name.split())


def _eponym(text: str, end: int) -> bool:
    word = _REST_OF_WORD.match(text, end)
    return bool(_EPONYM_NEXT.match(text, word.end() if word else end))


def _rule_names(text: str) -> bool:
    for m in _HONORIFIC.finditer(text):
        if not _eponym(text, m.end(1)):
            return True
    for m in _RELATION.finditer(text):
        if m.group(1).split()[0] not in _FUNCTION_WORDS and not _eponym(text, m.end(1)):
            return True
    return any(
        _uncommon(m.group(1)) and not _eponym(text, m.end(1)) for m in _SUBJECT.finditer(text)
    )


def detect_names(text: str) -> bool:
    """True if the text names a person without a label."""
    if _rule_names(text):
        return True
    return any(
        e["entity_group"] in _PERSON_LABELS
        and e["score"] >= DEID_THRESHOLD
        and not _eponym(text, e["end"])
        for e in _deid()(text)
    )
