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
