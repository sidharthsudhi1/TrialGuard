"""Phase 39: how often a surfaced trial requires a biomarker status the note contradicts.

Deterministic, like grounding. A status counts only when the text states it
(HER2-positive, ER/PR negative, KRAS wild-type, no EGFR mutation); a marker
named without a status is no status. A contradiction needs both sides explicit:
the note states one status and every mention in the trial's inclusion criteria
requires the other. A marker absent from the note is an unstated fact, which is
needs_review territory, never a contradiction. A text that states both statuses
for one marker (cohorts, "HER2-positive or HER2-negative", a rebiopsy) drops the
marker rather than guessing.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from trialguard.eval.significance import bootstrap_ci, ratio_of_sums

POS, NEG = "pos", "neg"

_MARKERS = {
    "HER2": r"(?i:her[\s-]?2(?:/neu)?|erbb2)",
    "ER": r"(?i:\ber\b|o?estrogen[\s-]receptors?)",
    "PR": r"(?i:\bpr\b|\bpgr\b|progesterone[\s-]receptors?)",
    "HR": r"(?i:\bhr\b|hormone[\s-]receptors?)",
    "EGFR": r"(?i:\begfr\b)",
    "ALK": r"\bALK\b",
    "ROS1": r"\bROS-?1\b",
    "KRAS": r"(?i:\bk-?ras\b)",
    "BRAF": r"(?i:\bbraf\b)",
    "BRCA": r"(?i:\bbrca[12]?(?:/2)?\b)",
}
_RECEPTORS = ("HER2", "ER", "PR", "HR")

_ANY_MARKER = "|".join(f"(?:{p})" for p in _MARKERS.values())
_GROUP = rf"(?:{_ANY_MARKER})(?:\s*(?:/|,|&|\band\b)\s*(?:{_ANY_MARKER}))*"

_SIGN = r"(?P<either>\+/-|-/\+|±)|(?P<sign>\+|[-\u2212\u02d7](?![\w]))"
_POS_WORDS = (
    r"positive|pos\b|over-?express\w*|amplified|amplification|"
    r"mutations?|mutated|mutant|rearrange\w*|fusions?|translocations?|alterations?"
)
_NEG_WORDS = (
    r"negative|neg\b|non-?amplified|not\s+amplified|"
    r"wild[\s-]?type|wt\b|non-?mutated|unmutated"
)
_SEP = r"[\s:]*(?:status\s*)?(?:(?:is|was)\s+)?[\s:-]*"
_VARIANT = r"(?:\s+[A-Z]\d{2,4}[A-Z])?"

_FORWARD = re.compile(
    rf"(?P<group>{_GROUP}){_VARIANT}(?:{_SEP}(?P<neg>{_NEG_WORDS})|{_SEP}(?P<pos>{_POS_WORDS})|{_SIGN})",
    re.I,
)
_PREFIX = re.compile(
    rf"(?P<lead>\bno\b|\bwithout\b|\babsence\s+of\b|\black\s+of\b|"
    rf"\bnegative\s+for\b|\bpositive\s+for\b)\s+(?:(?:an?|any|known)\s+)*(?P<group>{_GROUP})",
    re.I,
)
_HR_POS = r"(?:positive|pos\b|\+)"
# "ER-positive and/or PR-positive" asks for hormone-receptor positive disease,
# not for both receptors; read per marker it would require each.
_DISJUNCTION = re.compile(
    rf"(?P<a>{_MARKERS['ER']}|{_MARKERS['PR']})(?:{_SEP}{_HR_POS})?"
    rf"\s*(?:and\s*/\s*or|or)\s*"
    rf"(?P<b>{_MARKERS['ER']}|{_MARKERS['PR']}){_SEP}{_HR_POS}",
)
_TRIPLE = re.compile(r"(?i:triple[\s-]*(?P<t>negative|positive))|(?P<tnbc>\bTNBC\b)")


def _markers_in(group: str) -> list[str]:
    return [m for m, p in _MARKERS.items() if re.search(p, group)]


def _votes(text: str) -> dict[str, set[str]]:
    votes: dict[str, set[str]] = {}

    def vote(markers, status):
        for m in markers:
            votes.setdefault(m, set()).add(status)

    masked = list(text)
    for m in _DISJUNCTION.finditer(text):
        if set(_markers_in(m.group("a")) + _markers_in(m.group("b"))) == {"ER", "PR"}:
            vote(["HR"], POS)
            masked[m.start():m.end()] = " " * (m.end() - m.start())
    text = "".join(masked)
    for m in _PREFIX.finditer(text):
        lead = m.group("lead").lower()
        if lead.startswith("positive"):
            status = POS
        elif lead.startswith("negative"):
            status = NEG
        else:
            # "no HER2" alone is not a status; "no HER2 amplification" is.
            tail = text[m.end():m.end() + 30]
            if not re.match(rf"[\s-]*(?:{_POS_WORDS})", tail, re.I):
                continue
            status = NEG
        vote(_markers_in(m.group("group")), status)
        trail = re.match(rf"[\s-]*(?:{_POS_WORDS})?", text[m.end():], re.I)
        end = m.end() + (trail.end() if trail else 0)
        masked[m.start():end] = " " * (end - m.start())
    rest = "".join(masked)
    for m in _FORWARD.finditer(rest):
        markers = _markers_in(m.group("group"))
        if m.group("either"):
            vote(markers, POS)
            vote(markers, NEG)
            continue
        status = NEG if m.group("neg") or m.group("sign") not in (None, "+") else POS
        if m.group("pos") and re.match(r"(?i:mut|rearr|fusion|transl|alter)", m.group("pos")):
            markers = [x for x in markers if x not in _RECEPTORS]
        vote(markers, status)
    for m in _TRIPLE.finditer(text):
        status = NEG if m.group("tnbc") or m.group("t").lower() == "negative" else POS
        vote(("ER", "PR", "HER2"), status)
    return votes


def _close(statuses: dict[str, str], blocked: set[str]) -> dict[str, str]:
    implied: dict[str, set[str]] = {m: {s} for m, s in statuses.items()}

    def imply(marker, status):
        if marker not in blocked:
            implied.setdefault(marker, set()).add(status)

    if POS in (statuses.get("ER"), statuses.get("PR")):
        imply("HR", POS)
    if statuses.get("ER") == NEG and statuses.get("PR") == NEG:
        imply("HR", NEG)
    if statuses.get("HR") == NEG:
        imply("ER", NEG)
        imply("PR", NEG)
    return {m: s.pop() for m, s in implied.items() if len(s) == 1}


def note_statuses(text: str) -> dict[str, str]:
    """Explicitly stated biomarker statuses, closed over the receptor implications."""
    votes = _votes(text)
    # A text that allows HR+ or HR- (often "HR+ ... or TNBC") allows either
    # status of each receptor too.
    if votes.get("HR") == {POS, NEG}:
        for m in ("ER", "PR"):
            votes[m] = {POS, NEG}
    blocked = {m for m, s in votes.items() if len(s) > 1}
    return _close({m: next(iter(s)) for m, s in votes.items() if len(s) == 1}, blocked)


_INCL = re.compile(r"inclusion\s+criteria\s*:", re.I)
_EXCL = re.compile(r"exclusion\s+criteria\s*:", re.I)


def inclusion_section(text: str) -> str:
    """The inclusion criteria: after the last inclusion header, before exclusion."""
    heads = list(_INCL.finditer(text))
    start = heads[-1].end() if heads else 0
    end = _EXCL.search(text, start)
    return text[start:end.start() if end else len(text)]


def trial_requirements(text: str) -> dict[str, str]:
    """Statuses the inclusion criteria require; a marker allowed both ways requires none."""
    return note_statuses(inclusion_section(text))


def contradictions(note: dict[str, str], required: dict[str, str]) -> list[str]:
    return sorted(m for m in note.keys() & required.keys() if note[m] != required[m])


_FLIP_WORD = {
    "positive": "negative", "negative": "positive", "pos": "neg", "neg": "pos",
    "+": "-", "-": "+", "\u2212": "+", "\u02d7": "+",
    "amplified": "not amplified", "not amplified": "amplified",
    "wild-type": "mutant", "wild type": "mutant", "wildtype": "mutant",
    "mutant": "wild-type", "mutated": "wild-type", "mutation": "wild-type",
}


def _match_case(word: str, like: str) -> str:
    return word.upper() if like.isupper() and len(like) > 1 else word


def flip_text(text: str) -> str:
    """Invert every explicitly stated status, leaving the rest of the text alone."""
    edits: list[tuple[int, int, str]] = []
    for m in _FORWARD.finditer(text):
        name = next((g for g in ("neg", "pos", "sign") if m.group(g)), None)
        if name is None:
            continue
        word = m.group(name)
        key = re.sub(r"\s+", " ", word.lower())
        if key in _FLIP_WORD:
            edits.append((m.start(name), m.end(name), _match_case(_FLIP_WORD[key], word)))
    for m in _TRIPLE.finditer(text):
        if m.group("t"):
            word = m.group("t")
            edits.append((m.start("t"), m.end("t"), _match_case(_FLIP_WORD[word.lower()], word)))
    out, last = [], 0
    for start, end, new in sorted(edits):
        if start < last:
            continue
        out.append(text[last:start])
        out.append(new)
        last = end
    out.append(text[last:])
    return "".join(out)


def flip_keywords(keywords: list[str], flipped_note: str) -> tuple[list[str], list[str]]:
    """Flip each keyword; drop any the flipped note contradicts (an implied HR that stayed)."""
    stated = note_statuses(flipped_note)
    kept: list[str] = []
    dropped: list[str] = []
    for kw in map(flip_text, keywords):
        (dropped if contradictions(stated, note_statuses(kw)) else kept).append(kw)
    return kept, dropped


def score(rows: list[dict], trials: dict[str, str], k: int = 25) -> dict:
    """Pooled contradiction rate over each patient's top-k, with patient-bootstrap CIs."""
    per_patient = []
    for r in rows:
        stated = note_statuses(r["note"])
        ranked = r.get("retrieved")
        if ranked is None and stated:
            raise ValueError(f"{r['patient_id']} states {sorted(stated)} but has no ranking")
        surfaced = ranked[:k] if ranked is not None else [None] * k
        hit = [
            n for n in surfaced
            if n is not None and contradictions(stated, trial_requirements(trials[n]))
        ]
        per_patient.append({
            "patient_id": r["patient_id"],
            "stated": stated,
            "retrieved": ranked is not None,
            "surfaced": len(surfaced),
            "contradicted": len(hit),
            "contradicted_ids": hit,
        })
    num = [p["contradicted"] for p in per_patient]
    den = [p["surfaced"] for p in per_patient]
    marked = [p for p in per_patient if p["stated"]]
    return {
        "k": k,
        "patients": len(per_patient),
        "patients_with_marker": len(marked),
        "surfaced": sum(den),
        "contradicted": sum(num),
        "rate": bootstrap_ci(ratio_of_sums(num, den), len(per_patient)),
        "rate_given_marker": bootstrap_ci(
            ratio_of_sums([p["contradicted"] for p in marked], [p["surfaced"] for p in marked]),
            len(marked),
        ),
        "per_patient": per_patient,
    }


def _wilson(x: int, n: int, z: float = 1.96) -> list[float | None]:
    if not n:
        return [None, None]
    p = x / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def observational_share(nct_ids, study_types: dict[str, str | None]) -> dict:
    """Share of trials (unique ids) CT.gov types as OBSERVATIONAL, Wilson 95% CI."""
    ids = sorted(set(nct_ids))
    known = [study_types[n] for n in ids if study_types.get(n)]
    obs = sum(1 for t in known if t == "OBSERVATIONAL")
    return {
        "n": len(known),
        "observational": obs,
        "share": round(obs / len(known), 4) if known else None,
        "ci95": _wilson(obs, len(known)),
        "missing": len(ids) - len(known),
        "by_type": dict(Counter(known)),
    }
