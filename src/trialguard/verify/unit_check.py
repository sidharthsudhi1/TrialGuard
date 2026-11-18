"""E6: does a decisive verdict contradict the number in its own quote, after unit conversion.

Deterministic. Raises a badge reason only; never changes a verdict.
"""

from __future__ import annotations

import re

_OPS = {
    ">=": ">=", "=>": ">=", "at least": ">=", "greater than or equal to": ">=",
    "more than or equal to": ">=", "not less than": ">=",
    ">": ">", "greater than": ">", "more than": ">", "over": ">", "above": ">",
    "<=": "<=", "=<": "<=", "less than or equal to": "<=", "no more than": "<=",
    "not more than": "<=", "not greater than": "<=", "up to": "<=",
    "<": "<", "less than": "<", "under": "<", "below": "<",
}  # fmt: skip
_OP = "|".join(sorted((re.escape(k) for k in _OPS), key=len, reverse=True))
_NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
_PER_UL = r"(?:cells\s?)?(?:/|per\s)(?:ul|mm3|mm\^3|mcl|cubic\smillimeters?)"
_PER_L9 = r"(?:x|\*)\s?10\^?9\s?/\s?l|(?:x|\*)\s?109\s?/\s?l"
_PER_L3 = r"k\s?/\s?ul|(?:x|\*)\s?10\^?3\s?/\s?(?:ul|mm3)|(?:x|\*)\s?103\s?/\s?(?:ul|mm3)"
# Canonical units: mg/dL, g/dL, platelets x10^9/L, ANC cells/uL.
_LABS: dict[str, tuple[str, dict[str, float]]] = {
    "creatinine": (
        r"(?:serum\s)?creatinine(?!\sclearance)",
        {r"mg\s?/\s?dl": 1.0, r"(?:umol|micromol)\s?/\s?l": 1 / 88.4},
    ),
    "bilirubin": (
        r"(?:serum\s)?(?:total\s)?bilirubin",
        {r"mg\s?/\s?dl": 1.0, r"(?:umol|micromol)\s?/\s?l": 1 / 17.1},
    ),
    "hemoglobin": (
        r"(?:hemoglobin|haemoglobin|hgb|hb)(?!\s?a1c)",
        {r"g\s?/\s?dl": 1.0, r"g\s?/\s?l": 0.1},
    ),
    "platelets": (
        r"(?:platelets?(?:\scount)?|plt)",
        {_PER_L9: 1.0, _PER_L3: 1.0, _PER_UL: 0.001},
    ),
    "anc": (
        r"(?:anc|absolute\sneutrophil\scount|absolute\sneutrophils)",
        {_PER_UL: 1.0, _PER_L9: 1000.0, _PER_L3: 1000.0},
    ),
}
_SKIP = re.compile(r"\bor\b|\buln\b|upper limit|clearance|\bunless\b|\bexcept\b|\bif\b|a1c")
_OR_OK = re.compile(
    r"\bor\s(?:older|above|over|younger|equal|more|less|greater)\b|\d\s?,?\s?or\s\d"
)
_YEARS = r"\s?(?:years?|yrs?)"
_AGE_RANGE = re.compile(
    rf"\bage[sd]?\b\s?:?\s?(?:between\s|of\s)?(\d{{1,3}})"
    rf"\s?(?:-|to|and)\s?(\d{{1,3}}){_YEARS}"
)
_AGE_OP = re.compile(
    rf"\bage[sd]?\b\s?:?\s?(?:of\s|is\s)?({_OP})\s?(\d{{1,3}})"
    rf"(?:{_YEARS})?(?!\s?(?:months?|weeks?|days?))"
)
_AGE_TAIL = re.compile(
    rf"(\d{{1,3}}){_YEARS}(?:\sof\sage)?\s?(?:or|and)\s(older|above|over|younger)"
)
_ECOG = r"(?:ecog|zubrod)(?:\sperformance\sstatus|\sps)?(?:\s\(ps\))?(?:\sscore)?(?:\sof)?\s?:?\s?"
_ECOG_SET = re.compile(rf"{_ECOG}(\d)(?:\s?,\s?(\d))*\s?,?\s?or\s(\d)(?!\d)")
_ECOG_RANGE = re.compile(rf"{_ECOG}(\d)\s?(?:-|to)\s?(\d)(?!\d)")
_ECOG_OP = re.compile(rf"{_ECOG}({_OP})\s?(\d)(?!\d)")
_Q_AGE = re.compile(r"\b(\d{1,3})\s?-?\s?(?:year|yr)s?\s?-?\s?old\b")
_Q_ECOG = re.compile(
    r"(?:ecog|zubrod|performance\sstatus)(?:\sperformance\sstatus|\sps)?"
    r"(?:\sscore)?(?:\sof|\sis|\swas|:)?\s?(\d)(?![\d.]|\s?-\s?\d)"
)
_ANY_LAB = "|".join(name for name, _ in _LABS.values())  # fmt: skip


def _norm(text: str) -> str:
    text = text.lower()
    for a, b in (
        ("≥", ">="), ("≤", "<="), ("µ", "u"), ("μ", "u"),
        ("⁹", "9"), ("×", "x"), ("–", "-"), (" ", " "),
        ("³", "3"),
    ):  # fmt: skip
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text)


def _num(raw: str) -> float:
    return float(raw.replace(",", ""))


def _holds(clause: dict, value: float) -> bool:
    if "allowed" in clause:
        return value in clause["allowed"]
    if "low" in clause:
        return clause["low"] <= value <= clause["high"]
    b = clause["bound"]
    return {">=": value >= b, ">": value > b, "<=": value <= b, "<": value < b}[clause["op"]]


def criterion_clauses(criterion: str) -> list[dict] | None:
    """Numeric clauses in canonical units. None when the criterion is out of scope."""
    text = _norm(criterion)
    if _SKIP.search(_OR_OK.sub(" ", text)):
        return None
    clauses: list[dict] = []
    for m in _AGE_RANGE.finditer(text):
        clauses.append({"attr": "age", "low": float(m[1]), "high": float(m[2])})
    if not clauses:
        for m in _AGE_TAIL.finditer(text):
            op = "<=" if m[2] == "younger" else ">="
            clauses.append({"attr": "age", "op": op, "bound": float(m[1])})
    if not clauses:
        for m in _AGE_OP.finditer(text):
            clauses.append({"attr": "age", "op": _OPS[m[1]], "bound": float(m[2])})
    ecog: list[dict] = [
        {"attr": "ecog", "allowed": sorted({float(d) for d in m.groups() if d})}
        for m in _ECOG_SET.finditer(text)
    ]
    ecog = ecog or [
        {"attr": "ecog", "low": float(m[1]), "high": float(m[2])}
        for m in _ECOG_RANGE.finditer(text)
    ]
    ecog = ecog or [
        {"attr": "ecog", "op": _OPS[m[1]], "bound": float(m[2])} for m in _ECOG_OP.finditer(text)
    ]
    clauses.extend(ecog)
    for attr, (name, units) in _LABS.items():
        for unit, scale in units.items():
            pattern = rf"\b{name}\b[^0-9<>=;]{{0,25}}?({_OP})\s?{_NUM}\s?(?:{unit})"
            for m in re.finditer(pattern, text):
                clauses.append({"attr": attr, "op": _OPS[m[1]], "bound": _num(m[2]) * scale})
    return clauses or None


def quote_values(quote: str) -> dict[str, float]:
    """One value per attribute, only when the quote states it once with a unit."""
    text = _norm(quote)
    found: dict[str, list[float]] = {}
    for m in _Q_AGE.finditer(text):
        found.setdefault("age", []).append(float(m[1]))
    for m in _Q_ECOG.finditer(text):
        found.setdefault("ecog", []).append(float(m[1]))
    for attr, (name, units) in _LABS.items():
        for unit, scale in units.items():
            pattern = rf"\b{name}\b((?:(?!{_ANY_LAB})[^0-9<>=;]){{0,20}}?){_NUM}\s?(?:{unit})"
            for m in re.finditer(pattern, text):
                found.setdefault(attr, []).append(_num(m[2]) * scale)
    return {a: v[0] for a, v in found.items() if len(set(v)) == 1}


def numeric_conflict(criterion: str, verdict: str, quote: str) -> str | None:
    if verdict not in ("met", "not_met") or not quote:
        return None
    clauses = criterion_clauses(criterion)
    if not clauses:
        return None
    values = quote_values(quote)
    judged = [(c, _holds(c, values[c["attr"]])) for c in clauses if c["attr"] in values]
    if not judged:
        return None
    if len(clauses) == 1:
        clause, holds = judged[0]
        if holds != (verdict == "met"):
            return f"{clause['attr']} {values[clause['attr']]:g} vs criterion: verdict {verdict}"
        return None
    if verdict == "met":
        for clause, holds in judged:
            if not holds:
                return f"{clause['attr']} {values[clause['attr']]:g} fails a clause: verdict met"
    return None
