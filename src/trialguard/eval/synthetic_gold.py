"""G2: synthetic patients whose criterion truth is computed from a fact table.

Criteria are sentences taken from the TREC corpora. The note is rendered from
the fact table. Truth is the fact table, not the note, so a checker that only
sees the note can be scored against it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from trialguard.verify.entailment import _criterion_holds, _note_value

EVAL = Path("data/eval")
REPORT = Path("data/reports/g2_synthetic_gold.json")
SEED = 20261110

_SENTENCE = re.compile(r"[^.\n]{12,240}")
_ATTRS = ("age", "creatinine", "bilirubin", "platelets", "anc", "hemoglobin", "ecog")


def _sentences(path: Path, limit: int) -> list[dict]:
    found: list[dict] = []
    seen: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if len(found) >= limit:
                break
            obj = json.loads(line)
            text = obj.get("text") or obj.get("eligibility_raw") or ""
            nct = obj.get("_id") or obj.get("nct_id") or ""
            for sentence in _SENTENCE.findall(text):
                if sentence in seen:
                    continue
                for attr in _ATTRS:
                    if _criterion_holds(sentence, 1.0, attr) is None and attr != "age":
                        continue
                    if attr == "age" and _criterion_holds(sentence, 40.0, attr) is None:
                        continue
                    # A sentence that parses for this attribute.
                    if _criterion_holds(sentence, 1.0, attr) is None:
                        continue
                    seen.add(sentence)
                    found.append({"nct_id": nct, "criterion": sentence.strip(), "attr": attr})
                    break
                if len(found) >= limit:
                    break
    return found


def _facts(rng: np.random.Generator) -> dict[str, float]:
    return {
        "age": float(rng.integers(18, 85)),
        "creatinine": round(float(rng.uniform(0.6, 3.5)), 2),
        "bilirubin": round(float(rng.uniform(0.2, 3.0)), 2),
        "platelets": float(rng.integers(50, 400)),
        "anc": float(rng.integers(500, 8000)),
        "hemoglobin": round(float(rng.uniform(7.0, 16.0)), 1),
        "ecog": float(rng.integers(0, 4)),
    }


def _note(facts: dict[str, float]) -> str:
    sex = "woman" if int(facts["age"]) % 2 == 0 else "man"
    return (
        f"A {int(facts['age'])}-year-old {sex}. "
        f"Creatinine {facts['creatinine']} mg/dL. "
        f"Bilirubin {facts['bilirubin']} mg/dL. "
        f"Platelets {int(facts['platelets'])}. "
        f"ANC {int(facts['anc'])}. "
        f"Hemoglobin {facts['hemoglobin']} g/dL. "
        f"ECOG {int(facts['ecog'])}."
    )


def build(n: int = 200, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    pool: list[dict] = []
    for cohort in ("trec_2021", "trec_2022"):
        path = EVAL / cohort / f"{cohort}_corpus.jsonl"
        if path.exists():
            pool.extend(_sentences(path, 400))
    if not pool:
        raise FileNotFoundError("TREC corpus jsonl is not on disk")
    order = rng.permutation(len(pool))
    rows: list[dict] = []
    for i in order:
        if len(rows) >= n:
            break
        spec = pool[int(i)]
        facts = _facts(rng)
        holds = _criterion_holds(spec["criterion"], facts[spec["attr"]], spec["attr"])
        if holds is None:
            continue
        note = _note(facts)
        rows.append(
            {
                "nct_id": spec["nct_id"],
                "attr": spec["attr"],
                "criterion": spec["criterion"],
                "facts": facts,
                "note": note,
                "holds": bool(holds),
            }
        )
    return rows


def recover(rows: list[dict]) -> dict:
    """How often the note parser agrees with the fact-table truth."""
    ok = 0
    for row in rows:
        value = _note_value(row["attr"], row["note"])
        got = None if value is None else _criterion_holds(row["criterion"], value, row["attr"])
        ok += int(got is row["holds"])
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
    rate = round(ok / len(rows), 4) if rows else None
    return {"n": len(rows), "agree": ok, "agree_rate": rate, "digest": digest}


STRICT_SEED = 20261117
STRICT_REPORT = Path("data/reports/g2b_gold.json")
_PER_ATTR = 40

_OPS = {
    ">=": ">=",
    "=>": ">=",
    "at least": ">=",
    "greater than or equal to": ">=",
    ">": ">",
    "greater than": ">",
    "more than": ">",
    "over": ">",
    "above": ">",
    "<=": "<=",
    "=<": "<=",
    "less than or equal to": "<=",
    "no more than": "<=",
    "not more than": "<=",
    "up to": "<=",
    "<": "<",
    "less than": "<",
    "under": "<",
    "below": "<",
}
_OP_RE = "|".join(sorted((re.escape(k) for k in _OPS), key=len, reverse=True))
_NUM_RE = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
_PER_UL = r"(?:cells\s?)?(?:/|per\s)(?:ul|mm3|mm\^3|mcl|cubic\smillimeter)"
_PER_L9 = r"(?:x|\*)\s?10\^?9\s?/\s?l|(?:x|\*)\s?109\s?/\s?l"
_LABS = {
    "creatinine": (r"(?:serum\s)?creatinine(?:\slevel)?", {r"mg/dl": 1.0}),
    "bilirubin": (r"(?:serum\s)?(?:total\s)?bilirubin(?:\slevel)?", {r"mg/dl": 1.0}),
    "hemoglobin": (
        r"(?:hemoglobin|haemoglobin|hgb|hb)(?:\slevel)?",
        {r"g/dl": 1.0, r"g/l": 0.1},
    ),
    "platelets": (
        r"(?:platelets?|platelet\scount|plt)",
        {_PER_UL: 0.001, _PER_L9: 1.0},
    ),
    "anc": (
        r"(?:anc|absolute\sneutrophil\scount(?:\s\(anc\))?)",
        {_PER_UL: 1.0, _PER_L9: 1000.0},
    ),
}
_YEARS = r"(?:\s?(?:years?|yrs?)(?:\sold|\sof\sage)?)"
_AGE_OP = re.compile(rf"(?:age[d]?\s?:?\s?)?({_OP_RE})\s?(\d{{1,3}}){_YEARS}?")
_AGE_RANGE = re.compile(
    rf"age[d]?\s?:?\s?(?:between\s)?(\d{{1,3}})\s?(?:-|to|and)\s?(\d{{1,3}}){_YEARS}"
)
_AGE_TAIL = re.compile(
    rf"(?:age[d]?\s?:?\s?)?(\d{{1,3}}){_YEARS}?\s(?:or|and)\s(older|above|over|younger)"
)
_ECOG_NAME = (
    r"(?:ecog|eastern\scooperative\soncology\sgroup(?:\s\(ecog\))?)"
    r"(?:\sperformance\sstatus|\sps)?(?:\s\(ps\))?(?:\sscore)?(?:\sof)?\s?:?\s?"
)
_ECOG_OP = re.compile(rf"{_ECOG_NAME}({_OP_RE})\s?(\d)")
_ECOG_RANGE = re.compile(rf"{_ECOG_NAME}(\d)\s?(?:-|to|or)\s?(\d)")
_ECOG_SET = re.compile(rf"{_ECOG_NAME}(\d)\s?,\s?(\d)\s?,?\s?or\s(\d)")
_BULLET = re.compile(r"^(?:[-*•·]|\d{1,2}[.)]|[a-z][.)])\s*")


def _clean(line: str) -> str:
    text = line.strip().lower()
    for a, b in (
        ("≥", ">="),
        ("≤", "<="),
        ("µ", "u"),
        ("μ", "u"),
        ("⁹", "9"),
        ("×", "x"),
        ("–", "-"),
        (" ", " "),
    ):
        text = text.replace(a, b)
    text = _BULLET.sub("", text)
    text = re.sub(r"\s+", " ", text).rstrip(" .;,")
    return text


def _num(raw: str) -> float:
    return float(raw.replace(",", ""))


def parse_strict(line: str) -> dict | None:
    """One machine-checkable threshold in the note's units, or None."""
    text = _clean(line)
    if not 6 <= len(text) <= 90:
        return None
    m = _AGE_RANGE.fullmatch(text)
    if m:
        return {"attr": "age", "low": float(m[1]), "high": float(m[2])}
    m = _AGE_TAIL.fullmatch(text)
    if m and ("age" in text or "year" in text):
        op = "<=" if m[2] == "younger" else ">="
        return {"attr": "age", "op": op, "bound": float(m[1])}
    m = _AGE_OP.fullmatch(text)
    if m and ("age" in text or "year" in text):
        return {"attr": "age", "op": _OPS[m[1]], "bound": float(m[2])}
    m = _ECOG_SET.fullmatch(text)
    if m:
        return {"attr": "ecog", "allowed": sorted({float(m[i]) for i in (1, 2, 3)})}
    m = _ECOG_RANGE.fullmatch(text)
    if m:
        return {"attr": "ecog", "low": float(m[1]), "high": float(m[2])}
    m = _ECOG_OP.fullmatch(text)
    if m:
        return {"attr": "ecog", "op": _OPS[m[1]], "bound": float(m[2])}
    for attr, (name, units) in _LABS.items():
        for unit, scale in units.items():
            m = re.fullmatch(rf"{name}\s?(?:of\s)?:?\s?({_OP_RE})\s?{_NUM_RE}\s?(?:{unit})", text)
            if m:
                return {"attr": attr, "op": _OPS[m[1]], "bound": _num(m[2]) * scale}
    return None


def holds_strict(spec: dict, value: float) -> bool:
    if "allowed" in spec:
        return value in spec["allowed"]
    if "low" in spec:
        return spec["low"] <= value <= spec["high"]
    return {
        ">=": value >= spec["bound"],
        ">": value > spec["bound"],
        "<=": value <= spec["bound"],
        "<": value < spec["bound"],
    }[spec["op"]]


def _strict_note(facts: dict[str, float]) -> str:
    sex = "woman" if int(facts["age"]) % 2 == 0 else "man"
    return (
        f"A {int(facts['age'])}-year-old {sex}. "
        f"Creatinine {facts['creatinine']} mg/dL. "
        f"Bilirubin {facts['bilirubin']} mg/dL. "
        f"Platelets {int(facts['platelets'])} x10^9/L. "
        f"ANC {int(facts['anc'])}/uL. "
        f"Hemoglobin {facts['hemoglobin']} g/dL. "
        f"ECOG {int(facts['ecog'])}."
    )


_RANGES = {
    "age": (18, 84, 0),
    "creatinine": (0.4, 3.5, 2),
    "bilirubin": (0.2, 3.0, 2),
    "platelets": (30, 400, 0),
    "anc": (300, 8000, 0),
    "hemoglobin": (6.0, 16.0, 1),
    "ecog": (0, 4, 0),
}


def _near(spec: dict, attr: str, rng: np.random.Generator) -> float:
    """A value near the criterion's edge, so `<` vs `<=` and borders get tested."""
    lo, hi, digits = _RANGES[attr]
    edges = [spec[k] for k in ("bound", "low", "high") if k in spec] or spec["allowed"]
    edge = float(rng.choice(edges))
    width = 1.0 if digits == 0 and attr in ("age", "ecog") else max(abs(edge) * 0.15, 0.1)
    value = float(np.clip(edge + rng.uniform(-width, width), lo, hi))
    return float(round(value)) if digits == 0 else round(value, digits)


def _strict_pool() -> list[dict]:
    seen: set[str] = set()
    pool: list[dict] = []
    for cohort in ("trec_2021", "trec_2022"):
        path = EVAL / cohort / f"{cohort}_corpus.jsonl"
        if not path.exists():
            continue
        with path.open() as handle:
            for raw in handle:
                obj = json.loads(raw)
                for line in (obj.get("text") or "").split("\n"):
                    spec = parse_strict(line)
                    key = _clean(line)
                    if spec is None or key in seen:
                        continue
                    seen.add(key)
                    pool.append({"nct_id": obj.get("_id", ""), "criterion": line.strip(), **spec})
    if not pool:
        raise FileNotFoundError("TREC corpus jsonl is not on disk")
    return sorted(pool, key=lambda r: (r["attr"], r["criterion"], r["nct_id"]))


def build_strict(n: int = 200, seed: int = STRICT_SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    pool = _strict_pool()
    taken: dict[str, int] = {}
    rows: list[dict] = []
    for i in rng.permutation(len(pool)):
        if len(rows) >= n:
            break
        spec = dict(pool[int(i)])
        attr = spec["attr"]
        if taken.get(attr, 0) >= _PER_ATTR:
            continue
        taken[attr] = taken.get(attr, 0) + 1
        facts = _facts(rng)
        facts["platelets"] = float(rng.integers(30, 400))
        if rng.random() < 0.5:
            facts[attr] = _near(spec, attr, rng)
        criterion = spec.pop("criterion")
        rows.append(
            {
                "id": f"g2b-{len(rows):03d}",
                "nct_id": spec.pop("nct_id"),
                "criterion": criterion,
                "spec": spec,
                "facts": facts,
                "note": _strict_note(facts),
                "holds": holds_strict(spec, facts[attr]),
            }
        )
    return rows


def _rates(pairs: list[tuple[bool, str]]) -> dict:
    from trialguard.eval.significance import bootstrap_ci

    holds = np.array([h for h, _ in pairs], dtype=bool)
    verdict = np.array([v for _, v in pairs])
    met, not_met = verdict == "met", verdict == "not_met"
    decisive = met | not_met
    right = (holds & met) | (~holds & not_met)

    def mean(mask, base):
        def stat(idx):
            den = base[idx].sum()
            return (mask[idx] & base[idx]).sum() / den if den else np.nan

        return stat

    every = np.ones(len(pairs), dtype=bool)
    return {
        "n": len(pairs),
        "accuracy": bootstrap_ci(mean(right, every), len(pairs)),
        "coverage": bootstrap_ci(mean(decisive, every), len(pairs)),
        "decisive_accuracy": bootstrap_ci(mean(right, decisive), len(pairs)),
        "false_exclusion": bootstrap_ci(mean(not_met, holds), len(pairs)),
        "false_inclusion": bootstrap_ci(mean(met, ~holds), len(pairs)),
        "verdicts": {v: int((verdict == v).sum()) for v in sorted(set(verdict.tolist()))},
    }


def score_rows(gold: list[dict], calls: dict[str, dict]) -> dict:
    """Criterion accuracy against computed truth. Abstentions count as wrong."""
    scored = [(r, calls[r["id"]]) for r in gold if "verdict" in calls.get(r["id"], {})]
    overall = _rates([(r["holds"], c["verdict"]) for r, c in scored])
    by_attr = {}
    for attr in _RANGES:
        sub = [(r["holds"], c["verdict"]) for r, c in scored if r["spec"]["attr"] == attr]
        if sub:
            by_attr[attr] = _rates(sub)
    return {"overall": overall, "by_attr": by_attr}


def main_strict() -> None:
    first = build_strict()
    second = build_strict()
    digest = hashlib.sha256(json.dumps(first, sort_keys=True).encode()).hexdigest()[:16]
    stats = {
        "n": len(first),
        "seed": STRICT_SEED,
        "digest": digest,
        "reproducible": first == second,
        "by_attr": {a: sum(1 for r in first if r["spec"]["attr"] == a) for a in _RANGES},
        "holds_rate": round(sum(r["holds"] for r in first) / len(first), 4),
    }
    STRICT_REPORT.write_text(json.dumps({**stats, "rows": first}, indent=2))
    print(json.dumps(stats, indent=2))


def main() -> None:
    first = build()
    second = build()
    stats = recover(first)
    stats["reproducible"] = first == second
    stats["seed"] = SEED
    REPORT.write_text(json.dumps({**stats, "rows": first}, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    import sys

    main_strict() if "--strict" in sys.argv else main()
