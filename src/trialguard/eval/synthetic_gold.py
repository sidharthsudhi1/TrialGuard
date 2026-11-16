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
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()
    ).hexdigest()[:16]
    rate = round(ok / len(rows), 4) if rows else None
    return {"n": len(rows), "agree": ok, "agree_rate": rate, "digest": digest}


def main() -> None:
    first = build()
    second = build()
    stats = recover(first)
    stats["reproducible"] = first == second
    stats["seed"] = SEED
    REPORT.write_text(json.dumps({**stats, "rows": first}, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
