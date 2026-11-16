"""C1: time-to-first keyword versus the full extraction, on cold synthetic notes.

Rankings are covered by tests/test_keyword_stream.py. This only times the call.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from trialguard.retrieval.query_transform import generate_keywords

OUT = Path("data/reports/c1_stream_time.json")

NOTES = [
    "61-year-old man with metastatic prostate cancer, prior enzalutamide, ECOG 1, "
    "rising PSA, no visceral disease. Nonce c1-a.",
    "52-year-old woman with relapsed diffuse large B-cell lymphoma, two prior lines, "
    "ECOG 0, no CNS disease. Nonce c1-b.",
    "44-year-old man with unresectable cholangiocarcinoma, ECOG 1, bilirubin normal, "
    "no prior systemic therapy. Nonce c1-c.",
]


def _time_one(note: str) -> dict:
    first: dict[str, float] = {}

    def on_keyword(kw: str) -> None:
        first.setdefault("t", time.perf_counter())

    t0 = time.perf_counter()
    keywords = generate_keywords(note, on_keyword=on_keyword)
    total_ms = (time.perf_counter() - t0) * 1000
    first_ms = (first["t"] - t0) * 1000 if "t" in first else None
    return {
        "n_keywords": len(keywords),
        "first_ms": None if first_ms is None else round(first_ms, 1),
        "total_ms": round(total_ms, 1),
        "first_before_total": bool(first_ms is not None and first_ms < total_ms),
    }


def main() -> None:
    rows = [_time_one(note) for note in NOTES]
    report = {
        "n": len(rows),
        "rows": rows,
        "adopt": all(r["first_before_total"] for r in rows) and bool(rows),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
