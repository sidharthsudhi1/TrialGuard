"""The demo notes the UI offers, and the allowlist the API caches.

One definition, served to the client, because two definitions drifted: the web
app hardcoded its own notes while the API's allowlist came from the SIGIR
queries fixture, so `_is_preset` was False for every note the demo could
actually produce. Every demo run therefore set `skip_cache_write` and paid a
fresh LLM call per trial, forever, and the analyst cache the fixture presets
populated was never reachable from the UI.

Being on this list is what permits a cache write, so it is an allowlist and not
a convenience: these notes are authored here, synthetic, and fixed. Free text
still skips the write, which is what stops an attacker-controlled note being
persisted or growing the store without bound.
"""

from __future__ import annotations

DEMO_PRESETS: tuple[dict[str, str], ...] = (
    {
        "label": "NSCLC stage IV",
        "note": (
            "58-year-old woman with stage IV non-small cell lung cancer, ECOG "
            "performance status 1, never-smoker, EGFR wild-type. No prior systemic "
            "therapy. Adequate organ function."
        ),
    },
    {
        "label": "Breast cancer adjuvant",
        "note": (
            "45-year-old woman with early-stage hormone receptor-positive breast "
            "cancer, status post lumpectomy, planning adjuvant endocrine therapy. "
            "No metastatic disease. ECOG 0."
        ),
    },
)

DEMO_PRESET_NOTES: frozenset[str] = frozenset(p["note"] for p in DEMO_PRESETS)
