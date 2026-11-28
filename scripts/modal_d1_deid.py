"""D1 (Phase 40): transformer de-id candidates on the name-refusal sets.

Each model runs on 2 CPUs, matching the Fly VM, so its timings stand in for
gate (5). Writes raw entities; scoring is local (scripts/d1_score.py).

    python scripts/d1_score.py jobs
    modal run scripts/modal_d1_deid.py
    python scripts/d1_score.py score
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

JOBS = Path("data/reports/d1_jobs.json")
OUT = Path("data/reports/d1_entities.json")
MODELS = (
    "obi/deid_roberta_i2b2",
    "obi/deid_bert_i2b2",
    "StanfordAIMI/stanford-deidentifier-base",
    "StanfordAIMI/stanford-deidentifier-only-i2b2",
    "dslim/bert-base-NER",
)

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.14.0", "transformers==5.12.1", "numpy"
)
app = modal.App("trialguard-d1-deid", image=image)


@app.function(cpu=2, memory=8192, timeout=3 * 3600, scaledown_window=5)
def run(model: str, jobs: list[dict]) -> dict:
    import statistics
    import time

    import torch
    from transformers import AutoTokenizer, pipeline

    torch.set_num_threads(2)
    t0 = time.perf_counter()
    # Two of the models ship no max length, so long notes overflowed 512 positions.
    tokenizer = AutoTokenizer.from_pretrained(model, model_max_length=512)
    nlp = pipeline("token-classification", model=model, tokenizer=tokenizer,
                   aggregation_strategy="simple", stride=64)
    load_s = time.perf_counter() - t0
    nlp("Warm up the model once.")
    out, times = {}, []
    for job in jobs:
        t = time.perf_counter()
        ents = nlp(job["note"])
        times.append(time.perf_counter() - t)
        out[job["id"]] = [
            {"label": e["entity_group"], "word": e["word"], "score": round(float(e["score"]), 4),
             "start": int(e["start"]), "end": int(e["end"])}
            for e in ents
        ]
    long_note = " ".join(j["note"] for j in jobs[:8])[:4000]
    long_times = []
    for _ in range(10):
        t = time.perf_counter()
        nlp(long_note)
        long_times.append(time.perf_counter() - t)
    return {
        "model": model,
        "load_s": round(load_s, 2),
        "median_ms": round(1000 * statistics.median(times), 1),
        "p95_ms": round(1000 * sorted(times)[int(0.95 * len(times))], 1),
        "median_ms_4000_chars": round(1000 * statistics.median(long_times), 1),
        "entities": out,
    }


@app.local_entrypoint()
def main() -> None:
    jobs = json.loads(JOBS.read_text())["jobs"]
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [m for m in MODELS if m not in results]
    for r in run.starmap([(m, jobs) for m in todo], return_exceptions=True):
        if isinstance(r, Exception):
            print("failed:", r)
            continue
        results[r["model"]] = r
        print(r["model"], r["load_s"], r["median_ms"], r["p95_ms"], r["median_ms_4000_chars"])
    OUT.write_text(json.dumps(results))
