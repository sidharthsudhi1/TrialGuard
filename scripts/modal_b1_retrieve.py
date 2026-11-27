"""B1 retrieval on Modal: rankings for biomarker-marked notes and their flips.

Reads data/reports/b1_jobs.json (from `b1_biomarker.py prepare`). Uses each
job's keywords as given, so no keyword LLM call. Writes rankings to
data/reports/b1_rankings.json and the top-k trial texts to
data/reports/b1_top_trials.json.

    modal volume put trialguard-embed data/eval/trec_2021/trec_2021_corpus.jsonl /corpus/
    modal volume put trialguard-embed data/indexes/trec_2021_medcpt_excl_embeddings.npy /indexes/
    modal volume put trialguard-embed data/indexes/trec_2021_medcpt_excl_ids.json /indexes/
    modal run scripts/modal_b1_retrieve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

JOBS = Path("data/reports/b1_jobs.json")
RANKINGS = Path("data/reports/b1_rankings.json")
TRIALS = Path("data/reports/b1_top_trials.json")
TOP_K = 100  # list pool 200, the same pools as the V2 and V4 deep runs

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.0",
        "transformers==5.12.1",
        "numpy",
        "rank-bm25",
        "pydantic-settings",
        "python-dotenv",
    )
    .add_local_python_source("trialguard")
)
vol = modal.Volume.from_name("trialguard-embed", create_if_missing=False)
app = modal.App("trialguard-b1-retrieve", image=image)


@app.function(cpu=8, memory=65536, timeout=4 * 3600, scaledown_window=5, volumes={"/vol": vol})
def retrieve(cohort: str, jobs: list[dict], k: int) -> dict:
    import os

    os.environ["TG_EMBED_BACKEND"] = "medcpt"
    os.environ["TG_INDEX_EXCLUSION"] = "1"
    os.environ["TG_CHUNK_DOCS"] = "0"
    os.chdir("/root")
    corpus = f"/vol/corpus/{cohort}_corpus.jsonl"
    Path(f"/root/data/eval/{cohort}").mkdir(parents=True, exist_ok=True)
    Path("/root/data/indexes").mkdir(parents=True, exist_ok=True)
    _link(corpus, f"/root/data/eval/{cohort}/{cohort}_corpus.jsonl")
    for suffix in ("embeddings.npy", "ids.json"):
        name = f"{cohort}_medcpt_excl_{suffix}"
        _link(f"/vol/indexes/{name}", f"/root/data/indexes/{name}")

    from trialguard.eval.file_index import get_index

    idx = get_index(cohort)
    rankings: dict[str, list[str]] = {}
    needed: set[str] = set()
    for job in jobs:
        hits = idx.search(job["note"], top_k=TOP_K, use_keywords=True, queries=job["keywords"])
        ids = [nct for nct, _ in hits]
        rankings[f"{cohort}|{job['patient_id']}|{job['variant']}"] = ids
        needed.update(ids[:k])
        print(job["patient_id"], job["variant"], flush=True)
    return {"rankings": rankings, "trials": _select(corpus, needed)}


def _link(src: str, dest: str) -> None:
    source, target = Path(src), Path(dest)
    if not source.exists():
        raise FileNotFoundError(src)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source)


def _select(path: str, needed: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    with open(path) as handle:
        for line in handle:
            if len(found) == len(needed):
                break
            obj = json.loads(line) if line.strip() else {}
            if obj.get("_id") in needed:
                found[obj["_id"]] = {"title": obj.get("title") or "", "text": obj.get("text") or ""}
    missing = needed - set(found)
    if missing:
        raise RuntimeError(f"{len(missing)} ranked trials missing from the corpus")
    return found


@app.local_entrypoint()
def main() -> None:
    payload = json.loads(JOBS.read_text())
    by_cohort: dict[str, list[dict]] = {}
    for job in payload["jobs"]:
        by_cohort.setdefault(job["cohort"], []).append(job)
    calls = [(c, jobs, payload["k"]) for c, jobs in sorted(by_cohort.items())]
    rankings: dict[str, list[str]] = {}
    trials: dict[str, dict] = {}
    for result in retrieve.starmap(calls):
        rankings.update(result["rankings"])
        trials.update(result["trials"])
    RANKINGS.write_text(json.dumps(rankings))
    TRIALS.write_text(json.dumps(trials))
    print(f"wrote {RANKINGS} ({len(rankings)}) and {TRIALS} ({len(trials)})")
