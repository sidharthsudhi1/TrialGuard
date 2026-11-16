"""TREC 2023 retrieval on Modal. The 451k BM25 stays off the laptop.

Reads data/reports/v4_keywords.json. Writes rankings locally and the keyword
top-100 trial texts to data/reports/v4_top_trials.json.

    modal run scripts/modal_retrieve_trec2023.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

KEYWORDS = Path("data/reports/v4_keywords.json")
RANKINGS = Path("data/reports/v4_rankings.json")
TRIALS = Path("data/reports/v4_top_trials.json")
REMOTE_TRIALS = "/indexes/v4_top_trials.json"

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
app = modal.App("trialguard-retrieve-trec2023", image=image)


@app.function(
    cpu=8,
    memory=65536,
    timeout=6 * 3600,
    scaledown_window=5,
    volumes={"/vol": vol},
)
def retrieve(patients: list[dict]) -> dict:
    import os

    os.environ["TG_EMBED_BACKEND"] = "medcpt"
    os.environ["TG_INDEX_EXCLUSION"] = "1"
    os.environ["TG_CHUNK_DOCS"] = "0"
    os.chdir("/root")
    for path in (
        Path("/root/data/eval/trec_2023"),
        Path("/root/data/indexes"),
    ):
        path.mkdir(parents=True, exist_ok=True)
    _link("/vol/corpus/trec_2023_corpus.jsonl", "/root/data/eval/trec_2023/trec_2023_corpus.jsonl")
    _link(
        "/vol/indexes/trec_2023_medcpt_excl_embeddings.npy",
        "/root/data/indexes/trec_2023_medcpt_excl_embeddings.npy",
    )
    _link(
        "/vol/indexes/trec_2023_medcpt_excl_ids.json",
        "/root/data/indexes/trec_2023_medcpt_excl_ids.json",
    )

    from trialguard.eval.file_index import get_index

    idx = get_index("trec_2023")
    rankings: dict[str, dict[str, list[str]]] = {}
    needed: set[str] = set()
    for i, patient in enumerate(patients, start=1):
        keyword = idx.search(
            patient["note"],
            top_k=500,
            use_keywords=True,
            queries=patient["keywords"],
        )
        raw = idx.search(patient["note"], top_k=500, queries=[patient["note"]])
        kw_ids = [nct for nct, _ in keyword]
        rankings[patient["patient_id"]] = {
            "keyword": kw_ids,
            "raw": [nct for nct, _ in raw],
        }
        needed.update(kw_ids[:100])
        print(f"{i}/{len(patients)} {patient['patient_id']}", flush=True)

    trials = _select_trials("/vol/corpus/trec_2023_corpus.jsonl", needed)
    dest = Path("/vol") / REMOTE_TRIALS.lstrip("/")
    dest.write_text(json.dumps(trials))
    vol.commit()
    return rankings


def _link(src: str, dest: str) -> None:
    source, target = Path(src), Path(dest)
    if not source.exists():
        raise FileNotFoundError(src)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source)


def _select_trials(path: str, needed: set[str]) -> dict[str, dict]:
    import json

    found: dict[str, dict] = {}
    with open(path) as handle:
        for line in handle:
            if len(found) == len(needed):
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            nct = obj.get("_id", "")
            if nct not in needed:
                continue
            found[nct] = {"title": obj.get("title") or "", "text": obj.get("text") or ""}
    missing = needed - set(found)
    if missing:
        raise RuntimeError(f"{len(missing)} ranked trials missing from the corpus")
    return found


@app.local_entrypoint()
def main() -> None:
    payload = json.loads(KEYWORDS.read_text())
    rankings = retrieve.remote(payload["patients"])
    RANKINGS.parent.mkdir(parents=True, exist_ok=True)
    RANKINGS.write_text(json.dumps(rankings))
    print(f"wrote {RANKINGS} ({len(rankings)} patients)")
    with TRIALS.open("wb") as handle:
        for chunk in vol.read_file(REMOTE_TRIALS):
            handle.write(chunk)
    print(f"wrote {TRIALS} ({TRIALS.stat().st_size / 1e6:.1f} MB)")
