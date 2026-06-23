"""Load TrialGPT published eval corpora into the trials table.

Sources:
  TREC 2021/2022: NCBI FTP JSONL files
  SIGIR:          data/eval/sigir/retrieved.json (trial info embedded per patient)
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from rich.console import Console

from trialguard.ingestion.embed import eligibility_text_for_embedding, embed_batch
from trialguard.ingestion.loader import upsert_trials
from trialguard.ingestion.normalise import normalise_trial

console = Console()

EVAL_DIR = Path("data/eval")

TREC_SOURCES = {
    "trec_2021": "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_corpus.jsonl",
    "trec_2022": "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2022_corpus.jsonl",
}


def _download_stream(url: str, dest: Path) -> None:
    if dest.exists():
        console.print(f"  {dest.name} already downloaded, skipping.")
        return
    console.print(f"  Downloading {dest.name} (large file, streaming)...")
    with httpx.Client(timeout=600, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)


def _parse_trec_jsonl(path: Path, source: str) -> list[dict]:
    """Parse TREC corpus JSONL. One trial per line with _id and metadata fields."""
    trials = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("metadata", {})
            trials.append({
                "nct_id": obj.get("_id", ""),
                "title": obj.get("title", ""),
                "status": meta.get("overall_status", ""),
                "phase": meta.get("phase", ""),
                "conditions": [],
                "interventions": [],
                "eligibility_raw": obj.get("text", ""),
                "min_age": meta.get("minimum_age", ""),
                "max_age": meta.get("maximum_age", ""),
                "sex": meta.get("gender", ""),
                "healthy_volunteers": False,
                "last_updated": "",
                "source": source,
            })
    return trials


def _parse_sigir_corpus() -> list[dict]:
    """Extract unique trials embedded in the SIGIR retrieved_trials file."""
    path = EVAL_DIR / "sigir" / "retrieved.json"
    if not path.exists():
        console.print("  SIGIR retrieved.json not found, skipping.")
        return []

    with open(path) as f:
        data = json.load(f)

    seen: set[str] = set()
    trials = []
    for patient in data:
        for key, trial_list in patient.items():
            if key in ("patient_id", "patient") or not isinstance(trial_list, list):
                continue
            for t in trial_list:
                nct_id = t.get("NCTID", "")
                if not nct_id or nct_id in seen:
                    continue
                seen.add(nct_id)
                raw_elig = (
                    (t.get("inclusion_criteria") or "")
                    + "\n"
                    + (t.get("exclusion_criteria") or "")
                )
                trials.append({
                    "nct_id": nct_id,
                    "title": t.get("brief_title", ""),
                    "status": "",
                    "phase": t.get("phase", ""),
                    "conditions": [],
                    "interventions": [],
                    "eligibility_raw": raw_elig.strip(),
                    "min_age": "",
                    "max_age": "",
                    "sex": "",
                    "healthy_volunteers": False,
                    "last_updated": "",
                    "source": "sigir",
                })
    return trials


def _ingest_trials(trials: list[dict], source: str) -> None:
    if not trials:
        console.print(f"  No trials to ingest for {source}.")
        return

    BATCH = 200
    total = 0
    for i in range(0, len(trials), BATCH):
        batch = [normalise_trial(t) for t in trials[i: i + BATCH]]
        texts = [eligibility_text_for_embedding(t) for t in batch]
        embeddings = embed_batch(texts)
        for t, emb in zip(batch, embeddings):
            t["embedding"] = emb
        upserted = upsert_trials(batch, source=source)
        total += upserted
        console.print(f"  {source}: {total}/{len(trials)} upserted...")

    console.print(f"  [green]{source} done: {total} trials.[/green]")


def load_all_eval_corpora() -> None:
    console.print("[bold]Loading TrialGPT eval corpora into trials table...[/bold]")

    # TREC 2021 + 2022
    for source, url in TREC_SOURCES.items():
        dest = EVAL_DIR / source / f"{source}_corpus.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        _download_stream(url, dest)
        console.print(f"  Parsing {source}...")
        trials = _parse_trec_jsonl(dest, source)
        console.print(f"  {source}: {len(trials)} trials parsed.")
        _ingest_trials(trials, source)

    # SIGIR
    console.print("  Parsing SIGIR corpus from retrieved.json...")
    sigir_trials = _parse_sigir_corpus()
    console.print(f"  SIGIR: {len(sigir_trials)} unique trials parsed.")
    _ingest_trials(sigir_trials, "sigir")

    console.print("[bold green]All eval corpora loaded.[/bold green]")
