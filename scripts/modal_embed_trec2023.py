"""Embed the TREC 2023 corpus on a Modal L4.

Document encoder only (ncbi/MedCPT-Article-Encoder). fp16 compute, float32
unit vectors on disk, one row per trial in corpus order. The filenames match
FileIndex's embed_tag() on this machine.

    modal run scripts/modal_embed_trec2023.py
"""

from __future__ import annotations

from pathlib import Path

import modal

CORPUS = Path("data/eval/trec_2023/trec_2023_corpus.jsonl")
INDEX_DIR = Path("data/indexes")
REMOTE_CORPUS = "/corpus/trec_2023_corpus.jsonl"
REMOTE_DIR = "/indexes"
MODEL = "ncbi/MedCPT-Article-Encoder"
MAXLEN = 512
DIM = 768
BATCH = 64

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "transformers==5.12.1", "numpy")
    .add_local_python_source("trialguard")
)
vol = modal.Volume.from_name("trialguard-embed", create_if_missing=True)
app = modal.App("trialguard-embed-trec2023", image=image)


@app.function(
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=4 * 3600,
    scaledown_window=5,
    volumes={"/vol": vol},
)
def embed(include_exclusion: bool, tag: str) -> dict:
    import json
    import os
    import time

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ["TG_INDEX_EXCLUSION"] = "1" if include_exclusion else "0"
    os.environ["TG_CHUNK_DOCS"] = "0"
    from trialguard.ingestion.embed import eligibility_text_for_embedding
    from trialguard.ingestion.normalise import normalise_trial

    src = "/vol" + REMOTE_CORPUS
    n = 0
    with open(src) as handle:
        for line in handle:
            if line.strip():
                n += 1

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to("cuda").half().eval()
    matrix = np.empty((n, DIM), dtype=np.float32)
    ids: list[str] = []
    batch_text: list[str] = []
    batch_ids: list[str] = []
    done = 0
    t0 = time.perf_counter()

    def flush() -> None:
        nonlocal done
        if not batch_text:
            return
        enc = tok(
            batch_text,
            truncation=True,
            padding=True,
            max_length=MAXLEN,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            cls = model(**enc).last_hidden_state[:, 0, :]
        vecs = torch.nn.functional.normalize(cls.float(), p=2, dim=1)
        matrix[done : done + len(batch_text)] = vecs.cpu().numpy()
        ids.extend(batch_ids)
        done += len(batch_text)
        batch_text.clear()
        batch_ids.clear()
        if done % 8192 < BATCH:
            rate = done / (time.perf_counter() - t0)
            print(f"  {done}/{n}  {rate:.0f} docs/s", flush=True)

    with open(src) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            trial = normalise_trial({
                "nct_id": obj["_id"],
                "title": obj.get("title") or "",
                "eligibility_raw": obj.get("text") or "",
            })
            batch_text.append(eligibility_text_for_embedding(trial))
            batch_ids.append(obj["_id"])
            if len(batch_text) >= BATCH:
                flush()
        flush()

    if done != n:
        raise RuntimeError(f"embedded {done} rows, corpus has {n}")

    out_dir = Path("/vol") / REMOTE_DIR.lstrip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"trec_2023_{tag}_embeddings.npy", matrix)
    (out_dir / f"trec_2023_{tag}_ids.json").write_text(json.dumps(ids))
    vol.commit()
    secs = time.perf_counter() - t0
    return {"trials": n, "seconds": round(secs, 1), "docs_per_s": round(n / secs, 1), "tag": tag}


def _corpus_present() -> bool:
    try:
        names = [Path(entry.path).name for entry in vol.listdir("/corpus")]
    except Exception:
        return False
    return "trec_2023_corpus.jsonl" in names


@app.local_entrypoint()
def main(force_upload: bool = False) -> None:
    from trialguard.ingestion.embed import _index_exclusion, embed_tag

    if not CORPUS.exists():
        raise SystemExit(f"missing {CORPUS}")
    tag = embed_tag()
    if force_upload or not _corpus_present():
        print(f"uploading {CORPUS}")
        with vol.batch_upload(force=True) as batch:
            batch.put_file(str(CORPUS), REMOTE_CORPUS)
    else:
        print("corpus already on the volume")

    stats = embed.remote(_index_exclusion(), tag)
    print(stats)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    for entry in vol.listdir(REMOTE_DIR):
        name = Path(entry.path).name
        if tag not in name:
            continue
        dest = INDEX_DIR / name
        with dest.open("wb") as handle:
            for chunk in vol.read_file(entry.path):
                handle.write(chunk)
        print(f"saved {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
