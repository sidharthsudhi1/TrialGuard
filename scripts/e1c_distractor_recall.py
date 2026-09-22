"""E1c: does retrieval survive leaving oncology?

The oncology scope lock (CLAUDE.md) means every recall number in this repo was
measured on a haystack that had already been narrowed to the right specialty. A
buyer's corpus is not pre-filtered. This grows the haystack with real
all-conditions trials while holding the gold fixed, and reports what recall does.

Gold and the TREC 2021 corpus stay exactly as the committed eval uses them, so
the zero-distractor row reproduces the standing number and every later row is
comparable to it by construction. Distractors are real CT.gov records fetched
outside the oncology query, deduped against the eval corpus by NCT id.

Dense-only on purpose: this isolates the embedding model against corpus
diversity. The lexical arm and RRF are held out rather than measured, because
mixing them in would confound "does MedCPT still separate" with "does FTS still
separate".

    python scripts/e1c_distractor_recall.py --fetch 100000     # writes distractors
    python scripts/e1c_distractor_recall.py --eval --steps 0,25000,50000,100000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

EVAL_DIR = Path("data/eval/trec_2021")
INDEX_DIR = Path("data/indexes")
WORK = Path("data/e1c")
DIM = 768


def _doc_text(trial: dict) -> str:
    """Production's own builder, so distractors are embedded like the base index.

    Mirroring it here would silently drift from TG_INDEX_EXCLUSION, and the base
    index this is compared against is the _excl one.
    """
    from trialguard.ingestion.embed import eligibility_text_for_embedding

    return eligibility_text_for_embedding(trial)


def fetch_distractors(n: int, out_path: Path, exclude: set[str]) -> int:
    from trialguard.ingestion.ctgov import ALL_CONDITIONS, fetch_oncology_trials
    from trialguard.ingestion.normalise import normalise_trial

    kept = 0
    t0 = time.perf_counter()
    with out_path.open("w") as fh:
        # Over-fetch: a large share of all-conditions hits are already in the
        # eval corpus, and those are gold, not distractors.
        for trial in fetch_oncology_trials(max_trials=n * 3, condition=ALL_CONDITIONS):
            if trial["nct_id"] in exclude:
                continue
            norm = normalise_trial(trial)
            text = _doc_text(norm)
            if not text.strip():
                continue
            fh.write(json.dumps({"nct_id": norm["nct_id"], "text": text}) + "\n")
            kept += 1
            if kept % 5000 == 0:
                print(f"  fetched {kept}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True)
            if kept >= n:
                break
    return kept


def embed_distractors(jsonl: Path, out_npy: Path, batch_size: int) -> None:
    from trialguard.ingestion.embed import embed_matrix

    texts = [json.loads(line)["text"] for line in jsonl.open()]
    print(f"embedding {len(texts)} distractors", flush=True)
    t0 = time.perf_counter()
    mat = embed_matrix(texts, batch_size=batch_size, is_query=False)
    np.save(out_npy, mat.astype("float32"))
    rate = len(texts) / (time.perf_counter() - t0)
    print(f"embedded at {rate:.1f} docs/s -> {out_npy}", flush=True)


def load_gold() -> tuple[dict[str, list[str]], dict[str, str]]:
    gold: dict[str, list[str]] = {}
    for line in (EVAL_DIR / "qrels.tsv").read_text().splitlines()[1:]:
        qid, cid, score = line.split("\t")
        if int(score) > 0:
            gold.setdefault(qid, []).append(cid)
    queries = {
        json.loads(ln)["_id"]: json.loads(ln)["text"]
        for ln in (EVAL_DIR / "queries.jsonl").read_text().splitlines()
    }
    return gold, queries


def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", type=int, default=0, help="fetch N distractors and exit")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--steps", default="0,25000,50000,100000")
    ap.add_argument("--ks", default="10,50,100,200")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tag", default="medcpt_excl")
    ap.add_argument("--out", default="data/reports/e1c_distractor_recall.json")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    jsonl, npy = WORK / "distractors.jsonl", WORK / "distractors.npy"

    base_ids = json.loads((INDEX_DIR / f"trec_2021_{args.tag}_ids.json").read_text())

    if args.fetch:
        kept = fetch_distractors(args.fetch, jsonl, exclude=set(base_ids))
        print(f"kept {kept} distractors -> {jsonl}")
        return

    if args.embed:
        embed_distractors(jsonl, npy, args.batch_size)
        return

    if not args.eval:
        ap.error("pass --fetch, --embed or --eval")

    from trialguard.ingestion.embed import embed_matrix

    base = np.load(INDEX_DIR / f"trec_2021_{args.tag}_embeddings.npy").astype("float32")
    dist = np.load(npy).astype("float32") if npy.exists() else np.zeros((0, DIM), "float32")
    gold, queries = load_gold()
    qids = [q for q in queries if q in gold]
    qmat = embed_matrix([queries[q] for q in qids], batch_size=args.batch_size, is_query=True)

    ks = [int(k) for k in args.ks.split(",")]
    rows = []
    for step in [int(s) for s in args.steps.split(",")]:
        if step > len(dist):
            print(f"skip {step}: only {len(dist)} distractors available")
            continue
        corpus = np.vstack([base, dist[:step]]) if step else base
        ids = base_ids + [f"D{i}" for i in range(step)]
        scores = qmat @ corpus.T
        top = np.argsort(scores, axis=1)[:, ::-1][:, : max(ks)]
        per_k = {}
        for k in ks:
            per_k[f"recall@{k}"] = round(
                float(np.mean([
                    recall_at([ids[j] for j in top[i][:k]], set(gold[q]), k)
                    for i, q in enumerate(qids)
                ])), 4
            )
        row = {"distractors": step, "corpus_size": len(ids), **per_k}
        rows.append(row)
        print(json.dumps(row), flush=True)

    out = {
        "cohort": "trec_2021",
        "n_queries": len(qids),
        "base_corpus": len(base_ids),
        "distractors_available": len(dist),
        "note": "dense-only MedCPT; lexical and RRF held out to isolate the embedder",
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
