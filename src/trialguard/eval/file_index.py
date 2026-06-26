"""File-based retrieval index for eval corpora.

Bypasses pgvector entirely. Embeddings cached as .npy files in data/indexes/.
Dense: numpy cosine similarity (exact search, fast for <=50k vectors).
BM25: rank-bm25 in-memory.
RRF fusion applied internally.

No DB storage required — eval corpora never enter pgvector.
pgvector is reserved for the production demo (ctgov_live).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from trialguard.retrieval.fusion import rrf

INDEX_DIR = Path("data/indexes")
EVAL_DIR = Path("data/eval")


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def _cosine_search(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    nct_ids: list[str],
    top_k: int,
) -> list[tuple[str, float]]:
    scores = matrix @ query_vec
    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(nct_ids[i], float(scores[i])) for i in top_idx]


class FileIndex:
    """Retrieval index built from corpus files, no DB required."""

    def __init__(self, source: str) -> None:
        self.source = source
        self._nct_ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self._bm25 = None
        self._loaded = False

    def _cache_path(self) -> tuple[Path, Path]:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        return (
            INDEX_DIR / f"{self.source}_ids.json",
            INDEX_DIR / f"{self.source}_embeddings.npy",
        )

    def build(self, trials: list[dict]) -> None:
        from trialguard.ingestion.embed import eligibility_text_for_embedding, embed_batch
        from trialguard.ingestion.normalise import normalise_trial
        from rank_bm25 import BM25Okapi

        ids_path, emb_path = self._cache_path()

        if ids_path.exists() and emb_path.exists():
            print(f"  Loading cached index for {self.source}...")
            self._nct_ids = json.loads(ids_path.read_text())
            self._matrix = np.load(emb_path)
        else:
            print(f"  Building index for {self.source} ({len(trials)} trials)...")
            normalised = [normalise_trial(t) for t in trials]
            texts = [eligibility_text_for_embedding(t) for t in normalised]
            self._nct_ids = [t["nct_id"] for t in normalised]

            vecs = embed_batch(texts)
            self._matrix = np.array(vecs, dtype=np.float32)

            ids_path.write_text(json.dumps(self._nct_ids))
            np.save(emb_path, self._matrix)
            print(f"  Index cached: {emb_path}")

        tokenized = [
            _tokenize(t.get("title", "") + " " + " ".join(t.get("inclusion_criteria", [])))
            for t in ([normalise_trial(t) for t in trials] if trials else [{"title": "", "inclusion_criteria": []}] * len(self._nct_ids))
        ]
        self._bm25 = BM25Okapi(tokenized)
        self._loaded = True

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_pool: int = 50,
        bm25_pool: int = 50,
    ) -> list[tuple[str, float]]:
        from trialguard.ingestion.embed import embed_text

        assert self._loaded, "Call build() first."

        query_vec = np.array(embed_text(query, is_query=True), dtype=np.float32)
        dense_results = _cosine_search(query_vec, self._matrix, self._nct_ids, dense_pool)

        tokens = _tokenize(query)
        bm25_scores = self._bm25.get_scores(tokens)
        bm25_ranked = sorted(
            zip(self._nct_ids, bm25_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:bm25_pool]

        return rrf([dense_results, bm25_ranked], top_k=top_k)


# ---- Source-specific loaders ----

def _load_sigir_trials() -> list[dict]:
    path = EVAL_DIR / "sigir" / "retrieved.json"
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
                trials.append({
                    "nct_id": nct_id,
                    "title": t.get("brief_title", ""),
                    "eligibility_raw": (
                        (t.get("inclusion_criteria") or "")
                        + "\n"
                        + (t.get("exclusion_criteria") or "")
                    ).strip(),
                })
    return trials


def _load_trec_trials(source: str) -> list[dict]:
    path = EVAL_DIR / source / f"{source}_corpus.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m trialguard.scripts.load_eval_corpus"
        )
    trials = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            trials.append({
                "nct_id": obj.get("_id", ""),
                "title": obj.get("title", ""),
                "eligibility_raw": obj.get("text", ""),
            })
    return trials


_INDEX_CACHE: dict[str, FileIndex] = {}


def get_index(source: str) -> FileIndex:
    """Return built FileIndex for given source. Cached per process."""
    if source in _INDEX_CACHE:
        return _INDEX_CACHE[source]

    idx = FileIndex(source)

    if source == "sigir":
        trials = _load_sigir_trials()
    elif source in ("trec_2021", "trec_2022"):
        trials = _load_trec_trials(source)
    else:
        raise ValueError(f"Unknown eval source: {source}")

    idx.build(trials)
    _INDEX_CACHE[source] = idx
    return idx
