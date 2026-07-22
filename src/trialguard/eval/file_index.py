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
        self._trial_texts: dict[str, str] = {}
        self._loaded = False

    def _cache_path(self) -> tuple[Path, Path]:
        from trialguard.ingestion.embed import embed_tag
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        tag = embed_tag()
        return (
            INDEX_DIR / f"{self.source}_{tag}_ids.json",
            INDEX_DIR / f"{self.source}_{tag}_embeddings.npy",
        )

    def build(self, trials: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        from trialguard.ingestion.embed import eligibility_text_for_embedding, embed_batch
        from trialguard.ingestion.normalise import normalise_trial

        ids_path, emb_path = self._cache_path()

        # Always normalise: needed for BM25, trial_texts, and optionally embeddings.
        normalised = [normalise_trial(t) for t in trials]
        texts = [eligibility_text_for_embedding(t) for t in normalised]
        nct_ids_from_trials = [t["nct_id"] for t in normalised]
        self._trial_texts = dict(zip(nct_ids_from_trials, texts))

        if ids_path.exists() and emb_path.exists():
            print(f"  Loading cached index for {self.source}...")
            self._nct_ids = json.loads(ids_path.read_text())
            self._matrix = np.load(emb_path)
        else:
            print(f"  Building index for {self.source} ({len(trials)} trials)...")
            self._nct_ids = nct_ids_from_trials
            vecs = embed_batch(texts)
            self._matrix = np.array(vecs, dtype=np.float32)
            ids_path.write_text(json.dumps(self._nct_ids))
            np.save(emb_path, self._matrix)
            print(f"  Index cached: {emb_path}")

        from trialguard.ingestion.embed import _index_exclusion
        include_exc = _index_exclusion()
        tokenized = [
            _tokenize(
                t.get("title", "")
                + " "
                + " ".join(t.get("inclusion_criteria", []))
                + (" " + " ".join(t.get("exclusion_criteria", [])) if include_exc else "")
            )
            for t in normalised
        ]
        self._bm25 = BM25Okapi(tokenized)
        self._loaded = True

    def trial_texts(self) -> dict[str, str]:
        assert self._loaded, "Call build() first."
        return self._trial_texts

    def corpus_ids(self) -> set[str]:
        assert self._loaded, "Call build() first."
        return set(self._nct_ids)

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_pool: int = 50,
        bm25_pool: int = 50,
        use_keywords: bool = False,
    ) -> list[tuple[str, float]]:
        from trialguard.ingestion.embed import embed_text

        assert self._loaded, "Call build() first."

        if use_keywords:
            from trialguard.retrieval.query_transform import generate_keywords
            queries = generate_keywords(query)
        else:
            queries = [query]

        all_rankings: list[list[tuple[str, float]]] = []
        for q in queries:
            query_vec = np.array(embed_text(q, is_query=True), dtype=np.float32)
            all_rankings.append(_cosine_search(query_vec, self._matrix, self._nct_ids, dense_pool))

            tokens = _tokenize(q)
            bm25_scores = self._bm25.get_scores(tokens)
            bm25_ranked = sorted(
                zip(self._nct_ids, bm25_scores.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )[:bm25_pool]
            all_rankings.append(bm25_ranked)

        return rrf(all_rankings, top_k=top_k)


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
