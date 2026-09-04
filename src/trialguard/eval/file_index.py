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

from trialguard.retrieval.fusion import importance_weights, rrf

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
        # Owner trial id per matrix row. Equal to _nct_ids when chunking is off;
        # longer than it when a trial contributes several passages.
        self._chunk_owners: list[str] = []
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

        from trialguard.ingestion.embed import eligibility_text_for_embedding, embed_matrix
        from trialguard.ingestion.normalise import normalise_trial

        ids_path, emb_path = self._cache_path()

        # Always normalise: needed for BM25, trial_texts, and optionally embeddings.
        normalised = [normalise_trial(t) for t in trials]
        texts = [eligibility_text_for_embedding(t) for t in normalised]
        nct_ids_from_trials = [t["nct_id"] for t in normalised]
        self._trial_texts = dict(zip(nct_ids_from_trials, texts))

        self._nct_ids = nct_ids_from_trials

        if ids_path.exists() and emb_path.exists():
            print(f"  Loading cached index for {self.source}...")
            self._chunk_owners = json.loads(ids_path.read_text())
            self._matrix = np.load(emb_path)
        else:
            from trialguard.ingestion.embed import _chunking_enabled, chunk_document

            if _chunking_enabled():
                # One row per passage. The encoder truncates at 512 tokens, so a
                # single vector per trial leaves everything past the window
                # unsearchable; the embed tag carries the chunked flag so this
                # never loads against an unchunked cache.
                chunk_texts, owners = [], []
                for t in normalised:
                    for c in chunk_document(t):
                        chunk_texts.append(c)
                        owners.append(t["nct_id"])
                print(
                    f"  Building chunked index for {self.source} "
                    f"({len(trials)} trials -> {len(chunk_texts)} passages)..."
                )
            else:
                chunk_texts, owners = texts, nct_ids_from_trials
                print(f"  Building index for {self.source} ({len(trials)} trials)...")

            self._chunk_owners = owners
            # embed_matrix, not embed_batch: a whole corpus through the list path
            # costs ~1 GB of Python float objects before the array copy, which is
            # what killed the 26k-trial TREC build on an 8 GB machine.
            self._matrix = embed_matrix(chunk_texts)
            ids_path.write_text(json.dumps(self._chunk_owners))
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

    def _dense(self, query_vec: np.ndarray, pool: int) -> list[tuple[str, float]]:
        """Dense hits as trials, not passages.

        A trial scores as its best-matching passage: max, not sum, because a long
        trial would otherwise outrank a precise short one purely by having more
        chances to match. Over-fetch before collapsing, since several passages of
        the same trial can occupy the head of the list.
        """
        over = pool * 4 if len(self._chunk_owners) > len(self._nct_ids) else pool
        # Never ask for more rows than exist: argpartition raises rather than
        # clamping, so a small corpus would take down the search.
        over = min(over, self._matrix.shape[0])
        hits = _cosine_search(query_vec, self._matrix, self._chunk_owners, over)
        best: dict[str, float] = {}
        for nct, sc in hits:
            if sc > best.get(nct, float("-inf")):
                best[nct] = sc
        return sorted(best.items(), key=lambda x: x[1], reverse=True)[:pool]

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
            all_rankings.append(self._dense(query_vec, dense_pool))

            tokens = _tokenize(q)
            bm25_scores = self._bm25.get_scores(tokens)
            bm25_ranked = sorted(
                zip(self._nct_ids, bm25_scores.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )[:bm25_pool]
            all_rankings.append(bm25_ranked)

        return rrf(
            all_rankings, top_k=top_k, weights=importance_weights(len(all_rankings))
        )


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


def _load_trec_trials(source: str, keep_ids: set[str] | None = None) -> list[dict]:
    """Load TREC trials, optionally keeping only `keep_ids`.

    The agent eval needs a few hundred labelled trials, not the whole 26k corpus;
    materialising all of them (then normalising a second copy) is what makes a
    TREC agent run exceed memory on a small machine. Filtering while streaming
    keeps the selection identical and the footprint proportional to what is used.
    Retrieval eval passes no filter and still gets the full corpus.
    """
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
            nct_id = obj.get("_id", "")
            if keep_ids is not None and nct_id not in keep_ids:
                continue
            trials.append({
                "nct_id": nct_id,
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
