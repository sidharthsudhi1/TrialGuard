"""In-memory dense index for the production corpus.

Postgres is not doing approximate search on `ctgov_live`. The planner declines
the ivfflat index at `pgvector_probes = 40`, because probing 40 of 161 lists
costs more than reading the table, so every dense query is an exact scan of all
25,965 vectors — measured at ~230ms server-side, and there are 12 of them per
search once keywords are expanded.

At this corpus size the scan is better done here. 25,965 x 768 float32 is 80 MB,
the comparison is one matrix multiply, and BLAS answers every keyword in a single
pass: measured at 7.5ms for 12 query vectors against ~10s of cumulative database
work. `eval/file_index.py` has always done exactly this for eval corpora; the
production path kept using pgvector because of how the two were labelled, not
because anything measured said it was faster.

Two properties keep this from being a downgrade:

- **Loaded in the background.** Boot is already the slow part of this deploy, so
  startup does not wait on an 80 MB fetch. Searches run against Postgres until
  the matrix is resident, then switch.
- **Falls back.** Any failure to load, or a request for a source this cache does
  not hold, returns None and the caller uses SQL. The worst case is the speed
  this system already had.

It stops being the right answer somewhere past a million rows, where the matrix
no longer fits comfortably and a real ANN index earns its keep. That is far from
26k.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


class VectorCache:
    """Embeddings for one corpus source, held as a normalized float32 matrix."""

    def __init__(self, source: str):
        self.source = source
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self._lock = threading.Lock()
        self._loading = False
        self.load_seconds: float | None = None
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    def load(self) -> bool:
        """Fetch every embedding for this source and build the matrix."""
        with self._lock:
            if self._matrix is not None or self._loading:
                return self._matrix is not None
            self._loading = True
        try:
            t0 = time.perf_counter()
            from trialguard.db.schema import get_conn

            # embedding::text, parsed in bulk here, rather than pgvector's
            # register_vector adapter. The adapter converts row by row through
            # Python and took 474s for this corpus; splitting the text costs
            # seconds, and the fetch itself dominates either way.
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT nct_id, embedding::text FROM trials "
                    "WHERE embedding IS NOT NULL AND source = %s",
                    (self.source,),
                )
                rows = cur.fetchall()

            if not rows:
                self.error = "no embeddings for source"
                return False

            ids = [r[0] for r in rows]
            # One C-level parse over the whole corpus. Splitting per row in Python
            # materialises ~20 million string objects and measured 150s against
            # 1.9s for this; np.fromstring's text mode is deprecated but not
            # removed, so fall back rather than pay that.
            joined = ",".join(r[1][1:-1] for r in rows)
            try:
                flat = np.fromstring(joined, sep=",", dtype=np.float32)
            except (DeprecationWarning, TypeError, ValueError):
                flat = np.asarray(joined.split(","), dtype=np.float32)
            if flat.size % len(rows):
                raise ValueError(f"ragged embeddings: {flat.size} values over {len(rows)} rows")
            matrix = flat.reshape(len(rows), flat.size // len(rows))

            # Normalize once here so each query is a plain dot product. MedCPT
            # already returns unit vectors, but a corpus loaded by another path
            # must not silently rank by magnitude.
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix /= norms

            with self._lock:
                self._ids = ids
                self._matrix = matrix
            self.load_seconds = time.perf_counter() - t0
            log.info(
                "vector cache ready: %d x %d (%.0f MB) in %.1fs",
                matrix.shape[0], matrix.shape[1], matrix.nbytes / 1e6, self.load_seconds,
            )
            return True
        except Exception as e:  # noqa: BLE001 — degrade to SQL, but say why
            self.error = f"{type(e).__name__}: {e}"
            log.warning("vector cache load failed (%s); dense search stays on SQL", self.error)
            return False
        finally:
            with self._lock:
                self._loading = False

    def search(self, query_vec, top_k: int) -> list[tuple[str, float]] | None:
        """Top-k by cosine similarity, or None when the matrix is not resident."""
        matrix, ids = self._matrix, self._ids
        if matrix is None:
            return None
        q = np.asarray(query_vec, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm:
            q = q / norm
        scores = matrix @ q
        k = min(top_k, scores.shape[0])
        # argpartition finds the top k without sorting 26k scores, then only that
        # slice is ordered.
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(ids[i], float(scores[i])) for i in idx]


_caches: dict[str, VectorCache] = {}
_caches_lock = threading.Lock()


def get_cache(source: str) -> VectorCache:
    with _caches_lock:
        if source not in _caches:
            _caches[source] = VectorCache(source)
        return _caches[source]


def cached_search(query_vec, top_k: int, source: str | None) -> list[tuple[str, float]] | None:
    """Serve from memory when possible; None means the caller should use SQL."""
    from trialguard.config import settings

    if not settings.retrieval_vector_cache or not source:
        return None
    if source != settings.retrieval_vector_cache_source:
        return None
    return get_cache(source).search(query_vec, top_k)


def warm_in_background(source: str) -> threading.Thread | None:
    """Start loading without blocking startup. Searches use SQL until it lands."""
    from trialguard.config import settings

    if not settings.retrieval_vector_cache:
        return None
    thread = threading.Thread(
        target=get_cache(source).load, name="tg-vector-cache", daemon=True
    )
    thread.start()
    return thread


def status() -> dict:
    """Cache state for /api/health, so 'why is search slow' is answerable."""
    from trialguard.config import settings

    if not settings.retrieval_vector_cache:
        return {"enabled": False}
    cache = get_cache(settings.retrieval_vector_cache_source)
    return {
        "enabled": True,
        "source": cache.source,
        "ready": cache.ready,
        "rows": len(cache._ids),
        "load_seconds": round(cache.load_seconds, 1) if cache.load_seconds else None,
        "error": cache.error,
    }
