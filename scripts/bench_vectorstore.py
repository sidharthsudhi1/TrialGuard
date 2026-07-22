"""Phase 5 WS-5: vector-store benchmark — pgvector (production) vs local exact.

Runs over the cached MedCPT eval embeddings (data/indexes/*.npy), so the local
backends cost $0 and no re-embedding. Reports index build time, query latency
p50/p95, and recall parity vs exact brute force. The pgvector arm (opt-in with
--pgvector) loads the vectors into an isolated temp table on the configured Neon
DB, measures ivfflat query latency, and drops the table — a real production-backend
number without touching the trials table.

The decision axis behind AD-6 is memory, not speed: the free-tier size ceiling is
projected at the end. Full-corpus (26k TREC) pgvector cannot be measured on the
free tier by construction — that is the finding, not a gap.

    python scripts/bench_vectorstore.py --source sigir --n-queries 100 --k 50
    python scripts/bench_vectorstore.py --pgvector        # add the live pgvector arm
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

INDEX_DIR = Path("data/indexes")
DIM = 768


def load_embeddings(source: str, tag: str):
    emb = np.load(INDEX_DIR / f"{source}_{tag}_embeddings.npy").astype("float32")
    ids = json.loads((INDEX_DIR / f"{source}_{tag}_ids.json").read_text())
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)  # cosine == dot
    return emb, ids


def _pct(latencies: list[float], p: float) -> float:
    return round(float(np.percentile(latencies, p)) * 1000, 2)  # ms


def _recall_vs_exact(got: list[list[int]], exact: list[list[int]], k: int) -> float:
    overlaps = [len(set(g) & set(e)) / k for g, e in zip(got, exact)]
    return round(float(np.mean(overlaps)), 4)


def bench_numpy(emb, queries, k):
    lat, results = [], []
    for q in queries:
        t0 = time.perf_counter()
        scores = emb @ q
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        lat.append(time.perf_counter() - t0)
        results.append(top.tolist())
    return {
        "backend": "numpy_brute (FileIndex, exact)",
        "build_s": 0.0,
        "p50_ms": _pct(lat, 50),
        "p95_ms": _pct(lat, 95),
        "mem_mb": round(emb.nbytes / 1e6, 2),
    }, results


def bench_sklearn(emb, queries, k):
    from sklearn.neighbors import NearestNeighbors

    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(emb)
    build = time.perf_counter() - t0
    lat, results = [], []
    for q in queries:
        t0 = time.perf_counter()
        _, idx = nn.kneighbors(q.reshape(1, -1))
        lat.append(time.perf_counter() - t0)
        results.append(idx[0].tolist())
    return {
        "backend": "sklearn_brute (exact)",
        "build_s": round(build, 4),
        "p50_ms": _pct(lat, 50),
        "p95_ms": _pct(lat, 95),
        "mem_mb": round(emb.nbytes / 1e6, 2),
    }, results


def _vec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def bench_pgvector(emb, queries, k):
    from trialguard.db.schema import get_conn

    lat, results = [], []
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS bench_vectors")
        cur.execute(f"CREATE TABLE bench_vectors (id int PRIMARY KEY, embedding vector({DIM}))")
        t0 = time.perf_counter()
        cur.executemany(
            "INSERT INTO bench_vectors (id, embedding) VALUES (%s, %s::vector)",
            [(i, _vec_literal(e)) for i, e in enumerate(emb)],
        )
        lists = max(1, int(len(emb) ** 0.5))
        cur.execute(
            f"CREATE INDEX ON bench_vectors USING ivfflat (embedding vector_cosine_ops) "
            f"WITH (lists = {lists})"
        )
        conn.commit()
        build = time.perf_counter() - t0
        for q in queries:
            lit = _vec_literal(q)
            t0 = time.perf_counter()
            cur.execute(
                "SELECT id FROM bench_vectors ORDER BY embedding <=> %s::vector LIMIT %s",
                (lit, k),
            )
            results.append([r[0] for r in cur.fetchall()])
            lat.append(time.perf_counter() - t0)
        cur.execute("DROP TABLE bench_vectors")
        conn.commit()
    return {
        "backend": f"pgvector ivfflat lists={lists} (Neon free tier, approx)",
        "build_s": round(build, 4),
        "p50_ms": _pct(lat, 50),
        "p95_ms": _pct(lat, 95),
        "mem_mb": round(emb.nbytes / 1e6, 2),
    }, results


def size_ceiling() -> list[dict]:
    """Vectors-only footprint per corpus size; the AD-6 decision axis."""
    per_vec = DIM * 4  # float32
    rows = [
        ("SIGIR eval", 2_991),
        ("TREC 2021/2022", 26_000),
        ("scoped oncology prod", 50_000),
        ("full ClinicalTrials.gov", 500_000),
    ]
    return [
        {"corpus": name, "n": n, "vectors_mb": round(n * per_vec / 1e6, 1)}
        for name, n in rows
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sigir")
    ap.add_argument("--tag", default="medcpt_excl")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--pgvector", action="store_true", help="add the live pgvector arm")
    args = ap.parse_args()

    emb, ids = load_embeddings(args.source, args.tag)
    rng = np.random.default_rng(0)
    qidx = rng.choice(len(emb), size=min(args.n_queries, len(emb)), replace=False)
    queries = emb[qidx]

    exact_stats, exact_res = bench_numpy(emb, queries, args.k)
    exact_stats["recall_vs_exact"] = 1.0
    arms = [exact_stats]

    sk_stats, sk_res = bench_sklearn(emb, queries, args.k)
    sk_stats["recall_vs_exact"] = _recall_vs_exact(sk_res, exact_res, args.k)
    arms.append(sk_stats)

    if args.pgvector:
        pg_stats, pg_res = bench_pgvector(emb, queries, args.k)
        pg_stats["recall_vs_exact"] = _recall_vs_exact(pg_res, exact_res, args.k)
        arms.append(pg_stats)

    out = {
        "source": args.source,
        "tag": args.tag,
        "n_vectors": len(emb),
        "dim": DIM,
        "n_queries": len(queries),
        "k": args.k,
        "arms": arms,
        "size_ceiling": size_ceiling(),
        "neon_free_tier_mb": 512,
    }
    print(json.dumps(out, indent=2))
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    Path("data/reports/phase5_vectorstore.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
