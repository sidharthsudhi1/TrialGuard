"""E1a: where does the in-process dense matrix stop being the right answer?

AD-6 (Phase 9) moved dense search out of pgvector into an 80 MB in-process matrix
because the planner declined ivfflat at 26k rows and seq-scanned anyway. That
reasoning is a function of corpus size and it inverts somewhere above it. This
measures where, across exact-matrix / ivfflat / HNSW, on three axes: recall vs
exact, query latency, and resident memory.

Corpora are synthesised from real MedCPT vectors rather than sampled uniformly.
Uniform random vectors have no cluster structure, which is the pathological worst
case for ANN — the latency curve would transfer and the recall curve would not,
and recall is the axis the decision turns on.

The corpus lives in a memmap and exact ground truth is computed in chunks, so a
5M-row run does not need 15 GB of RAM to produce its baseline. The matrix arm is
skipped (not failed) above --ram-budget-gb; that skip is the finding.

    # local docker postgres with pgvector, never prod Neon
    python scripts/bench_ann_scale.py --dsn postgresql://postgres:postgres@localhost:5432/postgres \
        --sizes 100000,500000,1000000 --k 100 --n-queries 200
"""

from __future__ import annotations

import argparse
import io
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

INDEX_DIR = Path("data/indexes")
DIM = 768


def _peak_rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux. A magnitude heuristic
    # cannot tell 229 MB on macOS from 229 GB on Linux, so ask the platform.
    per_gb = 1e9 if sys.platform == "darwin" else 1e6
    return round(usage / per_gb, 3)


def _pct(latencies: list[float], p: float) -> float:
    return round(float(np.percentile(latencies, p)) * 1000, 2)


def _recall_vs_exact(got: list[list[int]], exact: list[list[int]], k: int) -> float:
    overlaps = [len(set(g) & set(e)) / k for g, e in zip(got, exact)]
    return round(float(np.mean(overlaps)), 4)


def _normalise(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def load_base(source: str, tag: str) -> np.ndarray:
    emb = np.load(INDEX_DIR / f"{source}_{tag}_embeddings.npy").astype("float32")
    return _normalise(emb)


def synthesise(base: np.ndarray, n: int, sigma: float, path: Path, seed: int) -> np.ndarray:
    """Grow `base` to n rows on disk, preserving its cluster structure.

    Resample with replacement and perturb, so the neighbour graph stays as dense
    and as clumpy as the real corpus.

    sigma is the perturbation norm *relative to the vector*, not a per-dimension
    standard deviation. In 768 dimensions a per-dim sigma of 0.05 carries a norm
    of 0.05*sqrt(768) = 1.39 — larger than the unit vector it perturbs, which
    erases the cluster structure and yields the near-uniform corpus this harness
    exists to avoid. Dividing by sqrt(DIM) makes sigma mean what it reads as.
    """
    rng = np.random.default_rng(seed)
    per_dim = sigma / np.sqrt(DIM)
    out = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(n, DIM))
    step = 100_000
    for start in range(0, n, step):
        size = min(step, n - start)
        picks = rng.integers(0, len(base), size=size)
        block = base[picks] + rng.normal(0, per_dim, size=(size, DIM)).astype("float32")
        out[start : start + size] = _normalise(block)
    out.flush()
    return out


def tie_tolerant_recall(
    corpus: np.ndarray, queries: np.ndarray, got: list[list[int]],
    kth_score: np.ndarray, eps: float = 1e-5,
) -> float:
    """Recall that does not punish an index for picking an equally-close neighbour.

    Resampling with replacement puts near-duplicates in the corpus, so the true
    top-k contains ties and exact-id overlap under-counts a correct index. A
    retrieved vector counts when it scores at least as well as the k-th exact
    hit, which is the property a ranker actually owes.
    """
    hits = []
    for qi, ids in enumerate(got):
        scores = np.asarray(corpus[ids]) @ queries[qi]
        hits.append(float(np.mean(scores >= kth_score[qi] - eps)))
    return round(float(np.mean(hits)), 4)


def exact_topk(
    corpus: np.ndarray, queries: np.ndarray, k: int, step: int = 200_000
) -> tuple[list[list[int]], np.ndarray, dict]:
    """Ground truth by chunked scan, so RAM stays flat in corpus size."""
    n_q = len(queries)
    best_scores = np.full((n_q, k), -np.inf, dtype="float32")
    best_idx = np.zeros((n_q, k), dtype="int64")
    t0 = time.perf_counter()
    for start in range(0, len(corpus), step):
        block = np.asarray(corpus[start : start + step])
        scores = queries @ block.T
        block_ids = np.tile(np.arange(start, start + len(block), dtype="int64"), (n_q, 1))
        merged_s = np.hstack([best_scores, scores])
        merged_i = np.hstack([best_idx, block_ids])
        top = np.argpartition(merged_s, -k, axis=1)[:, -k:]
        rows = np.arange(n_q)[:, None]
        best_scores = merged_s[rows, top]
        best_idx = merged_i[rows, top]
    order = np.argsort(best_scores, axis=1)[:, ::-1]
    rows = np.arange(n_q)[:, None]
    results = best_idx[rows, order].tolist()
    kth_score = best_scores[rows, order][:, -1].copy()
    return results, kth_score, {"scan_s": round(time.perf_counter() - t0, 3)}


def bench_matrix(corpus: np.ndarray, queries: np.ndarray, k: int, budget_gb: float) -> dict:
    """The AD-6 arm: whole corpus resident, exact search, no index."""
    need_gb = corpus.nbytes / 1e9
    if need_gb > budget_gb:
        return {
            "backend": "in-process matrix (exact)",
            "skipped": f"needs {need_gb:.2f} GB, budget {budget_gb} GB",
            "resident_gb": round(need_gb, 3),
        }
    t0 = time.perf_counter()
    resident = np.array(corpus)
    load_s = time.perf_counter() - t0
    lat = []
    for q in queries:
        t0 = time.perf_counter()
        scores = resident @ q
        top = np.argpartition(scores, -k)[-k:]
        _ = top[np.argsort(scores[top])[::-1]]
        lat.append(time.perf_counter() - t0)
    stats = {
        "backend": "in-process matrix (exact)",
        "build_s": round(load_s, 3),
        "p50_ms": _pct(lat, 50),
        "p95_ms": _pct(lat, 95),
        "recall_vs_exact": 1.0,
        "recall_tie_tolerant": 1.0,
        "resident_gb": round(resident.nbytes / 1e9, 3),
    }
    del resident
    return stats


def _copy_load(cur, corpus: np.ndarray, table: str) -> float:
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    cur.execute(f"CREATE TABLE {table} (id int, embedding vector({DIM}))")
    t0 = time.perf_counter()
    step = 50_000
    for start in range(0, len(corpus), step):
        block = np.asarray(corpus[start : start + step])
        buf = io.StringIO()
        for i, vec in enumerate(block):
            buf.write(f"{start + i}\t[{','.join(f'{x:.6f}' for x in vec)}]\n")
        buf.seek(0)
        cur.copy_from(buf, table, columns=("id", "embedding"))
    return time.perf_counter() - t0


def _query_pg(
    cur, table: str, queries: np.ndarray, k: int, warmup: int = 20
) -> tuple[list[list[int]], list[float]]:
    """Warm the page cache before timing.

    The first arm queried after an index build otherwise pays the whole warm-up
    and reads as the slowest configuration when it is merely the first — at 1M
    that made ef_search=40 look 38x slower than ef_search=200.
    """
    for q in queries[:warmup]:
        lit = "[" + ",".join(f"{x:.6f}" for x in q) + "]"
        cur.execute(
            f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s", (lit, k)
        )
        cur.fetchall()

    lat, results = [], []
    for q in queries:
        lit = "[" + ",".join(f"{x:.6f}" for x in q) + "]"
        t0 = time.perf_counter()
        cur.execute(
            f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s", (lit, k)
        )
        results.append([r[0] for r in cur.fetchall()])
        lat.append(time.perf_counter() - t0)
    return results, lat


def bench_pgvector(
    dsn: str, corpus: np.ndarray, queries: np.ndarray, k: int, exact: list[list[int]],
    kth_score: np.ndarray, probes: list[int], ef_search: list[int],
) -> list[dict]:
    import psycopg2

    table = "bench_ann"
    arms: list[dict] = []
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        load_s = _copy_load(cur, corpus, table)
        conn.commit()
        cur.execute(f"SELECT pg_table_size('{table}')")
        heap_gb = round(cur.fetchone()[0] / 1e9, 3)

        lists = max(1, int(len(corpus) ** 0.5))
        t0 = time.perf_counter()
        cur.execute(
            f"CREATE INDEX bench_ivf ON {table} "
            f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
        )
        conn.commit()
        ivf_build = time.perf_counter() - t0
        cur.execute("SELECT pg_relation_size('bench_ivf')")
        ivf_gb = round(cur.fetchone()[0] / 1e9, 3)
        for p in probes:
            cur.execute(f"SET ivfflat.probes = {p}")
            res, lat = _query_pg(cur, table, queries, k)
            arms.append({
                "backend": f"pgvector ivfflat lists={lists} probes={p}",
                "load_s": round(load_s, 2),
                "build_s": round(ivf_build, 2),
                "p50_ms": _pct(lat, 50),
                "p95_ms": _pct(lat, 95),
                "recall_vs_exact": _recall_vs_exact(res, exact, k),
                "recall_tie_tolerant": tie_tolerant_recall(corpus, queries, res, kth_score),
                "heap_gb": heap_gb,
                "index_gb": ivf_gb,
            })
        cur.execute("DROP INDEX bench_ivf")
        conn.commit()

        t0 = time.perf_counter()
        cur.execute(
            f"CREATE INDEX bench_hnsw ON {table} "
            f"USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )
        conn.commit()
        hnsw_build = time.perf_counter() - t0
        cur.execute("SELECT pg_relation_size('bench_hnsw')")
        hnsw_gb = round(cur.fetchone()[0] / 1e9, 3)
        for ef in [e for e in ef_search if e >= k]:
            cur.execute(f"SET hnsw.ef_search = {ef}")
            res, lat = _query_pg(cur, table, queries, k)
            arms.append({
                "backend": f"pgvector hnsw m=16 ef_construction=64 ef_search={ef}",
                "load_s": round(load_s, 2),
                "build_s": round(hnsw_build, 2),
                "p50_ms": _pct(lat, 50),
                "p95_ms": _pct(lat, 95),
                "recall_vs_exact": _recall_vs_exact(res, exact, k),
                "recall_tie_tolerant": tie_tolerant_recall(corpus, queries, res, kth_score),
                "heap_gb": heap_gb,
                "index_gb": hnsw_gb,
            })

        cur.execute(f"DROP TABLE {table}")
        conn.commit()
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="trec_2021")
    ap.add_argument("--tag", default="medcpt_excl")
    ap.add_argument("--sizes", default="100000,500000,1000000")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--n-queries", type=int, default=200)
    # Calibrated, not chosen: at 0.15 the synthetic corpus reproduces the real
    # TREC MedCPT mean top-1 neighbour cosine (0.8573 vs 0.8548). See e1a_findings.
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--ram-budget-gb", type=float, default=8.0)
    ap.add_argument("--dsn", help="pgvector DSN; omit to run matrix arm only")
    ap.add_argument("--probes", default="10,40,100")
    ap.add_argument("--ef-search", default="100,200,400")
    ap.add_argument("--work-dir", default="/tmp/e1a")
    ap.add_argument("--out", default="data/reports/e1a_ann_scale.json")
    args = ap.parse_args()

    base = load_base(args.source, args.tag)
    rng = np.random.default_rng(0)
    qidx = rng.choice(len(base), size=min(args.n_queries, len(base)), replace=False)
    queries = base[qidx]

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(",")]
    probes = [int(p) for p in args.probes.split(",")]
    ef_search = [int(e) for e in args.ef_search.split(",")]

    meta = {"base_n": len(base), "n_queries": len(queries)}
    runs = []
    for n in sizes:
        path = work / f"corpus_{n}.npy"
        t0 = time.perf_counter()
        corpus = synthesise(base, n, args.sigma, path, seed=n)
        synth_s = time.perf_counter() - t0

        exact, kth_score, exact_meta = exact_topk(corpus, queries, args.k)
        arms = [bench_matrix(corpus, queries, args.k, args.ram_budget_gb)]
        if args.dsn:
            arms.extend(
                bench_pgvector(
                    args.dsn, corpus, queries, args.k, exact, kth_score, probes, ef_search
                )
            )

        runs.append({
            "n_vectors": n,
            "vectors_gb": round(n * DIM * 4 / 1e9, 3),
            "synthesise_s": round(synth_s, 2),
            "exact_scan_s": exact_meta["scan_s"],
            "arms": arms,
        })
        print(json.dumps(runs[-1], indent=2), flush=True)
        del corpus
        path.unlink(missing_ok=True)
        _write(args, runs, meta)  # a later size crashing must not cost the earlier ones

    _write(args, runs, meta)


def _write(args, runs: list[dict], meta: dict) -> None:
    out = {
        "base_source": args.source,
        **meta,
        "dim": DIM,
        "k": args.k,
        "sigma": args.sigma,
        "ram_budget_gb": args.ram_budget_gb,
        "peak_rss_gb": _peak_rss_gb(),
        "runs": runs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
