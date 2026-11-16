"""S2: closed-loop load on the staging app, one step per concurrency.

Refuses the production host. Sends only preset notes, whose keywords are cached,
so a run costs no LLM call.

    TG_STAGING_TOKEN=... python scripts/s2_load.py --machines 2 --concurrency 8,16
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

STAGING = "https://trialguard-api-staging.fly.dev"
PRODUCTION_HOST = "trialguard-api.fly.dev"


def _pct(xs: list[float], q: float) -> float:
    ordered = sorted(xs)
    rank = max(1, -(-len(ordered) * q // 100))
    return round(ordered[int(rank) - 1], 1)


def step(base: str, token: str, notes: list[str], width: int, warmup: float, measure: float,
         top_k: int) -> dict:
    import httpx

    rows: list[tuple[float, float, int, str, float | None]] = []
    lock = threading.Lock()
    start = time.perf_counter()
    stop_at = start + warmup + measure

    def worker(seed: int) -> None:
        i = seed
        with httpx.Client(headers={"X-Staging-Token": token}, timeout=60) as client:
            while time.perf_counter() < stop_at:
                note = notes[i % len(notes)]
                i += 1
                t0 = time.perf_counter()
                try:
                    r = client.post(f"{base}/api/search", json={"note": note, "top_k": top_k})
                    code = r.status_code
                    machine = r.headers.get("x-served-by", "")
                    server = None
                    if code == 200:
                        server = (r.json().get("latency_ms") or {}).get("total_ms")
                except httpx.HTTPError:
                    code, machine, server = 0, "", None
                t1 = time.perf_counter()
                with lock:
                    rows.append((t0 - start, (t1 - t0) * 1000, code, machine, server))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(width)]
    cpu0 = time.process_time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    client_cpu = (time.process_time() - cpu0) / (warmup + measure)

    kept = [r for r in rows if warmup <= r[0] < warmup + measure]
    ok = [r for r in kept if r[2] == 200]
    wall = [r[1] for r in ok]
    server = [r[4] for r in ok if r[4] is not None]
    by_machine = Counter(r[3] for r in ok)
    return {
        "concurrency": width,
        "measured_s": measure,
        "requests": len(kept),
        "ok": len(ok),
        "error_rate": round(1 - len(ok) / len(kept), 4) if kept else None,
        "statuses": dict(Counter(str(r[2]) for r in kept)),
        "rps_ok": round(len(ok) / measure, 3),
        "wall_ms": {"p50": _pct(wall, 50), "p95": _pct(wall, 95)} if wall else {},
        "server_ms": {"p50": _pct(server, 50), "p95": _pct(server, 95)} if server else {},
        "by_machine": dict(by_machine),
        "min_machine_share": round(min(by_machine.values()) / len(ok), 4) if ok else None,
        "client_cpu_cores": round(client_cpu, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=STAGING)
    ap.add_argument("--machines", type=int, required=True, help="started machines, for the label")
    ap.add_argument("--machine-ids", default="", help="comma ids to warm one by one")
    ap.add_argument("--concurrency", required=True, help="comma widths")
    ap.add_argument("--warmup", type=float, default=20.0)
    ap.add_argument("--measure", type=float, default=90.0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import httpx

    base = args.base_url.rstrip("/")
    if PRODUCTION_HOST in base:
        raise SystemExit("refusing to load the production app")
    token = os.environ.get("TG_STAGING_TOKEN", "")
    if not token:
        raise SystemExit("TG_STAGING_TOKEN is not set")

    with httpx.Client(headers={"X-Staging-Token": token}, timeout=180) as client:
        notes = [p["note"] for p in client.get(f"{base}/api/limits").json()["presets"]]
        # Every machine loads its own matrix on its first search. Warm each one by
        # id so the measured window is steady state on all of them.
        for mid in [m for m in args.machine_ids.split(",") if m]:
            for note in notes:
                for _ in range(2):
                    r = client.post(
                        f"{base}/api/search",
                        json={"note": note, "top_k": args.top_k},
                        headers={"fly-force-instance-id": mid},
                    )
                    r.raise_for_status()
        health = client.get(f"{base}/api/health").json()

    steps = []
    for width in [int(x) for x in args.concurrency.split(",")]:
        result = step(base, token, notes, width, args.warmup, args.measure, args.top_k)
        print(json.dumps(result), flush=True)
        steps.append(result)
        time.sleep(10)

    out = Path(args.out or f"data/reports/s2_load_n{args.machines}.json")
    out.write_text(
        json.dumps(
            {
                "base_url": base,
                "machines": args.machines,
                "top_k": args.top_k,
                "n_notes": len(notes),
                "vector_cache": health.get("vector_cache", {}),
                "steps": steps,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
