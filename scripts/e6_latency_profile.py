"""E6: a latency distribution for the served API, to replace an n=1 claim.

`served_slo.json` gates search at 1500 ms server-side off a probe that issues one
request, and `production_readiness.md` says plainly that p95 is not claimed
because a probe is n=1. This measures the distribution instead.

It is not a load test, and cannot be one from a single host: `api_search_rate_per_min`
is 10 per IP, so the limiter binds long before the server does. Attempting it
anyway would measure the rate limiter and, on a shared daily budget, could take
the live demo dark. What is measurable from one client is what one client
actually experiences, which is the number the SLO is about.

Two paths, because they are different systems:

- **warm**: the same preset note every time. Its keywords are cached, so this is
  the retrieval path alone -- what "780 ms warm" refers to.
- **cold-keyword**: a note never seen before, which pays an LLM keyword
  extraction first. This is what a user pasting their own note meets, and it has
  never been reported separately.

    python scripts/e6_latency_profile.py --n 40 --novel 5
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

DEFAULT_URL = "https://trialguard-api.fly.dev"

# Synthetic, and varied enough that no two share a keyword-cache entry. Held here
# rather than generated so a rerun measures the same notes.
NOVEL_NOTES = [
    "62-year-old man with metastatic castration-resistant prostate cancer, prior "
    "docetaxel, ECOG 1, rising PSA on abiraterone. Adequate renal function.",
    "54-year-old woman with relapsed follicular lymphoma, two prior lines including "
    "rituximab, ECOG 0, no CNS involvement, normal LVEF.",
    "47-year-old man with unresectable hepatocellular carcinoma, Child-Pugh A, "
    "ECOG 1, hepatitis B controlled on entecavir, no prior systemic therapy.",
    "38-year-old woman with triple-negative breast cancer, neoadjuvant "
    "chemotherapy completed, residual disease at surgery, ECOG 0, BRCA wild-type.",
    "70-year-old man with newly diagnosed acute myeloid leukemia, unfit for "
    "intensive induction, ECOG 2, normal cytogenetics, no prior treatment.",
    # A second block, because the first was burned: an earlier run of this script
    # crashed while summarising, after having already issued every request. That
    # populated the keyword cache, so the notes above are no longer cold and a
    # rerun measured keyword_ms at 0.3 ms. A cold-path note is single-use.
    "59-year-old woman with platinum-resistant ovarian cancer, three prior lines, "
    "ECOG 1, no bowel obstruction, CA-125 rising.",
    "66-year-old man with locally advanced pancreatic adenocarcinoma, ECOG 1, "
    "biliary stent in place, no prior radiotherapy, bilirubin normalised.",
    "41-year-old woman with recurrent glioblastoma, prior temozolomide and "
    "radiotherapy, KPS 80, on stable dexamethasone dose.",
    "73-year-old man with metastatic renal cell carcinoma, clear cell histology, "
    "prior nivolumab, ECOG 1, no brain metastases.",
    "35-year-old man with refractory Hodgkin lymphoma, prior ABVD and brentuximab, "
    "ECOG 0, adequate marrow reserve, no active infection.",
]


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    # Nearest-rank (ceil), so a reported percentile is always an observed value
    # rather than an interpolation between two of them. At n=40 that distinction
    # is the difference between a real measurement and a smoothed one.
    k = max(0, min(len(xs) - 1, math.ceil(p / 100 * len(xs)) - 1))
    return round(xs[k], 1)


def _summary(wall: list[float], server: list[float], label: str, n_req: int) -> dict:
    out: dict = {"label": label, "requests": n_req, "ok": len(wall)}
    for name, xs in (("wall_ms", wall), ("server_ms", [s for s in server if s is not None])):
        if not xs:
            continue
        out[name] = {
            "min": round(min(xs), 1),
            "p50": _pct(xs, 50),
            "p90": _pct(xs, 90),
            "p95": _pct(xs, 95),
            "p99": _pct(xs, 99),
            "max": round(max(xs), 1),
            "mean": round(st.mean(xs), 1),
            "stdev": round(st.stdev(xs), 1) if len(xs) > 1 else 0.0,
        }
    return out


def _search(client, base_url: str, note: str, top_k: int):
    t0 = time.perf_counter()
    r = client.post(f"{base_url}/api/search", json={"note": note, "top_k": top_k}, timeout=120)
    wall = (time.perf_counter() - t0) * 1000
    server = None
    stages: dict = {}
    if r.status_code == 200:
        body = r.json()
        lat = body.get("latency_ms") or {}
        # latency_ms is a per-stage breakdown, not a scalar; total_ms is the
        # server-side figure the SLO gates on.
        server = lat.get("total_ms") if isinstance(lat, dict) else lat
        stages = lat if isinstance(lat, dict) else {}
    return wall, server, r.status_code, stages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_URL)
    ap.add_argument("--n", type=int, default=40, help="warm samples")
    ap.add_argument("--novel", type=int, default=5, help="cold-keyword samples")
    ap.add_argument("--novel-offset", type=int, default=0,
                    help="skip this many notes; a cold-path note is single-use")
    ap.add_argument("--skip-warm", action="store_true")
    ap.add_argument("--burst", default="", help="comma widths, e.g. 2,4,8")
    ap.add_argument("--burst-pause", type=float, default=65.0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--spacing", type=float, default=6.5,
                    help="seconds between requests; must keep under api_search_rate_per_min")
    ap.add_argument("--out", default="data/reports/e6_latency_profile.json")
    args = ap.parse_args()

    import httpx

    client = httpx.Client()
    base = args.base_url.rstrip("/")

    # Resume the machine before timing anything. fly.toml suspends on idle, and
    # the first request after a resume measured 20,576 ms in WS-6a -- a real
    # number, but not one that belongs in a steady-state distribution.
    t0 = time.perf_counter()
    health = client.get(f"{base}/api/health", timeout=120)
    resume_ms = (time.perf_counter() - t0) * 1000
    health_body = health.json() if health.status_code == 200 else {}

    preset = client.get(f"{base}/api/limits", timeout=60).json()["presets"][0]["note"]

    # One unmeasured call so the keyword cache and vector matrix are both hot.
    _search(client, base, preset, args.top_k)

    warm_stages: list[dict] = []
    warm_wall: list[float] = []
    warm_server: list[float] = []
    statuses: dict[str, int] = {}
    for i in range(0 if not args.skip_warm else args.n, args.n):
        w, s, code, st_ = _search(client, base, preset, args.top_k)
        if st_:
            warm_stages.append(st_)
        statuses[str(code)] = statuses.get(str(code), 0) + 1
        if code == 200:
            warm_wall.append(w)
            warm_server.append(s)
        if i < args.n - 1:
            time.sleep(args.spacing)

    cold_stages: list[dict] = []
    cold_wall: list[float] = []
    cold_server: list[float] = []
    for i, note in enumerate(NOVEL_NOTES[args.novel_offset : args.novel_offset + args.novel]):
        w, s, code, st_ = _search(client, base, note, args.top_k)
        if st_:
            cold_stages.append(st_)
        statuses[str(code)] = statuses.get(str(code), 0) + 1
        if code == 200:
            cold_wall.append(w)
            cold_server.append(s)
        if i < args.novel - 1:
            time.sleep(args.spacing)

    raw = {"warm_wall": warm_wall, "warm_server": warm_server,
           "cold_wall": cold_wall, "cold_server": cold_server,
           "warm_stages": warm_stages, "cold_stages": cold_stages}
    raw_path = Path(args.out).with_name(Path(args.out).stem + "_raw.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, indent=2))

    result = {
        "base_url": base,
        "top_k": args.top_k,
        "spacing_s": args.spacing,
        "rate_limit_note": "api_search_rate_per_min=10 per IP; this is a single-client "
                           "latency distribution, not a throughput test",
        "resume_health_ms": round(resume_ms, 1),
        "vector_cache": health_body.get("vector_cache", {}),
        "statuses": statuses,
        "warm": _summary(warm_wall, warm_server, "warm (cached keywords)", args.n),
        "cold_keyword": _summary(cold_wall, cold_server, "novel note (LLM keyword extraction)",
                                 args.novel),
    }
    if args.burst:
        widths = [int(x) for x in args.burst.split(",")]
        result["burst"] = burst(base, preset, args.top_k, widths, args.burst_pause)

    if warm_wall and cold_wall:
        result["cold_keyword_penalty_ms"] = round(st.median(cold_wall) - st.median(warm_wall), 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))




def burst(base_url: str, note: str, top_k: int, widths: list[int], pause: float) -> list[dict]:
    """Concurrency probe that stays inside the per-IP budget.

    A throughput test is impossible from one host (10/min), but a single burst of
    k simultaneous requests fits inside one minute's budget and still shows
    whether concurrent callers contend. Each burst is followed by a pause so the
    sliding window drains before the next one, so this measures the server rather
    than the limiter.
    """
    import concurrent.futures as cf

    import httpx

    out = []
    for k in widths:
        with httpx.Client() as c:
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(max_workers=k) as pool:
                res = list(pool.map(lambda _: _search(c, base_url, note, top_k), range(k)))
            span = (time.perf_counter() - t0) * 1000
        ok = [w for w, _s, code, _st in res if code == 200]
        codes: dict[str, int] = {}
        for _w, _s, code, _st in res:
            codes[str(code)] = codes.get(str(code), 0) + 1
        out.append({
            "concurrency": k,
            "wall_span_ms": round(span, 1),
            "ok": len(ok),
            "statuses": codes,
            "p50_ms": _pct(ok, 50),
            "max_ms": round(max(ok), 1) if ok else None,
        })
        time.sleep(pause)
    return out


if __name__ == "__main__":
    main()
