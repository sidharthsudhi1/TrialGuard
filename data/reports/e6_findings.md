# E6 — a latency distribution, and the capacity behind it

Measured 2026-09-23 against the deployed API (`trialguard-api.fly.dev`, `syd`,
vector matrix resident, 26,037 rows). Raw:
[`e6_latency_profile.json`](e6_latency_profile.json),
[`e6_coldpath.json`](e6_coldpath.json), [`e6_burst.json`](e6_burst.json).
Search costs no LLM call except keyword extraction, so this cost cents.

`production_readiness.md` says p95 is not claimed because a probe is n=1, and
`served_slo.json` gates on a single request. This replaces that.

## It cannot be a load test, and that is a finding

`api_search_rate_per_min` is **10 per IP**. From a single host the limiter binds
long before the server does, so a conventional load test would measure the rate
limiter. What one client can honestly measure is its own latency distribution,
plus capacity inferred from bursts that fit inside one minute's budget.

## Warm path: the 780 ms claim survives, with a shape

n=40, 6.5 s spacing, same preset note, 40/40 HTTP 200.

| | min | p50 | p90 | p95 | p99 / max | stdev |
|---|---|---|---|---|---|---|
| wall | 724.1 | **787.3** | 936.0 | **981.9** | 1106.3 | 84.7 |
| server (`total_ms`) | 636.9 | **703.8** | 756.2 | **859.3** | 1023.1 | 74.5 |

The standing "780 ms warm" is accurate as a median. **p95 is 981.9 ms wall and
859.3 ms server-side**, and the worst of 40 was 1106.3 ms. The SLO's 1500 ms
server-side bound holds with real headroom rather than by luck: p99 is 1023 ms.

Percentiles are nearest-rank, so every figure above is an observed request rather
than an interpolation.

## Cold path: what a real user meets is 14x worse

A user pasting their own note pays LLM keyword extraction first. n=5, genuinely
unseen notes.

| | min | p50 | max |
|---|---|---|---|
| wall | 3824.6 | **10973.0** | 19129.9 |
| `keyword_ms` | 3133.0 | **10295.8** | 18121.3 |

**Median 11.0 s, worst 19.1 s**, and 94% of it is keyword extraction. The retrieval
stage is unchanged — `fanout_ms` median 592.9 ms, right where the warm path sits.

So "780 ms" describes the demo's preset buttons. **The first search on a pasted
note is a 4-19 second wait**, and that number has never been reported. The UI
quotes cost and wait for Deep search but not for this.

> **The first attempt at this measured 13.7 ms and was wrong.** An earlier run of
> the harness crashed while summarising *after* issuing all 45 requests, which
> populated the keyword cache; the rerun then measured those same notes at
> `keyword_ms` 0.3 ms and reported a cold-path penalty of nothing. A cold-path
> note is single-use, and the fix was a second block of notes plus
> `--novel-offset`. The harness now also writes raw samples before summarising,
> so a crash in reporting cannot cost the run again.

## Capacity: concurrency buys almost nothing

Single bursts of k simultaneous requests, 65 s apart so the sliding window drains
between them. All 14 requests returned 200.

| | p50 latency | vs single | throughput |
|---|---|---|---|
| single | 787.3 ms | 1.00x | 1.27 req/s |
| c=2 | 1447.1 ms | 1.84x | 1.33 req/s |
| c=4 | 2798.3 ms | 3.55x | 1.43 req/s |
| c=8 | 5349.9 ms | 6.80x | **1.45 req/s** |

**Latency grows almost exactly linearly with concurrency while throughput is
flat at ~1.4 req/s.** The service is effectively serial for search: eight
concurrent callers are not served in parallel, they queue.

That is consistent with where the time goes. `fanout_ms` is ~590-690 ms of a
~705 ms request, and that stage already fans 12 keywords across threads doing
numpy matrix work and Postgres FTS. The cores are busy before a second request
arrives, so request-level concurrency contends with the fan-out rather than
filling idle capacity.

**Practical capacity is therefore ~1.4 searches/second, ~85/minute**, on the
current machine. Against a 10/min per-IP limit that is roughly **8 concurrent
users before the service saturates** — comfortable for a demo, and a number that
now exists rather than being assumed.

## What this does not claim

**Not a sustained load test.** Bursts are single, 65 s apart. Sustained
concurrency could behave differently, and could not be tested from one IP without
measuring the limiter.

**n=40 warm, n=5 cold, n=8 at the widest burst.** p99 on 40 samples is the
maximum by construction. The cold-path figures rest on five notes with a 5x
spread (3.8 s to 19.1 s), so the median is indicative, not tight.

**One client, one region, one time of day.** The client is in Australia and the
API is in `syd`, so network latency here is near-best-case: wall minus server is
about 84 ms. A user in Europe or North America pays materially more.

**Resume not measured here.** Health answered in 87 ms, so the machine was
already awake. WS-6a's 20,576 ms first-search-after-resume stands as the cold-start
figure and is not superseded.

## Consequence

Two numbers are worth acting on. The SLO can legitimately claim a p95, which it
previously could not. And **the cold-keyword path is the real user experience and
is 14x the advertised figure** — an 11 second median wait with no progress
indication implied anywhere in the current copy. Caching keywords for the demo
presets made the demo fast; it did not make the product fast.
