# S2: scale-out adds throughput, but not on the written bar

Run 2026-11-16/17 against `trialguard-api-staging` (Fly, syd, performance 2x,
4 GB, the production VM). Database is a Neon branch of production
(`staging`, 0.25 to 8 CU, the production range), 26,151 trials. Closed loop on
`/api/search`, the two preset notes (keywords cached, no LLM call), `top_k=5`,
20 s discarded and 90 s measured per step, every machine warmed by id first.
The load client never used more than 0.06 of a core. No machine stopped during
a step. Production was not touched. Raw: `s2_run1/`, `s2_run2/`.

The bar, written before any load: capacity(N) is the best successful req/s
among steps with at most 1% errors; capacity(2) ≥ 1.6x and capacity(4) ≥ 3.2x
of capacity(1); every machine serves at least 10% of a step.

## Run 1, the registered run: rejected

| machines | concurrency | ok req/s | errors | server p50 / p95 ms | least machine share |
|---|---|---|---|---|---|
| 1 | 4 | 5.567 | 0.0% | 636 / 809 | 1.00 |
| 1 | 8 | 5.967 | 2.4% | 1176 / 1555 | 1.00 |
| 2 | 8 | 8.044 | 0.0% | 733 / 1836 | 0.36 |
| 2 | 16 | 8.511 | 5.3% | 1274 / 3119 | 0.34 |
| 4 | 16 | 15.100 | 4.3% | 711 / 2133 | 0.16 |
| 4 | 32 | 16.678 | 16.5% | 1234 / 3186 | 0.15 |

Capacity(1) 5.567. Capacity(2) 8.044, which is 1.44x. Four machines have no
step under 1% errors. DONE-REJECTED.

Every failing step failed the same way: `psycopg2.pool.PoolError: connection
pool exhausted`, answered as HTTP 500. The pool holds 20 connections and
raises when they are out. Eight concurrent searches on one machine reach that.
Production runs the same code, so it has the same failure at the same load.

## Run 2, post-hoc, one change: no errors

`get_conn` now waits up to 30 s for a free slot instead of raising. The
amendment was written after run 1 and before run 2. Nothing else changed.

| machines | concurrency | ok req/s | errors | server p50 / p95 ms | least machine share |
|---|---|---|---|---|---|
| 4 | 16 | 14.889 | 0.0% | 755 / 2628 | 0.18 |
| 4 | 32 | 17.567 | 0.0% | 1282 / 4386 | 0.17 |
| 2 | 8 | 6.678 | 0.0% | 1075 / 1911 | 0.42 |
| 2 | 16 | 7.278 | 0.0% | 1955 / 3226 | 0.42 |
| 1 | 4 | 2.856 | 0.0% | 1313 / 1614 | 1.00 |
| 1 | 8 | 3.089 | 0.0% | 2475 / 2990 | 1.00 |

4,712 requests, 0 errors. On its own numbers run 2 clears both ratios (2.36x
and 5.69x). That is not a pass, and it is not reported as one.

## Why the ratios do not mean much

The machines are not equal. Run 2's single machine (`8ed97d9ad33598`) does
3.1 req/s. Run 1's single machine (`8edde1a7616158`) did 5.6. Same VM size,
same image apart from the pool change, same notes. In run 1's two-machine
step, before the fix, those two served 264 and 460 requests in 90 s, which is
2.9 and 5.1 req/s: each machine's solo rate. So the gap is the machine, not
the fix, and a ratio depends on which machine happens to be the baseline.
Against run 1's faster baseline, run 2 is 1.31x and 3.16x, under both bars.

Why one machine is slower was not established. It reports the same CPU model
string and low steal. The other three were destroyed before they could be
measured alone.

What the data does support:

- Throughput adds. Two machines served the sum of their solo rates.
- Four machines served 17.6 req/s with no errors, against 3.1 to 5.6 for one.
- The proxy spread is uneven (least share 0.15 to 0.42), with
  `soft_limit = 2` per-request balancing. Tail latency grows with it: p95 is
  3 to 4 times p50 on four machines.
- A single machine gains little from concurrency. The slow machine served 364
  ms single requests and 2.9 req/s at four in flight, which is serial.

## Decision

S2 is DONE-REJECTED on the registered run. "Near-linear" is not claimed.
Production stays one machine; the count is a user decision and nothing here
changes it.

The pool fix is adopted on its own evidence: it removes a 500 that production
can hit at eight concurrent searches. It is in this PR and is not deployed to
production.

Not measured: the cold keyword path under load, assess, more than four
machines, per-machine solo rates for all four, and Neon compute during the
run.
