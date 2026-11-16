# AD-42 — A database lease waits, and scale-out is not claimed linear

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

`get_conn` waits up to 30 s for a free pool slot instead of letting psycopg2 raise. On a staging app with the production VM, eight concurrent searches on one machine exhausted the 20-connection pool and the API answered 500 (2.4% of requests at one machine, 16.5% at four machines and 32 in flight). With the wait, 4,712 requests over the same steps returned no error.

Horizontal scale is not adopted as "near-linear". The registered run gave 1.44x at two machines against a 1.6x bar and had no error-free step at four. A post-hoc rerun after the fix reached 17.6 req/s on four machines, but identical machines differed by about 1.8x in solo rate (3.1 and 5.6 req/s), so the ratio depends on the baseline machine. Production stays at one machine. [`s2_findings.md`](../../data/reports/s2_findings.md).

## Alternatives considered

- **Raise the pool ceiling.** It moves the failure to a higher load and spends Neon connections; waiting bounds it.
- **Report the rerun's 5.69x as a pass.** Its baseline was the slowest machine seen, and the rerun was written after the first result.
