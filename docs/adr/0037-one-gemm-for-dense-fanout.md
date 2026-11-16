# AD-37 — Dense fan-out is one GEMM, and the matrix stays float32

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

The keyword dense search computes every query vector in one `(N, D) @ (D, K)` product. On the TREC 2021 matrix (26149 x 768, 12 queries) the top-500 id sets match the per-query product, concurrency-1 p95 falls from 56.1 ms to 18.0 ms, and eight concurrent searches finish 2.54x sooner. Float16 storage does not keep that top-500 set, so the resident matrix stays float32. [`s1_findings.md`](../../data/reports/s1_findings.md).

## Alternatives considered

- **Leave the per-query product.** It is the same dots with a launch per keyword, and it loses at both concurrencies measured.
- **Float16 to cut bandwidth.** Rejected on the set-equality constraint.
- **Cap BLAS at one thread.** Slightly faster at concurrency 1 in this bench, and not required for the 2x bar. The default stays uncapped.
