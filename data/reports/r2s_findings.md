# R2S: the R2 cross-encoder is too slow to serve

Run 2026-11-18. Registered in `r2s_registration.md` before any variant was timed.
Stopped at Stage A, as registered.

## Measured on the serving hardware

Fly performance-2x (2 vCPU, 4 GB), one-off machines in the API image, no traffic.
MedCPT loaded first, as in serving.

First probe, synthetic docs at the 512-token cap, 500 candidates, fp32 (R2 as
measured), one query each:

| Batch | Peak RSS (probe process) | Wall time |
|---|---|---|
| 128 (code on main) | 3,281 MB | 476 s |
| 32 | 1,658 MB | 451 s |
| 16 | 1,530 MB | 438 s |

The production API process idles at ~940 MB with 2.9 GB available and no swap,
so batch 128 would risk an out-of-memory kill. Batch 16 fits.

Registered Stage A, 300 real `ctgov_live` trials, 3 keyword queries:

| Variant | Weights | Max tokens | Candidates | Median wall time | Peak RSS |
|---|---|---|---|---|---|
| Q256-300 | int8 dynamic | 256 | 300 | **95.0 s** (95.0, 95.3, 93.4) | 1,651 MB |

Bar: <= 30 s and <= 2.2 GB. Q256-300 does the least work of the four variants, so
Q256, Q512 and F512 cannot pass either. They were not run. 167 of 300 real trial
texts hit the 256-token cap, so real trials are long, and trimming them changes
what the model reads.

## Why R2 said 7-8 s

R2 scored on a c7i.8xlarge: 16 physical cores, bf16 with AMX. That is about
60-70 pairs/s. The serving machine does about 1 pair/s in fp32 at 512 tokens and
about 3 pairs/s with int8 at 256. The 7-8 s figure was right for that box and
never true for production. It was carried into `config.py` and the R2 end-to-end
recommendation as a serving cost.

## Decision

Stop, per registration. Stage B and Stage C did not run, and nothing was spent on
Modal. R2 stays off in production. The +0.032 / +0.040 surfaced-recall gain
(`r2e2e_findings.md`) is real but costs about 95 s to 7.5 min of CPU per deep
search on this hardware, before the first result is shown, and it blocks both
vCPUs for other requests. AD-44.

Paths not taken, each a new item if wanted:
- Rerank only the tail after the head is shown (keeps latency off the first
  results; the measured configuration changes, so it needs its own run).
- A GPU or a large-CPU machine for deep searches only.
- A smaller cross-encoder distilled from R2, which needs training and a fresh recall check.
