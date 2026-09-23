# E1 runbook — scale ceiling and corpus generalisation

Two experiments, run on a throwaway EC2 box. Neither touches the Mac, and E1a
never touches production Neon.

- **E1a** — where does the in-process dense matrix stop being the right answer?
  Synthetic scale-out from real MedCPT vectors, measuring exact-matrix vs ivfflat
  vs HNSW on recall, latency and resident memory. **~$3-6, one session.**
- **E1c** — does the system hold outside oncology, at full ClinicalTrials.gov
  scale? Real ingest of ~500k trials. **Storage tier decision required before
  starting; see §4.**

Run E1a first. It produces the crossover curve, and the curve decides whether E1c
is worth its storage bill.

---

## 0 — Why the corpus is synthesised rather than random

Uniform random 768-dim vectors have no cluster structure. That is the
pathological worst case for any ANN index: recall measured against it is far
below what real MedCPT embeddings get, so the latency curve would transfer to
production and the recall curve would not. Recall is the axis the decision turns
on, so the corpus has to keep the clumpiness of the real thing.

`bench_ann_scale.py` grows the real 26,148-vector TREC 2021 MedCPT index by
resampling with replacement and perturbing on the unit sphere (`--sigma`, default
0.05). Neighbour density is preserved, so the top-k problem stays as hard as the
real one. Queries are drawn from the unperturbed base, so they carry the real
query distribution.

This is a proxy and should be reported as one. It answers *where does the
architecture break*, not *what is production recall at 5M*. Only E1c answers the
second.

---

## 1 — Box

| Sizes to run | Instance | RAM | Why |
|---|---|---|---|
| ≤ 1M | `c6i.2xlarge` | 16 GB | 1M vectors = 3.07 GB resident, fits with headroom |
| ≤ 5M | `r6i.4xlarge` | 128 GB | 5M = 15.4 GB resident; 16 GB would skip the matrix arm |

Non-burstable on purpose — a `t3` throttles mid-benchmark and silently corrupts
the latency numbers. Your existing `trialguard-trec.pem` key pair works; the EC2
recipe in memory applies (no IAM instance profile — `PassRole` is denied on this
account, so `scp` in and out rather than pulling from S3).

Storage: 100 GB gp3. The 5M memmap alone is 15.4 GB and the Postgres table is
larger; the harness deletes each corpus before building the next.

```bash
# from the Mac, once the instance is up
scp -i trialguard-trec.pem \
  data/indexes/trec_2021_medcpt_excl_embeddings.npy \
  data/indexes/trec_2021_medcpt_excl_ids.json \
  scripts/bench_ann_scale.py \
  ec2-user@$HOST:~/
```

80 MB total. Nothing else from the repo is needed — the harness imports only
numpy and psycopg2.

---

## 2 — Postgres with pgvector, on the box

Local Docker, so the benchmark cannot reach Neon even by misconfiguration.

```bash
sudo yum install -y docker && sudo service docker start
sudo docker run -d --name pgbench \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  --shm-size=1g \
  pgvector/pgvector:pg16

python3 -m venv .venv && . .venv/bin/activate
pip install numpy psycopg2-binary
```

`--shm-size=1g` matters: HNSW builds at 1M+ rows fail on Docker's 64 MB default
shared memory with a misleading out-of-memory error.

Tune before loading, or the index builds dominate the runtime:

```sql
ALTER SYSTEM SET maintenance_work_mem = '4GB';
ALTER SYSTEM SET max_parallel_maintenance_workers = 4;
SELECT pg_reload_conf();
```

---

## 3 — Run E1a

```bash
mkdir -p data/indexes data/reports
mv trec_2021_medcpt_excl_*.npy trec_2021_medcpt_excl_*.json data/indexes/ 2>/dev/null

python scripts/bench_ann_scale.py \
  --dsn postgresql://postgres:postgres@localhost:5432/postgres \
  --sizes 100000,500000,1000000 \
  --k 100 --n-queries 200 \
  --ram-budget-gb 12 \
  --out data/reports/e1a_ann_scale.json
```

Run under `nohup` or `tmux` — the 1M HNSW build alone runs tens of minutes, and
an SSH drop kills the job otherwise.

Each size prints its block as it completes, so partial results survive a crash.
Expect roughly 4-8 h for three sizes; add the 5M row only on the `r6i` box.

Pull the result back and tear down:

```bash
scp -i trialguard-trec.pem ec2-user@$HOST:~/data/reports/e1a_ann_scale.json data/reports/
aws ec2 terminate-instances --instance-ids $ID
```

**Terminate explicitly.** A forgotten `r6i.4xlarge` is ~$25/day and is the only
way this experiment becomes expensive.

### What the output answers

For each corpus size, one row per arm:

- `recall_vs_exact` — ivfflat and HNSW against the exact top-100. The matrix arm
  is 1.0 by construction.
- `p50_ms` / `p95_ms` — query latency.
- `resident_gb` for the matrix arm; `heap_gb` and `index_gb` for the Postgres
  arms. Heap is measured once after load and each index is measured on its own,
  so ivfflat is never charged for HNSW's footprint.
- A `skipped` field on the matrix arm once the corpus exceeds `--ram-budget-gb`.
  That skip is the headline result, not an error.

The crossover is the size where an index first matches the matrix on recall while
beating it on latency-per-GB-resident. Quote it with the sigma, because it is a
proxy.

---

## 4 — E1c, and the decision it needs first

E1c loads the real corpus across all conditions. Two blockers to settle before
starting.

**Code change.** `ingestion/ctgov.py:78` hardcodes the scope:

```python
"query.cond": "cancer OR oncology OR tumor OR neoplasm",
```

`fetch_oncology_trials` needs a condition parameter before it can fetch anything
else. CLAUDE.md records oncology as locked scope, so this is a deliberate
reversal and belongs in the AD log, not a quiet edit.

**Storage.** Measured from your own deployment: 531 MB / 26,037 trials =
**20.4 KB per trial**, of which the embedding is only 3 KB — the rest is
`eligibility_raw`, the criteria arrays, `doc_tsv` and the GIN index.

| Scope | Trials | Neon storage | Matrix RAM | Embed time |
|---|---|---|---|---|
| Oncology, recruiting (today) | 26,037 | 531 MB | 80 MB | — |
| All conditions, recruiting | ~120,000 | ~2.4 GB | 368 MB | ~3.1 h |
| All conditions, all statuses | ~500,000 | **~10 GB** | 1.54 GB | ~12.9 h |

Embed time is measured, not estimated: MedCPT on Apple MPS ran **10.8 docs/s**
steady state at `batch_size=32` (batch 64 was slower, 9.6/s). On the benchmark
box embedding is CPU-only and will differ — measure it there rather than
assuming, or attach a GPU instance for the real run.

Fetch time is bounded by politeness, not bandwidth: `ctgov_page_size=100` and
`ctgov_request_delay=1.5` means 500k trials is 5,000 requests ≈ **2.1 h**.

Neon storage is the one recurring cost, and it is the constraint that produced
AD-6 in the first place. **Confirm current per-GB-month pricing against your Neon
dashboard before committing** — going from 531 MB to ~10 GB is a plan tier, not a
rounding error.

**Do E1c on a separate Neon branch or database, never on `ctgov_live`.** The live
corpus serves the deployed demo.

---

## 5 — What lands in the repo

- `data/reports/e1a_ann_scale.json` — raw arms, all sizes.
- `data/reports/e1a_findings.md` — the crossover, stated with the sigma caveat
  and with what it does not claim.
- An AD entry for the outcome. If the matrix holds to 1M, that is a result worth
  recording as much as if it breaks at 300k — the repo's convention is that
  negatives are logged, not deleted.
