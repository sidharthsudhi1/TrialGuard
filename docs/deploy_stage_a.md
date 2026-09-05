# Deploying Stage A (FastAPI + Next.js)

Gradio on HF Spaces remains the **$0 SIGIR** demo (`app.py` + `docs/deploy.md`).
Stage A is the production-corpus web app: structured JSON so quotes can be
highlighted inside `eligibility_raw`.

## API (Fly.io)

Needs ≥2 GB RAM for MedCPT. Dockerfile: `Dockerfile.api`, config: `fly.toml`.

```bash
fly apps create trialguard-api   # once
fly secrets set \
  DATABASE_URL=... \
  DEEPINFRA_API_KEY=... \
  LLM_PROVIDER=deepinfra \
  DAILY_USD_CAP=2.00 \
  TG_PROMPT_VERSION=v4 \
  API_CORS_ORIGIN=https://YOUR_VERCEL_APP.vercel.app
fly deploy
```

Local:

```bash
pip install -e ".[web]"
export DATABASE_URL=... DEEPINFRA_API_KEY=... API_CORS_ORIGIN=http://localhost:3000
python -m trialguard.api
# → http://localhost:8000/api/health
```

## Frontend (Vercel)

```bash
cd web
cp .env.example .env.local
# NEXT_PUBLIC_API_URL=https://trialguard-api.fly.dev
npm install && npm run dev
```

Deploy the `web/` directory as a Vercel project; set `NEXT_PUBLIC_API_URL` to the
Fly URL. CORS on the API must match the Vercel origin exactly (no `*`).

## Synthetic-only posture

Every free-text note is checked with `detect_injection` server-side (reject, do
not sanitize). The UI and `/api/health` state that only synthetic notes are
accepted. Free-text assessments skip analyst cache writes; presets may write.

The synthetic-only rule is **procedural, not enforced**: nothing in the code can
tell a synthetic note from a real one. Served requests are traced to Langfuse
with full prompts, so submitted text leaves this infrastructure — `/api/search`
under session `request_id`, `/api/assess` under the `job_id`, both tagged
`served`. Say this plainly rather than implying a guarantee the code does not
provide.

## Scheduled corpus refresh

`python -m trialguard.scripts.refresh` reconciles `ctgov_live` against
ClinicalTrials.gov: expired trials are deleted, new ones embedded and inserted,
and records whose `lastUpdatePostDate` moved are re-embedded because their
eligibility text may have moved with it. An already-current corpus writes nothing
and makes zero embedding calls, so running it more often than necessary costs a
CT.gov crawl and nothing else.

It runs as a **Fly scheduled machine in the same app**, not as a GitHub Actions
job. Three reasons: the image already has MedCPT baked in, so embedding needs no
new build; `DATABASE_URL` is an app secret the machine inherits rather than a
credential copied into a second system; and it runs in `syd` beside Neon, where
the corpus SELECT and the upserts are local rather than crossing a region.

```bash
# The image the API is currently running.
fly image show --app trialguard-api

fly machine run <that-image-ref> \
  --app trialguard-api \
  --region syd \
  --schedule daily \
  --vm-memory 4096 --vm-cpu-kind performance --vm-cpus 2 \
  --restart no \
  python -m trialguard.scripts.refresh
```

`--vm-memory 4096` because MedCPT loads to embed new and revised trials; a
refresh that finds nothing to embed never loads it, but the machine has to be
sized for the run that does. `--restart no` so a failed crawl waits for the next
schedule instead of retrying into the CT.gov rate limit.

The crawl is paced at `ctgov_request_delay` (1.5 s) over `ctgov_page_size` (100),
so ~26k trials is ~260 requests and roughly six to seven minutes before any
embedding. **Measure the first run rather than trusting that estimate** — it is
arithmetic, not an observation.

**Verify after the first `fly deploy` that the scheduled machine still exists**
(`fly machines list`). Deploys reconcile the app's machines, and whether a
scheduled machine survives one depends on Fly behaviour this repo has not
measured. If it is removed, re-create it with the command above as a deploy step,
or move it to its own app with `DATABASE_URL` set separately.

### Seeing the result

`/api/health` reports `corpus_refresh` — the counts from the last run and the
timestamp it finished — so a refresh that silently stopped running is visible
rather than inferred. Per-trial `last_updated` is served on `/api/search` and
`/api/trials/{id}` and rendered in the UI, with records CT.gov has not touched in
over a year marked visibly stale.

## Cost bounds

Per-request trial cap (`API_MAX_ASSESS_TRIALS`, default 5), per-IP rate limits on
`/api/search` and `/api/assess`, and the existing global `DAILY_USD_CAP` ledger
(surfaced at `/api/budget` and as HTTP 402 when exhausted).

## TREC v4 caveat (unchanged)

TREC 2021 prompt-v4: exclusion unsupported-rate 31.2% vs inclusion 9.2%; retry
not significant (p=0.2514). See
[`data/reports/phase9v4_agent_trec_2021.json`](../data/reports/phase9v4_agent_trec_2021.json).
Serving wraps `retrieve()` / `assess()` unchanged — do not treat UI issues as
agent regressions.
