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

## Latency SLO

Enforced by `scripts/deploy_api.sh` after every deploy, against
`data/reports/served_slo.json`. Not a CI gate: CI has no deployed environment,
which is why E3 deferred latency in the first place and why the deferral then sat
unbuilt.

| | bound | measured 2026-09-05/07 |
|---|---|---|
| `/api/health`, warm | 500 ms | 28.7–45.3 ms |
| `/api/search`, server-side `total_ms`, warm | 1500 ms | 506–646 ms |
| `/api/search` wall time | 30 s | 683 ms warm |
| search returns trials | ≥ 1 | 10 |

**Cold start is bounded by Fly, not here.** `fly.toml` sets `grace_period = "5m"`
on the health check, so a boot slower than that flaps the machine into a restart
and the deploy fails on its own. By the time the smoke check runs, the machine
has already answered `/api/health`, so any number it measures is warm — gating it
would assert something it did not observe. The probe records the resume path
instead.

**What "780 ms warm" does not describe.** The first search after a resume from
suspend was measured at **20,576 ms server-side** with the keyword cache already
warm (`keyword_ms` 24.2), settling to 542–570 ms by the third call, with
`dense_ms` falling 3,277 → 1,514 across them. `fly.toml` suspends idle machines,
so this is what a user meets opening the demo after a quiet period — not the
780 ms the reports quote, which is the steady state. The probe measures both and
gates only the warm one; the first-call number is reported so it stops being
invisible.

Two other timings worth having beside these: resuming a suspended machine costs
12.2–16.3 s to first `/api/health` response, and the in-process vector cache
reports `load_seconds: 32.9` for its 25,965 rows.

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

**Deploy with `scripts/deploy_api.sh`, not bare `fly deploy`.** The script
deploys, then destroys and re-creates the refresh machine on the image it just
shipped.

```bash
scripts/deploy_api.sh                 # deploy + re-create the refresh machine
scripts/deploy_api.sh --skip-deploy   # re-create it only
```

Re-creating rather than checking is deliberate. Whether `fly deploy` preserves a
scheduled machine — and whether it updates that machine's *image* if it does — is
Fly behaviour this repo has not measured, and guessing wrong fails silently in
both directions:

| | outcome | consequence |
|---|---|---|
| a | machine destroyed by the deploy | the corpus stops being refreshed |
| b | preserved, image unchanged | worse, because it looks fine: the refresh runs last release's code forever, so a fix to `refresh.py` never ships |
| c | preserved and updated | the intended state |

(b) is indistinguishable from (c) in `fly machines list` unless you compare image
refs. Recreating collapses all three into one known state for the cost of two API
calls, and does not depend on a platform behaviour that can change in a Fly
release without anyone noticing.

The machine runs in its own `fly_process_group=refresh`, so a deploy does not
apply the `http_service` config — and its health checks — to a machine that runs
a batch script and exits.

`--vm-memory 4096` because MedCPT loads to embed new and revised trials; a
refresh that finds nothing to embed never loads it, but the machine has to be
sized for the run that does. `--restart no` so a failed crawl waits for the next
schedule instead of retrying straight back into the rate limit that failed it.

The crawl is paced at `ctgov_request_delay` (1.5 s) over `ctgov_page_size` (100),
so ~26k trials is ~260 requests and roughly six to seven minutes before any
embedding. **Measure the first run rather than trusting that estimate** — it is
arithmetic, not an observation.

### Seeing the result

`/api/health` reports `corpus_refresh` — the counts from the last run and the
timestamp it finished — so a refresh that silently stopped running is visible
rather than inferred. That field, not the deploy script exiting 0, is the
evidence the schedule actually fires: the script proves the machine was created,
not that Fly ran it. Per-trial `last_updated` is served on `/api/search` and
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
