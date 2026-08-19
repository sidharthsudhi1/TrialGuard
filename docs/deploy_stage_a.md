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
