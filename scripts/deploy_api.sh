#!/usr/bin/env bash
#
# Deploy the Stage A API and re-create the scheduled corpus-refresh machine.
#
# The refresh machine is destroyed and re-created on every deploy rather than
# left alone. Whether `fly deploy` preserves a scheduled machine, and whether it
# updates that machine's image if it does, is Fly behaviour this repo has not
# measured -- and the failure mode of guessing wrong is silent in both
# directions. A destroyed machine stops refreshing the corpus; a preserved one
# that keeps its old image runs last release's refresh code forever, so a fix to
# scripts/refresh.py never actually ships. Recreating collapses both cases into
# one known state at the cost of two API calls.
#
#   scripts/deploy_api.sh              deploy, then re-create the refresh machine
#   scripts/deploy_api.sh --skip-deploy   only re-create the refresh machine
#
set -euo pipefail

APP="${FLY_APP:-trialguard-api}"
REGION="${FLY_REGION:-syd}"
# Its own process group so `fly deploy` does not apply the http_service config to
# a machine that runs a batch script and exits. Without this it lands in `app`
# and is health-checked as a web server.
GROUP="refresh"
SCHEDULE="${REFRESH_SCHEDULE:-daily}"

for cmd in fly jq curl; do
  command -v "$cmd" >/dev/null || { echo "error: $cmd not found on PATH" >&2; exit 1; }
done

# The SLO probe below needs this repo's dependencies, and bare `python` is not a
# thing on a current macOS. Prefer the project venv, then python3.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
  else
    PYTHON="$(command -v python3 || true)"
  fi
fi
[[ -n "$PYTHON" ]] || { echo "error: no python found; set PYTHON=/path/to/python" >&2; exit 1; }

if [[ "${1:-}" != "--skip-deploy" ]]; then
  echo "==> Deploying $APP"
  fly deploy --app "$APP"
fi

echo "==> Resolving the image just deployed"
# Read it off the app machine's own config, not `fly image show`. That command
# returns an array of {Registry, Repository, Tag, Digest} with no ref field to
# index, and no process group on the rows -- so once this script has also created
# a refresh machine it cannot answer "which image is the web app running". The
# machine config carries the fully qualified ref and is filterable by group.
IMAGE="$(fly machines list --app "$APP" --json | jq -r '
  [.[] | select(.config.metadata.fly_process_group == "app") | .config.image]
  | unique
  | if length == 1 then .[0] else "" end')"
if [[ -z "$IMAGE" || "$IMAGE" == "null" ]]; then
  echo "error: could not resolve a single image for the 'app' process group." >&2
  echo "       Is $APP deployed, and do its machines agree on an image?" >&2
  fly machines list --app "$APP" --json | jq -r '.[] |
    "  \(.id) group=\(.config.metadata.fly_process_group // "-") image=\(.config.image)"' >&2
  exit 1
fi
echo "    $IMAGE"

echo "==> Removing any existing refresh machine"
# Tolerate none existing: this script is also the way to create the first one.
existing="$(fly machines list --app "$APP" --json \
  | jq -r --arg g "$GROUP" '.[] | select(.config.metadata.fly_process_group == $g) | .id')"
for id in $existing; do
  echo "    destroying $id"
  fly machine destroy --force --app "$APP" "$id"
done

echo "==> Creating the scheduled refresh machine ($SCHEDULE)"
# The `--` before the command is load-bearing: -m is flyctl's short form of
# --metadata, so without it `python -m trialguard.scripts.refresh` has its -m
# parsed as a flag and the run fails with "invalid key/value pairs specified for
# flag metadata", pointing at the wrong argument entirely.
# --restart no: a failed CT.gov crawl waits for the next schedule instead of
# retrying straight back into the rate limit that failed it.
# 4 GB because MedCPT loads to embed new and revised trials. A refresh that finds
# nothing to embed never loads it, but the machine is sized for the run that does.
fly machine run "$IMAGE" \
  --app "$APP" \
  --region "$REGION" \
  --schedule "$SCHEDULE" \
  --restart no \
  --vm-memory 4096 \
  --vm-cpu-kind performance \
  --vm-cpus 2 \
  --metadata "fly_process_group=$GROUP" \
  -- python -m trialguard.scripts.refresh

echo "==> Result"
fly machines list --app "$APP" --json | jq -r --arg g "$GROUP" '
  .[] | select(.config.metadata.fly_process_group == $g)
  | "  id=\(.id)  schedule=\(.config.schedule // "NONE")  image=\(.config.image)"'

# fly.toml keeps min_machines_running = 0 and suspends idle machines, so a deploy
# leaves the app stopped and the probe's own first request is what resumes it.
# Measured on this app: that path answers 502, then takes 34.9 s for the call the
# probe records as "warm", and the gate fails on a boot instead of a regression.
# The SLO file used to assert the machine would already have answered /api/health
# by this point; it does not. Establishing that precondition is the gate's job,
# not the probe's -- the probe should keep reporting whatever it actually sees.
echo "==> Waiting for $APP to answer /api/health"
BASE="${TG_API_BASE_URL:-https://$APP.fly.dev}"
code=""
for attempt in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$BASE/api/health" || true)"
  [[ "$code" == "200" ]] && { echo "    healthy after $attempt attempt(s)"; break; }
  sleep 5
done
if [[ "$code" != "200" ]]; then
  echo "error: $APP never answered /api/health (last status: ${code:-none})." >&2
  echo "       The release is live; check: fly logs --app $APP" >&2
  exit 1
fi

# Health answering 200 does not mean the search path is warm: the dense index
# loads lazily and its first queries are still descending. Measured across
# resumes, server-side total_ms goes 20576 -> 1023.7 -> 542.8 -> 570.1, and this
# gate failed at 1512.4 ms against a 1500 ms bound on what was only the second
# call. Two samples do not reach steady state, so the gate takes a throwaway pass
# first and measures on the one after it. Loosening the bound instead would have
# tuned away a real signal to hide a warm-up artifact.
echo "==> Warming the search path"
"$PYTHON" -m trialguard.eval.served_probe --base-url "$BASE" >/dev/null 2>&1 || true

# WS-6a. CI has no deployed environment, so latency cannot be gated there; this
# is the post-deploy smoke check that can. Tighter bounds than the nightly
# alerting run, because this fires immediately after a known change against a
# machine Fly has already health-checked -- it is a regression gate, not a drift
# detector.
echo "==> Post-deploy SLO check"
if ! "$PYTHON" -m trialguard.eval.served_probe \
      --base-url "${TG_API_BASE_URL:-https://$APP.fly.dev}" \
      --thresholds data/reports/served_slo.json; then
  echo
  echo "SLO check FAILED against the deployment just shipped." >&2
  echo "The release is live -- this does not roll it back. Investigate or run:" >&2
  echo "  fly releases --app $APP    # then: fly deploy --image <previous>" >&2
  exit 1
fi

cat <<'EOF'

The refresh writes its counts and finish time to cache_entries under
("corpus", "last_refresh"), surfaced at /api/health as corpus_refresh. If that
field stops advancing, the schedule is not running -- check it there rather than
assuming this script's success means the machine fired.
EOF
