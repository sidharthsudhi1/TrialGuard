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

for cmd in fly jq; do
  command -v "$cmd" >/dev/null || { echo "error: $cmd not found on PATH" >&2; exit 1; }
done

if [[ "${1:-}" != "--skip-deploy" ]]; then
  echo "==> Deploying $APP"
  fly deploy --app "$APP"
fi

echo "==> Resolving the image just deployed"
IMAGE="$(fly image show --app "$APP" --json | jq -r '.Ref')"
[[ -n "$IMAGE" && "$IMAGE" != "null" ]] || { echo "error: could not resolve image ref" >&2; exit 1; }
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
  python -m trialguard.scripts.refresh

echo "==> Result"
fly machines list --app "$APP" --json | jq -r --arg g "$GROUP" '
  .[] | select(.config.metadata.fly_process_group == $g)
  | "  id=\(.id)  schedule=\(.config.schedule // "NONE")  image=\(.config.image)"'

# WS-6a. CI has no deployed environment, so latency cannot be gated there; this
# is the post-deploy smoke check that can. Tighter bounds than the nightly
# alerting run, because this fires immediately after a known change against a
# machine Fly has already health-checked -- it is a regression gate, not a drift
# detector.
echo "==> Post-deploy SLO check"
if ! python -m trialguard.eval.served_probe \
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
