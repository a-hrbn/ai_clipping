#!/usr/bin/env bash
# Generate Jynxzi clip profile from @Iboclipss YouTube Shorts.
#
# Fetches the top clips from the last 90 days, calls an LLM to extract clip
# archetypes + selection rules, and writes:
#   backend/ml/artifacts/jynxzi_profile.json
#
# Usage:
#   ./scripts/generate_jynxzi_profile.sh                      # generate profile
#   ./scripts/generate_jynxzi_profile.sh --dry-run            # preview JSON, no write
#   ./scripts/generate_jynxzi_profile.sh --top-n 80           # limit to 80 clips
#   ./scripts/generate_jynxzi_profile.sh --window-days 60
#   ./scripts/generate_jynxzi_profile.sh --provider anthropic
#   ./scripts/generate_jynxzi_profile.sh --provider gemini --model gemini-2.5-pro
#   ./scripts/generate_jynxzi_profile.sh --help               # show all options
set -euo pipefail

CONTAINER="ai-clipping-backend"
WORKDIR="/app/backend"
OUT_PATH="$WORKDIR/ml/artifacts/jynxzi_profile.json"
START_EPOCH=$SECONDS

log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }
die() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; exit 1; }

# resolve provider/model from args (fall back to script defaults)
_PROVIDER="anthropic"; _MODEL="claude-opus-4-8"
_ARGS=("$@")
for (( i=0; i<${#_ARGS[@]}; i++ )); do
  case "${_ARGS[$i]}" in
    --provider) _PROVIDER="${_ARGS[$((i+1))]:-$_PROVIDER}" ;;
    --model)    _MODEL="${_ARGS[$((i+1))]:-$_MODEL}" ;;
  esac
done

log "=== generate_jynxzi_profile ==="
log "Container : $CONTAINER"
log "Output    : $OUT_PATH"
log "Provider  : $_PROVIDER"
log "Model     : $_MODEL"
log "Args      : ${*:-(defaults)}"
log ""

# --- container health check --------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
  die "docker not found in PATH."
fi

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  die "Container '$CONTAINER' does not exist. Start it first: docker compose up -d"
fi

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q "^true$"; then
  die "Container '$CONTAINER' is not running. Start it first: docker compose up -d"
fi

log "Container '$CONTAINER' is running — starting profile generation…"
log ""

# --- run ---------------------------------------------------------------------

docker exec -w "$WORKDIR" "$CONTAINER" python ml/generate_jynxzi_profile.py "$@"
EXIT_CODE=$?

# --- summary -----------------------------------------------------------------

log ""
ELAPSED=$(( SECONDS - START_EPOCH ))
if [[ $EXIT_CODE -eq 0 ]]; then
  log "Done in ${ELAPSED}s."
  # Only show output path hint when not a dry-run
  if [[ ! " $* " =~ " --dry-run " ]]; then
    log "Profile written → $OUT_PATH"
    log "Reload the workbench to pick up the new profile."
  fi
else
  log "Failed (exit $EXIT_CODE) after ${ELAPSED}s."
fi

exit $EXIT_CODE
