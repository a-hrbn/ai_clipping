#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
source "$REPO_ROOT/scripts/lib_backend_container.sh"

PYTHON_BIN="${PYTHON_BIN:-$BACKEND_DIR/.venv/bin/python}"
BACKEND_CONTAINER="${AI_CLIPPING_BACKEND_CONTAINER:-ai-clipping-backend}"
DEV_MODE="$(normalize_boolish "${AI_CLIPPING_DEV:-1}")"
BACKEND_RUNTIME="$(normalize_boolish "${AI_CLIPPING_BACKEND_RUNTIME:-auto}")"
REBUILD_MODE="$(normalize_boolish "${AI_CLIPPING_REBUILD:-auto}")"
COMPOSE_FILES=(-f docker-compose.yml)
RUN_LOCAL=0
PHASE_A_ENV_ARGS=(
  -e ENABLE_DENSE_CANDIDATE_INVENTORY=true
  -e DENSE_CANDIDATE_INVENTORY_SOURCE_CAP=700
)

case "$DEV_MODE" in
  1|true|yes|on)
    COMPOSE_FILES+=(-f docker-compose.dev.yml)
    ;;
  0|false|no|off|"")
    ;;
  *)
    echo "Invalid AI_CLIPPING_DEV=$DEV_MODE. Use true/false." >&2
    exit 1
    ;;
esac

case "$BACKEND_RUNTIME" in
  auto|"")
    if docker_gpu_available; then
      BACKEND_RUNTIME="cuda"
      COMPOSE_FILES+=(-f docker-compose.cuda.yml)
    else
      BACKEND_RUNTIME="cpu"
    fi
    ;;
  cuda|gpu)
    BACKEND_RUNTIME="cuda"
    if ! docker_gpu_available; then
      echo "CUDA backend requested, but Docker GPU access is not available." >&2
      echo "Use AI_CLIPPING_BACKEND_RUNTIME=cpu to force CPU." >&2
      exit 1
    fi
    COMPOSE_FILES+=(-f docker-compose.cuda.yml)
    ;;
  cpu)
    ;;
  *)
    echo "Invalid AI_CLIPPING_BACKEND_RUNTIME=$BACKEND_RUNTIME. Use auto, cuda, gpu, or cpu." >&2
    exit 1
    ;;
esac

print_wrapper_usage() {
  cat <<'EOF'
Usage: scripts/run_serving_recall_holdout.sh [wrapper options] [runner options]

Wrapper options:
  --local                 Run with backend/.venv on the host instead of Docker.
  --container NAME        Backend container name. Default: ai-clipping-backend.
  --wrapper-help          Show this wrapper help.

Runner options are forwarded to ml.run_serving_recall_holdout, for example:
  --limit 2

Runner defaults: --fail-fast, --progress-interval-sec 3, --workers 2, judge-off.
Phase-A defaults are forced for this holdout: dense inventory on, source cap 700.
Use --enable-judge for a Judge-on reference run.
Use --enable-llm-moment-finder for the Phase-2 LLM Moment-Finder recall-floor run.
Use --no-fail-fast, --progress-interval-sec N, or --workers 1 to override.

Set AI_CLIPPING_BACKEND_CONTAINER to override the default container name.
Set AI_CLIPPING_BACKEND_RUNTIME=auto|cuda|cpu. Default: auto.
Set AI_CLIPPING_REBUILD=auto|1|0 to control backend image rebuild checks.
EOF
}

RUNNER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      RUN_LOCAL=1
      shift
      ;;
    --container)
      BACKEND_CONTAINER="${2:?Missing value for --container}"
      shift 2
      ;;
    --container=*)
      BACKEND_CONTAINER="${1#--container=}"
      shift
      ;;
    --wrapper-help)
      print_wrapper_usage
      exit 0
      ;;
    --)
      shift
      RUNNER_ARGS+=("$@")
      break
      ;;
    *)
      RUNNER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$RUN_LOCAL" == "1" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing backend interpreter: $PYTHON_BIN" >&2
    exit 1
  fi
  cd "$REPO_ROOT"
  export ENABLE_DENSE_CANDIDATE_INVENTORY=true
  export DENSE_CANDIDATE_INVENTORY_SOURCE_CAP=700
  PYTHONPATH="$BACKEND_DIR${PYTHONPATH:+:$PYTHONPATH}" exec "$PYTHON_BIN" -m ml.run_serving_recall_holdout "${RUNNER_ARGS[@]}"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for the default containerized holdout run. Use --local to run on the host." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is required for the default containerized holdout run. Use --local to run on the host." >&2
  exit 1
fi

if [[ "$BACKEND_CONTAINER" == "ai-clipping-backend" ]]; then
  ensure_backend_image_current "$REPO_ROOT" "$BACKEND_CONTAINER" "$DEV_MODE" "$REBUILD_MODE" "${COMPOSE_FILES[@]}"
  ensure_backend_container_running "$BACKEND_CONTAINER" "${COMPOSE_FILES[@]}"
  wait_for_backend_container "$BACKEND_CONTAINER" 180
else
  if ! docker inspect "$BACKEND_CONTAINER" >/dev/null 2>&1; then
    echo "Backend container not found: $BACKEND_CONTAINER" >&2
    echo "Start it with scripts/start_containerized.sh or pass --container NAME." >&2
    exit 1
  fi

  if [[ "$(docker inspect -f '{{.State.Running}}' "$BACKEND_CONTAINER")" != "true" ]]; then
    echo "Backend container is not running: $BACKEND_CONTAINER" >&2
    echo "Start it with scripts/start_containerized.sh or pass --container NAME." >&2
    exit 1
  fi
fi

exec docker exec -i \
  -w /app \
  -e PYTHONPATH=/app/backend \
  "${PHASE_A_ENV_ARGS[@]}" \
  "$BACKEND_CONTAINER" \
  python -m ml.run_serving_recall_holdout "${RUNNER_ARGS[@]}"
