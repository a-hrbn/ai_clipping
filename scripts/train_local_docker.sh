#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib_backend_container.sh"

IMAGE_MODE="$(normalize_boolish "${AI_CLIPPING_LOCAL_TRAIN_IMAGE_MODE:-legacy}")"
LEGACY_IMAGE_TAG="${AI_CLIPPING_LOCAL_TRAIN_IMAGE:-ai-clipping-train-local:latest}"
DOCKERFILE_PATH="${AI_CLIPPING_LOCAL_TRAIN_DOCKERFILE:-$REPO_ROOT/backend/Dockerfile.local-train}"
ENV_FILE="${AI_CLIPPING_LOCAL_TRAIN_ENV_FILE:-$REPO_ROOT/backend/.env}"
ARTIFACT_DIR="$REPO_ROOT/backend/ml/artifacts"
TMP_DIR="$REPO_ROOT/backend/tmp"
PREWARMED_MODEL_DIR="${AI_CLIPPING_LOCAL_PREWARMED_MODEL_DIR:-$REPO_ROOT/tmp/prewarmed_sentence_transformers_local}"
NVIDIA_DRIVER_CAPABILITIES_VALUE="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,video}"
ENTRYPOINT_OVERRIDE="${AI_CLIPPING_LOCAL_TRAIN_ENTRYPOINT:-}"
CONTAINER_USER="${AI_CLIPPING_LOCAL_TRAIN_USER:-$(id -u):$(id -g)}"
TRAIN_WHISPER_BACKEND="${AI_CLIPPING_LOCAL_TRAIN_WHISPER_BACKEND:-${WHISPER_BACKEND:-faster-whisper}}"
TRAIN_WHISPER_DEVICE="${AI_CLIPPING_LOCAL_TRAIN_WHISPER_DEVICE:-${WHISPER_DEVICE:-}}"
TRAIN_WHISPER_COMPUTE_TYPE="${AI_CLIPPING_LOCAL_TRAIN_WHISPER_COMPUTE_TYPE:-${WHISPER_COMPUTE_TYPE:-}}"
TRAIN_FULL_VOD_AUDIO_WORKERS="${AI_CLIPPING_LOCAL_TRAIN_FULL_VOD_AUDIO_WORKERS:-${CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS:-3}}"
BACKEND_RUNTIME="$(normalize_boolish "${AI_CLIPPING_LOCAL_TRAIN_BACKEND_RUNTIME:-${AI_CLIPPING_BACKEND_RUNTIME:-auto}}")"
REBUILD_MODE="$(normalize_boolish "${AI_CLIPPING_REBUILD:-auto}")"
COMPOSE_FILES=(-f docker-compose.yml)

show_help() {
  cat <<'EOF'
Usage:
  scripts/train_local_docker.sh [local_train.py args]

Environment:
  AI_CLIPPING_LOCAL_TRAIN_IMAGE_MODE=backend|legacy
      legacy (default): build and use backend/Dockerfile.local-train.
      backend: use the current backend image, rebuilding it when needed.
  AI_CLIPPING_LOCAL_TRAIN_BACKEND_RUNTIME=auto|cuda|gpu|cpu
      Backend image runtime for backend mode. Default: AI_CLIPPING_BACKEND_RUNTIME, then auto.
  AI_CLIPPING_REBUILD=auto|1|0
      Auto-check backend image freshness, force rebuild, or disable rebuild checks.
  AI_CLIPPING_LOCAL_TRAIN_ENTRYPOINT=python
      Override entrypoint for custom training commands.
  AI_CLIPPING_LOCAL_TRAIN_WHISPER_BACKEND=faster-whisper
      Training-only Whisper backend override applied after backend/.env.
  AI_CLIPPING_LOCAL_TRAIN_WHISPER_DEVICE=cuda
      Training-only Whisper device override applied after backend/.env.
  AI_CLIPPING_LOCAL_TRAIN_WHISPER_COMPUTE_TYPE=float16
      Training-only compute type override applied after backend/.env.
  AI_CLIPPING_LOCAL_TRAIN_FULL_VOD_AUDIO_WORKERS=3
      Training-only full-VOD transcription/fingerprint worker default.

Any arguments after the wrapper options are passed to the training command.
EOF
}

if [[ "${1:-}" == "--wrapper-help" ]]; then
  show_help
  exit 0
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR" "$TMP_DIR"

docker_gpu_runtime_available() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'
}

case "$IMAGE_MODE" in
  backend)
    case "$BACKEND_RUNTIME" in
      auto|"")
        if docker_gpu_runtime_available; then
          BACKEND_RUNTIME="cuda"
        else
          BACKEND_RUNTIME="cpu"
        fi
        ;;
      cuda|gpu)
        BACKEND_RUNTIME="cuda"
        ;;
      cpu)
        ;;
      *)
        echo "Invalid AI_CLIPPING_LOCAL_TRAIN_BACKEND_RUNTIME=$BACKEND_RUNTIME. Use auto, cuda, gpu, or cpu." >&2
        exit 1
        ;;
    esac
    if [[ "$BACKEND_RUNTIME" == "cuda" ]]; then
      COMPOSE_FILES+=(-f docker-compose.cuda.yml)
    fi
    ensure_backend_image_current "$REPO_ROOT" "__ai_clipping_training_image_check__" 0 "$REBUILD_MODE" "${COMPOSE_FILES[@]}"
    IMAGE_TAG="$(backend_image_from_compose "${COMPOSE_FILES[@]}")"
    if [[ -z "$IMAGE_TAG" ]]; then
      echo "Could not determine backend image for local training." >&2
      exit 1
    fi
    echo "Using backend image for local training: $IMAGE_TAG runtime=$BACKEND_RUNTIME"
    ;;
  legacy)
    IMAGE_TAG="$LEGACY_IMAGE_TAG"
    echo "Using legacy local training image: $IMAGE_TAG"
    docker build \
      -f "$DOCKERFILE_PATH" \
      -t "$IMAGE_TAG" \
      "$REPO_ROOT"
    ;;
  *)
    echo "Invalid AI_CLIPPING_LOCAL_TRAIN_IMAGE_MODE=$IMAGE_MODE. Use backend or legacy." >&2
    exit 1
    ;;
esac

if [[ -z "$TRAIN_WHISPER_DEVICE" ]]; then
  if [[ "$IMAGE_MODE" == "backend" && "$BACKEND_RUNTIME" == "cpu" ]]; then
    TRAIN_WHISPER_DEVICE="auto"
  else
    TRAIN_WHISPER_DEVICE="cuda"
  fi
fi

if [[ -z "$TRAIN_WHISPER_COMPUTE_TYPE" && "$TRAIN_WHISPER_DEVICE" == "cuda" ]]; then
  TRAIN_WHISPER_COMPUTE_TYPE="float16"
fi

DOCKER_ARGS=(
  --rm
  --ipc=host
  --shm-size=16g
  --add-host=host.docker.internal:host-gateway
  --user "$CONTAINER_USER"
  --env-file "$ENV_FILE"
  -e "HOME=/app/backend/tmp"
  -e "MPLCONFIGDIR=/app/backend/tmp/matplotlib"
  -e "OLLAMA_HOST=${OLLAMA_HOST:-http://host.docker.internal:11434}"
  -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES_VALUE}"
  -e "WHISPER_BACKEND=${TRAIN_WHISPER_BACKEND}"
  -e "WHISPER_DEVICE=${TRAIN_WHISPER_DEVICE}"
  -e "WHISPER_COMPUTE_TYPE=${TRAIN_WHISPER_COMPUTE_TYPE}"
  -e "CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS=${TRAIN_FULL_VOD_AUDIO_WORKERS}"
  -v "$ARTIFACT_DIR:/app/backend/ml/artifacts"
  -v "$TMP_DIR:/app/backend/tmp"
)

if [[ "$BACKEND_RUNTIME" == "cuda" || "$IMAGE_MODE" == "legacy" ]]; then
  DOCKER_ARGS+=(--gpus all)
fi

if [[ -d "$PREWARMED_MODEL_DIR" ]]; then
  DOCKER_ARGS+=(
    -v "$PREWARMED_MODEL_DIR:/app/backend/tmp/cache/sentence_transformers_local:ro"
  )
fi

if [[ -n "$ENTRYPOINT_OVERRIDE" ]]; then
  DOCKER_ARGS+=(
    --entrypoint "$ENTRYPOINT_OVERRIDE"
  )
  TRAIN_COMMAND=("$@")
else
  if [[ "$IMAGE_MODE" == "backend" ]]; then
    DOCKER_ARGS+=(--entrypoint python)
    TRAIN_COMMAND=(-m ml.local_train "$@")
  else
    TRAIN_COMMAND=("$@")
  fi
fi

docker run \
  "${DOCKER_ARGS[@]}" \
  "$IMAGE_TAG" \
  "${TRAIN_COMMAND[@]}"
