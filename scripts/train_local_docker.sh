#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_TAG="${AI_CLIPPING_LOCAL_TRAIN_IMAGE:-ai-clipping-train-local:latest}"
DOCKERFILE_PATH="${AI_CLIPPING_LOCAL_TRAIN_DOCKERFILE:-$REPO_ROOT/backend/Dockerfile.local-train}"
ENV_FILE="${AI_CLIPPING_LOCAL_TRAIN_ENV_FILE:-$REPO_ROOT/backend/.env}"
ARTIFACT_DIR="$REPO_ROOT/backend/ml/artifacts"
TMP_DIR="$REPO_ROOT/backend/tmp"
PREWARMED_MODEL_DIR="${AI_CLIPPING_LOCAL_PREWARMED_MODEL_DIR:-$REPO_ROOT/tmp/prewarmed_sentence_transformers_local}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR" "$TMP_DIR"

docker build \
  -f "$DOCKERFILE_PATH" \
  -t "$IMAGE_TAG" \
  "$REPO_ROOT"

DOCKER_ARGS=(
  --rm
  --gpus all
  --ipc=host
  --shm-size=16g
  --add-host=host.docker.internal:host-gateway
  --env-file "$ENV_FILE"
  -e "OLLAMA_HOST=${OLLAMA_HOST:-http://host.docker.internal:11434}"
  -v "$ARTIFACT_DIR:/app/backend/ml/artifacts"
  -v "$TMP_DIR:/app/backend/tmp"
)

if [[ -d "$PREWARMED_MODEL_DIR" ]]; then
  DOCKER_ARGS+=(
    -v "$PREWARMED_MODEL_DIR:/app/backend/tmp/cache/sentence_transformers_local:ro"
  )
fi

docker run \
  "${DOCKER_ARGS[@]}" \
  "$IMAGE_TAG" \
  "$@"
