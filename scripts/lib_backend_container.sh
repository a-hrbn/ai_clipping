#!/usr/bin/env bash

normalize_boolish() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

is_truthy() {
  case "$(normalize_boolish "${1:-}")" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_falsey() {
  case "$(normalize_boolish "${1:-}")" in
    0|false|no|off|"")
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

docker_gpu_available() {
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi
  docker run --rm --gpus all --entrypoint /bin/sh alpine:3.20 -c true >/dev/null 2>&1
}

backend_image_from_compose() {
  docker compose "$@" config --images | awk '/^ai-clipping-backend/ { print; exit }'
}

image_created_epoch() {
  local image="$1"
  local created
  created="$(docker image inspect -f '{{.Created}}' "$image" 2>/dev/null || true)"
  if [ -z "$created" ]; then
    echo 0
    return 0
  fi
  date -d "$created" +%s 2>/dev/null || echo 0
}

container_backend_bind_mount() {
  local container="$1"
  docker inspect -f '{{range .Mounts}}{{if eq .Destination "/app/backend"}}{{.Source}}{{end}}{{end}}' "$container" 2>/dev/null || true
}

latest_backend_build_input_epoch() {
  local root_dir="$1"
  local dev_like="$2"
  local find_args=(
    "$root_dir/backend"
    "$root_dir/docker-compose.yml"
    "$root_dir/docker-compose.cuda.yml"
    "$root_dir/docker-compose.ollama.yml"
    "$root_dir/docker-compose.dev.yml"
  )

  if is_truthy "$dev_like"; then
    find_args=(
      "$root_dir/backend/Dockerfile"
      "$root_dir/backend/Dockerfile.cuda"
      "$root_dir/backend/pyproject.toml"
      "$root_dir/backend/.env.example"
      "$root_dir/docker-compose.yml"
      "$root_dir/docker-compose.cuda.yml"
      "$root_dir/docker-compose.ollama.yml"
      "$root_dir/docker-compose.dev.yml"
    )
  fi

  find "${find_args[@]}" \
    -type f \
    ! -path "$root_dir/backend/tmp/*" \
    ! -path "$root_dir/backend/ml/artifacts/*" \
    ! -path "$root_dir/backend/graphify-out/*" \
    ! -path "$root_dir/backend/.venv/*" \
    ! -path "$root_dir/backend/__pycache__/*" \
    -printf '%T@\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | awk -F. '{ print $1 }'
}

compose_service_build_backend() {
  docker compose "$@" build backend
}

compose_service_up_backend() {
  docker compose "$@" up -d backend
}

wait_for_ollama_container() {
  local container="${1:-ai-clipping-ollama}"
  local timeout_sec="${2:-120}"
  local deadline=$((SECONDS + timeout_sec))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! docker inspect "$container" >/dev/null 2>&1; then
      sleep 2
      continue
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" != "true" ]; then
      sleep 2
      continue
    fi
    if docker exec "$container" ollama list >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for Ollama container: $container" >&2
  docker inspect -f 'status={{.State.Status}}' "$container" >&2 || true
  return 1
}

ensure_ollama_model_available() {
  local container="${1:-ai-clipping-ollama}"
  local model="${2:-}"
  local pull_model="${3:-0}"

  if [ -z "$model" ]; then
    return 0
  fi

  if docker exec "$container" ollama show "$model" >/dev/null 2>&1; then
    return 0
  fi

  if ! is_truthy "$pull_model"; then
    echo "Ollama is running, but model '$model' is not installed in $container." >&2
    echo "Set AI_CLIPPING_OLLAMA_PULL_MODEL=1 to let the wrapper pull it automatically." >&2
    return 1
  fi

  echo "Pulling Ollama model '$model' in $container."
  docker exec "$container" ollama pull "$model"
}

wait_for_backend_ollama_access() {
  local backend_container="$1"
  local timeout_sec="${2:-60}"
  local deadline=$((SECONDS + timeout_sec))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if docker exec "$backend_container" python -c 'import os, urllib.request; host=os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/"); urllib.request.urlopen(host + "/api/tags", timeout=5).read(); print(host)' >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for backend container to reach Ollama." >&2
  docker exec "$backend_container" python -c 'import os; print("OLLAMA_HOST=" + os.environ.get("OLLAMA_HOST", ""))' >&2 || true
  return 1
}

ensure_backend_ollama_ready() {
  local backend_container="${1:-ai-clipping-backend}"
  local ollama_container="${2:-ai-clipping-ollama}"
  local model="${3:-}"
  local pull_model="${4:-0}"
  local timeout_sec="${5:-180}"

  wait_for_ollama_container "$ollama_container" "$timeout_sec"
  ensure_ollama_model_available "$ollama_container" "$model" "$pull_model"
  wait_for_backend_ollama_access "$backend_container" 60
}

ensure_backend_image_current() {
  local root_dir="$1"
  local container="$2"
  local dev_mode="$3"
  local rebuild_mode="$4"
  shift 4
  local compose_args=("$@")
  local normalized_rebuild
  normalized_rebuild="$(normalize_boolish "$rebuild_mode")"

  case "$normalized_rebuild" in
    1|true|yes|on)
      echo "Rebuilding backend image because AI_CLIPPING_REBUILD=$rebuild_mode."
      compose_service_build_backend "${compose_args[@]}"
      return 0
      ;;
    0|false|no|off)
      return 0
      ;;
    auto|"")
      ;;
    *)
      echo "Invalid AI_CLIPPING_REBUILD=$rebuild_mode. Use auto, true, or false." >&2
      return 1
      ;;
  esac

  local image
  image="$(backend_image_from_compose "${compose_args[@]}")"
  if [ -z "$image" ]; then
    echo "Could not determine backend image from docker compose config." >&2
    return 1
  fi

  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Backend image missing: $image. Building backend image."
    compose_service_build_backend "${compose_args[@]}"
    return 0
  fi

  local dev_like="$dev_mode"
  local mount_source
  mount_source="$(container_backend_bind_mount "$container")"
  if [ -n "$mount_source" ]; then
    dev_like=1
  fi

  local image_epoch latest_epoch
  image_epoch="$(image_created_epoch "$image")"
  latest_epoch="$(latest_backend_build_input_epoch "$root_dir" "$dev_like")"
  latest_epoch="${latest_epoch:-0}"

  if [ "$latest_epoch" -gt "$image_epoch" ]; then
    echo "Backend image is older than build inputs. Rebuilding $image."
    echo "  image_created_epoch=$image_epoch"
    echo "  latest_input_epoch=$latest_epoch"
    compose_service_build_backend "${compose_args[@]}"
  else
    echo "Backend image is current: $image"
  fi
}

ensure_backend_container_running() {
  local container="$1"
  shift
  local compose_args=("$@")
  if docker inspect "$container" >/dev/null 2>&1 \
    && [ "$(docker inspect -f '{{.State.Running}}' "$container")" = "true" ]; then
    return 0
  fi
  echo "Starting backend container: $container"
  compose_service_up_backend "${compose_args[@]}"
}

wait_for_backend_container() {
  local container="$1"
  local timeout_sec="${2:-120}"
  local deadline=$((SECONDS + timeout_sec))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! docker inspect "$container" >/dev/null 2>&1; then
      sleep 2
      continue
    fi
    local running
    running="$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)"
    if [ "$running" != "true" ]; then
      sleep 2
      continue
    fi
    local health
    health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$container" 2>/dev/null || true)"
    if [ -z "$health" ] || [ "$health" = "healthy" ]; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for backend container: $container" >&2
  docker inspect -f 'status={{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{end}}' "$container" >&2 || true
  return 1
}
