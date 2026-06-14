#!/usr/bin/env bash

# Resolve a working docker binary. Under Docker Desktop on WSL *without* WSL
# integration enabled for this distro, bare `docker` is a stub that errors
# ("could not be found in this WSL 2 distro"); the Windows `docker.exe` still
# works via interop. Prefer native `docker` when its daemon is reachable, else
# fall back to docker.exe. Override by exporting DOCKER=... .
resolve_docker_bin() {
  if command -v docker >/dev/null 2>&1 && docker version >/dev/null 2>&1; then
    printf 'docker'
  elif command -v docker.exe >/dev/null 2>&1 && docker.exe version >/dev/null 2>&1; then
    printf 'docker.exe'
  else
    printf 'docker'   # nothing works — let callers fail with docker's own message
  fi
}
DOCKER="${DOCKER:-$(resolve_docker_bin)}"

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

docker_gpu_available() {
  if ! command -v "$DOCKER" >/dev/null 2>&1; then
    return 1
  fi
  "$DOCKER" run --rm --gpus all --entrypoint /bin/sh alpine:3.20 -c true >/dev/null 2>&1
}

backend_image_from_compose() {
  "$DOCKER" compose "$@" config --images | awk '/^ai-clipping-backend/ { print; exit }'
}

image_created_epoch() {
  local image="$1"
  local created
  created="$("$DOCKER" image inspect -f '{{.Created}}' "$image" 2>/dev/null || true)"
  if [ -z "$created" ]; then
    echo 0
    return 0
  fi
  date -d "$created" +%s 2>/dev/null || echo 0
}

container_backend_bind_mount() {
  local container="$1"
  "$DOCKER" inspect -f '{{range .Mounts}}{{if eq .Destination "/app/backend"}}{{.Source}}{{end}}{{end}}' "$container" 2>/dev/null || true
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
  "$DOCKER" compose "$@" build backend
}

compose_service_up_backend() {
  "$DOCKER" compose "$@" up -d backend
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

  if ! "$DOCKER" image inspect "$image" >/dev/null 2>&1; then
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

# --- container runtime helpers ---

# True wenn der genannte Container existiert UND läuft.
container_running() {
  [ "$("$DOCKER" inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]
}

# CUDA-Backend bevorzugen, sonst CPU-Backend; sonst mit Fehler abbrechen.
# Ersetzt den laufenden Prozess durch `docker exec` (exec → kein Rücksprung).
# Container-Namen und Workdir via Env überschreibbar.
exec_in_backend() {
  local module="$1"; shift
  local cuda="${BACKEND_CONTAINER_CUDA:-ai-clipping-backend-cuda}"
  local cpu="${BACKEND_CONTAINER:-ai-clipping-backend}"
  local workdir="${BACKEND_WORKDIR:-/app/backend}"

  local container="$cuda"
  container_running "$container" || container="$cpu"   # Fallback CUDA → CPU
  if ! container_running "$container"; then
    echo "ERROR: No backend container running (tried $cuda, $cpu)." >&2
    echo "Start it first: docker compose -f docker-compose.cuda.yml up -d" >&2
    exit 1
  fi

  exec "$DOCKER" exec -w "$workdir" "$container" python -m "$module" "$@"
}
