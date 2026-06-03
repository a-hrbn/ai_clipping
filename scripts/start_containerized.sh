#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not on PATH." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is required. Install Docker Desktop or the docker compose plugin." >&2
  exit 1
fi

normalize_boolish() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

stop_pid_file() {
  local pid_file="$1"
  if [ ! -f "$pid_file" ]; then
    return 0
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
  fi
  rm -f "$pid_file"
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local deadline=$((SECONDS + 60))
  until curl -fsS --max-time 5 "$url" >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "Timed out waiting for $label at $url." >&2
      return 1
    fi
    sleep 2
  done
}

extract_ngrok_url() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
for line in reversed(log_path.read_text(errors="replace").splitlines()):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    url = row.get("url")
    if row.get("msg") == "started tunnel" and isinstance(url, str) and url.startswith("https://"):
        print(url)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

start_ngrok_remote() {
  local ngrok_bin="${AI_CLIPPING_NGROK_BIN:-}"
  if [ -z "$ngrok_bin" ]; then
    if command -v ngrok >/dev/null 2>&1; then
      ngrok_bin="$(command -v ngrok)"
    elif [ -x "$HOME/.local/bin/ngrok" ]; then
      ngrok_bin="$HOME/.local/bin/ngrok"
    else
      echo "AI_CLIPPING_REMOTE=ngrok requested, but ngrok was not found on PATH or at ~/.local/bin/ngrok." >&2
      exit 1
    fi
  fi

  if [ ! -x "$ngrok_bin" ]; then
    echo "ngrok binary is not executable: $ngrok_bin" >&2
    exit 1
  fi

  mkdir -p backend/tmp/ngrok
  stop_pid_file backend/tmp/ngrok/ngrok.pid
  stop_pid_file backend/tmp/ngrok/proxy.pid
  rm -f backend/tmp/ngrok/ngrok.log backend/tmp/ngrok/proxy.log

  wait_for_http http://127.0.0.1:3000 frontend
  wait_for_http http://127.0.0.1:3000/api/status "frontend API rewrite"

  setsid "$ngrok_bin" http 3000 --log=stdout --log-format=json \
    > backend/tmp/ngrok/ngrok.log 2>&1 < /dev/null &
  echo "$!" > backend/tmp/ngrok/ngrok.pid

  local public_url=""
  local deadline=$((SECONDS + 30))
  while [ "$SECONDS" -lt "$deadline" ]; do
    public_url="$(extract_ngrok_url backend/tmp/ngrok/ngrok.log 2>/dev/null || true)"
    if [ -n "$public_url" ]; then
      break
    fi
    if ! kill -0 "$(cat backend/tmp/ngrok/ngrok.pid)" >/dev/null 2>&1; then
      echo "ngrok exited before a public URL was created. Log:" >&2
      tail -80 backend/tmp/ngrok/ngrok.log >&2 || true
      exit 1
    fi
    sleep 1
  done

  if [ -z "$public_url" ]; then
    echo "Timed out waiting for ngrok public URL. Log:" >&2
    tail -80 backend/tmp/ngrok/ngrok.log >&2 || true
    exit 1
  fi

  echo
  echo "Remote frontend: $public_url"
  echo "Remote backend:  $public_url/api/status"
  echo
  echo "Stop remote tunnel:"
  echo "  kill \"\$(cat backend/tmp/ngrok/ngrok.pid)\""
}

if [ ! -f backend/.env ]; then
  cp backend/.env.example backend/.env
  echo "Created backend/.env from backend/.env.example. Fill API keys there if needed."
fi

if [ ! -f frontend/.env.local ]; then
  cp frontend/.env.example frontend/.env.local
fi

docker_gpu_available() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  docker run --rm --gpus all --entrypoint /bin/sh alpine:3.20 -c true >/dev/null 2>&1
}

BACKEND_RUNTIME="${AI_CLIPPING_BACKEND_RUNTIME:-auto}"
REMOTE_MODE="$(normalize_boolish "${AI_CLIPPING_REMOTE:-ngrok}")"
DEV_MODE="$(normalize_boolish "${AI_CLIPPING_DEV:-1}")"
REBUILD_MODE="$(normalize_boolish "${AI_CLIPPING_REBUILD:-0}")"
COMPOSE_FILES=(-f docker-compose.yml)
ENABLE_OLLAMA="${AI_CLIPPING_ENABLE_OLLAMA:-}"

if [ -z "$ENABLE_OLLAMA" ] && [ -f backend/.env ]; then
  if grep -Eq '^[[:space:]]*JUDGE_PROVIDER[[:space:]]*=[[:space:]]*ollama[[:space:]]*$' backend/.env; then
    ENABLE_OLLAMA=1
  fi
fi
ENABLE_OLLAMA_NORMALIZED="$(normalize_boolish "${ENABLE_OLLAMA:-0}")"

case "$REMOTE_MODE" in
  none|off|false|0|"")
    REMOTE_MODE="none"
    ;;
  ngrok)
    export AI_CLIPPING_PUBLIC_API_URL="${AI_CLIPPING_PUBLIC_API_URL-}"
    export AI_CLIPPING_INTERNAL_API_URL="${AI_CLIPPING_INTERNAL_API_URL:-http://backend:8000}"
    ;;
  *)
    echo "Invalid AI_CLIPPING_REMOTE=$REMOTE_MODE. Use ngrok, none, off, false, or 0." >&2
    exit 1
    ;;
esac

case "$BACKEND_RUNTIME" in
  auto)
    if docker_gpu_available; then
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
    echo "Invalid AI_CLIPPING_BACKEND_RUNTIME=$BACKEND_RUNTIME. Use auto, cuda, gpu, or cpu." >&2
    exit 1
    ;;
esac

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

if [ "$BACKEND_RUNTIME" = "cuda" ]; then
  if ! docker_gpu_available; then
    if [ "${AI_CLIPPING_BACKEND_RUNTIME:-auto}" = "auto" ]; then
      BACKEND_RUNTIME="cpu"
    else
      echo "CUDA backend requested, but Docker GPU access is not available." >&2
      echo "Check NVIDIA Container Toolkit / Docker Desktop GPU integration, or use AI_CLIPPING_BACKEND_RUNTIME=cpu." >&2
      exit 1
    fi
  else
    COMPOSE_FILES+=(-f docker-compose.cuda.yml)
  fi
fi

case "$ENABLE_OLLAMA_NORMALIZED" in
  1|true|yes|on)
    COMPOSE_FILES+=(-f docker-compose.ollama.yml)
    ;;
  0|false|no|off|"")
    ;;
  *)
    echo "Invalid AI_CLIPPING_ENABLE_OLLAMA=$ENABLE_OLLAMA. Use true/false." >&2
    exit 1
    ;;
esac

mkdir -p \
  backend/tmp/storage \
  backend/tmp/cache \
  backend/tmp/matplotlib \
  backend/ml/artifacts

echo "Starting AI Clipping with backend runtime: $BACKEND_RUNTIME"
if [ "$DEV_MODE" = "1" ] || [ "$DEV_MODE" = "true" ] || [ "$DEV_MODE" = "yes" ] || [ "$DEV_MODE" = "on" ]; then
  echo "Dev live reload: enabled"
  echo "Set AI_CLIPPING_DEV=0 for image-only production-style containers."
fi
if [ "$ENABLE_OLLAMA_NORMALIZED" = "1" ] || [ "$ENABLE_OLLAMA_NORMALIZED" = "true" ] || [ "$ENABLE_OLLAMA_NORMALIZED" = "yes" ] || [ "$ENABLE_OLLAMA_NORMALIZED" = "on" ]; then
  echo "Ollama service: enabled"
fi
if [ "$REMOTE_MODE" = "ngrok" ]; then
  echo "Remote mode: ngrok"
  echo "Set AI_CLIPPING_REMOTE=none to start without a remote tunnel."
fi
case "$REBUILD_MODE" in
  1|true|yes|on)
    docker compose "${COMPOSE_FILES[@]}" up --build -d
    ;;
  0|false|no|off|"")
    docker compose "${COMPOSE_FILES[@]}" up -d
    ;;
  *)
    echo "Invalid AI_CLIPPING_REBUILD=$REBUILD_MODE. Use true/false." >&2
    exit 1
    ;;
esac

if [ "$REMOTE_MODE" = "ngrok" ]; then
  start_ngrok_remote
fi

cat <<'EOF'

AI Clipping containers are starting.

Frontend: http://localhost:3000
Backend:  http://localhost:8000

Remote ngrok tunnel starts by default.
Disable it with:
  AI_CLIPPING_REMOTE=none ./scripts/start_containerized.sh

Useful commands:
  docker compose logs -f backend
  docker compose logs -f frontend
  docker compose down

Development defaults:
  Backend and frontend source are bind-mounted with reload/watch enabled.
  Normal code edits should not require a container rebuild or manual restart.
  Use AI_CLIPPING_REBUILD=1 only after dependency or Dockerfile changes.
  Use AI_CLIPPING_DEV=0 for image-only production-style containers.
EOF
