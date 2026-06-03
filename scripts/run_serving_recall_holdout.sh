#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
PYTHON_BIN="${PYTHON_BIN:-$BACKEND_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing backend interpreter: $PYTHON_BIN" >&2
  exit 1
fi

cd "$REPO_ROOT"
PYTHONPATH="$BACKEND_DIR${PYTHONPATH:+:$PYTHONPATH}" exec "$PYTHON_BIN" -m ml.run_serving_recall_holdout "$@"
