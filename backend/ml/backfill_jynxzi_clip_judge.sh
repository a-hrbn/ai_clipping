#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$BACKEND_DIR/.venv/bin/python"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$BACKEND_DIR/tmp/matplotlib}"
export PYTHONPATH="${PYTHONPATH:-$BACKEND_DIR}"
export AI_CLIPPING_DISABLE_FACE_DETECTION="${AI_CLIPPING_DISABLE_FACE_DETECTION:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"

exec "$PYTHON_BIN" "$BACKEND_DIR/ml/backfill_clip_judge_snapshots.py" \
  --profile jynxzi \
  --broadcaster jynxzi \
  "$@"
