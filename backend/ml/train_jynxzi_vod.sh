#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$BACKEND_DIR/.venv/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

LABELS_PATH="$BACKEND_DIR/ml/artifacts/jynxzi_twitch_clip_labels.jsonl"
DATASET_PATH="$BACKEND_DIR/ml/artifacts/jynxzi_vod_training_rows.jsonl"
SHORT_FORM_PATH="$BACKEND_DIR/ml/artifacts/real_short_form_training_rows.jsonl"
ARTIFACT_DIR="$BACKEND_DIR/ml/artifacts"

cd "$BACKEND_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv interpreter: $PYTHON_BIN" >&2
  echo "Create backend/.venv and install the backend dependencies before training." >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR" "$MPLCONFIGDIR"

if [[ $# -ge 1 ]]; then
  DATE_FROM="$1"
else
  DATE_FROM="$("$PYTHON_BIN" -c 'from datetime import date, timedelta; print((date.today() - timedelta(days=30)).isoformat())')"
fi

if [[ $# -ge 2 ]]; then
  DATE_TO="$2"
else
  DATE_TO="$("$PYTHON_BIN" -c 'from datetime import date; print(date.today().isoformat())')"
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ai_clipping_twitch_training.XXXXXX")"

cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT INT TERM

export DATE_FROM DATE_TO
"$PYTHON_BIN" - <<'PY'
from datetime import datetime
import os

from config import settings

missing = []
if not settings.twitch_client_id:
    missing.append("TWITCH_CLIENT_ID")
if not settings.twitch_client_secret:
    missing.append("TWITCH_CLIENT_SECRET")
if missing:
    raise SystemExit(
        "Missing required backend/.env settings: " + ", ".join(missing)
    )

for value, label in (
    (settings.storage_dir, "STORAGE_DIR"),
    (settings.whisper_model, "WHISPER_MODEL"),
):
    if not str(value).strip():
        raise SystemExit(f"Missing required backend/.env setting: {label}")

for raw in (os.environ["DATE_FROM"], os.environ["DATE_TO"]):
    datetime.fromisoformat(raw)
PY

if [[ ! -f "$SHORT_FORM_PATH" ]]; then
  echo "Warning: optional short-form dataset not found at $SHORT_FORM_PATH" >&2
  echo "Training will continue with Twitch VOD labels only." >&2
fi

echo "Using isolated work dir: $WORK_DIR"
echo "Collecting Twitch clip labels for jynxzi (${DATE_FROM} to ${DATE_TO})..."
"$PYTHON_BIN" ml/collect_twitch_clip_labels.py \
  --broadcaster jynxzi \
  --date-from "$DATE_FROM" \
  --date-to "$DATE_TO" \
  --output "$LABELS_PATH"

echo "Training VOD-first ranker for jynxzi..."
"$PYTHON_BIN" ml/train_vod_ranker.py \
  --labels "$LABELS_PATH" \
  --profile jynxzi \
  --dataset-out "$DATASET_PATH" \
  --artifact-dir "$ARTIFACT_DIR" \
  --work-dir "$WORK_DIR" \
  --include-short-form "$SHORT_FORM_PATH"

echo "Evaluating resulting dataset..."
"$PYTHON_BIN" ml/evaluate_twitch_ranker.py \
  --dataset "$DATASET_PATH"

echo
echo "Done."
echo "Labels:   $LABELS_PATH"
echo "Dataset:  $DATASET_PATH"
echo "Artifact: $ARTIFACT_DIR/jynxzi_ranker.json"
