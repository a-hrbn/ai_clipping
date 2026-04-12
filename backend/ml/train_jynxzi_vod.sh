#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$BACKEND_DIR/.venv/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

DATE_FROM="${1:-$(date -v-30d +%F)}"
DATE_TO="${2:-$(date +%F)}"

LABELS_PATH="$BACKEND_DIR/ml/artifacts/jynxzi_twitch_clip_labels.jsonl"
DATASET_PATH="$BACKEND_DIR/ml/artifacts/jynxzi_vod_training_rows.jsonl"
SHORT_FORM_PATH="$BACKEND_DIR/ml/artifacts/real_short_form_training_rows.jsonl"
ARTIFACT_DIR="$BACKEND_DIR/ml/artifacts"
WORK_DIR="/tmp/ai_clipping_twitch_training"

cd "$BACKEND_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

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
