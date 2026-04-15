#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$BACKEND_DIR/.venv/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$BACKEND_DIR/tmp/matplotlib}"

LABELS_PATH="$BACKEND_DIR/ml/artifacts/jynxzi_twitch_clip_labels.jsonl"
PROGRESS_PATH="$BACKEND_DIR/ml/artifacts/jynxzi_twitch_clip_label_progress.json"
DATASET_PATH="$BACKEND_DIR/ml/artifacts/jynxzi_vod_training_rows.jsonl"
SHORT_FORM_PATH="$BACKEND_DIR/ml/artifacts/jynxzi_training_rows.jsonl"
ARTIFACT_DIR="$BACKEND_DIR/ml/artifacts"
ARTIFACT_JSON_PATH="$ARTIFACT_DIR/jynxzi_ranker.json"
BOUNDARY_ARTIFACT_JSON_PATH="$ARTIFACT_DIR/jynxzi_boundary_ranker.json"
EVAL_DATASET_PATH="$ARTIFACT_DIR/jynxzi_vod_eval_dataset.jsonl"
EVAL_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_vod_eval_dataset_summary.json"
RANKER_EVAL_PATH="$ARTIFACT_DIR/jynxzi_ranker_eval.json"
PIPELINE_EVAL_PATH="$ARTIFACT_DIR/jynxzi_pipeline_eval.json"
WORK_DIR_BASE="${AI_CLIPPING_TRAIN_TMPDIR:-$BACKEND_DIR/tmp/training_runs}"
WORK_DIR="$WORK_DIR_BASE/jynxzi_resume"

cd "$BACKEND_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv interpreter: $PYTHON_BIN" >&2
  echo "Create backend/.venv and install the backend dependencies before training." >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR" "$MPLCONFIGDIR" "$WORK_DIR_BASE" "$WORK_DIR"

DATE_FROM=""
DATE_TO=""
SKIP_COVERED=1
MERGE_OUTPUT=1

usage() {
  cat <<'EOF'
Usage: backend/ml/train_jynxzi_vod.sh [DATE_FROM] [DATE_TO] [options]

Options:
  --date-from YYYY-MM-DD
  --date-to YYYY-MM-DD
  --skip-covered
  --no-skip-covered
  --merge-output
  --no-merge-output
  --help
EOF
}

POSITIONAL_DATES=()
POSITIONAL_DATE_FROM=""
POSITIONAL_DATE_TO=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --date-from)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --date-from" >&2
        exit 1
      fi
      DATE_FROM="$2"
      shift 2
      ;;
    --date-to)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --date-to" >&2
        exit 1
      fi
      DATE_TO="$2"
      shift 2
      ;;
    --skip-covered)
      SKIP_COVERED=1
      shift
      ;;
    --no-skip-covered)
      SKIP_COVERED=0
      shift
      ;;
    --merge-output)
      MERGE_OUTPUT=1
      shift
      ;;
    --no-merge-output)
      MERGE_OUTPUT=0
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -z "$POSITIONAL_DATE_FROM" ]]; then
        POSITIONAL_DATE_FROM="$1"
      elif [[ -z "$POSITIONAL_DATE_TO" ]]; then
        POSITIONAL_DATE_TO="$1"
      else
        echo "Too many positional arguments." >&2
        usage >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$DATE_FROM" && -n "$POSITIONAL_DATE_FROM" ]]; then
  DATE_FROM="$POSITIONAL_DATE_FROM"
fi

if [[ -z "$DATE_TO" && -n "$POSITIONAL_DATE_TO" ]]; then
  DATE_TO="$POSITIONAL_DATE_TO"
fi

if [[ -z "$DATE_FROM" ]]; then
  DATE_FROM="$("$PYTHON_BIN" -c 'from datetime import date, timedelta; print((date.today() - timedelta(days=30)).isoformat())')"
fi

if [[ -z "$DATE_TO" ]]; then
  DATE_TO="$("$PYTHON_BIN" -c 'from datetime import date; print(date.today().isoformat())')"
fi

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

dataset_label_state() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(f"missing:{path}")
    raise SystemExit(0)

rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    print(f"empty:{path}")
    raise SystemExit(0)

positive = sum(1 for row in rows if int(row.get("label", 0)) == 1)
negative = sum(1 for row in rows if int(row.get("label", 0)) == 0)
if positive > 0 and negative > 0:
    print(f"balanced:rows={len(rows)}:positive={positive}:negative={negative}")
elif positive > 0:
    print(f"one_class_positive:rows={len(rows)}:positive={positive}:negative={negative}")
else:
    print(f"one_class_negative:rows={len(rows)}:positive={positive}:negative={negative}")
PY
}

run_vod_training() {
  "$PYTHON_BIN" ml/train_vod_ranker.py \
    --labels "$LABELS_PATH" \
    --profile jynxzi \
    --dataset-out "$DATASET_PATH" \
    --artifact-dir "$ARTIFACT_DIR" \
    --work-dir "$WORK_DIR" \
    --include-short-form "$SHORT_FORM_PATH"
}

generate_short_form_backfill() {
  echo "Generating short-form backfill for jynxzi..."
  "$PYTHON_BIN" ml/auto_train_short_form.py \
    --per-streamer \
    --only-streamers jynxzi \
    --training-mode short-form \
    --artifact-dir "$ARTIFACT_DIR" \
    --work-dir "$BACKEND_DIR/tmp/short_form_training"
}

if [[ ! -f "$SHORT_FORM_PATH" ]]; then
  echo "Warning: optional short-form dataset not found at $SHORT_FORM_PATH" >&2
  echo "Training will continue with Twitch VOD labels only." >&2
fi

echo "Using training temp base: $WORK_DIR_BASE"
echo "Using resumable work dir: $WORK_DIR"
echo "Trainiere VOD von ${DATE_FROM} bis ${DATE_TO}"
echo "Sammle Twitch-Clip-Labels für jynxzi..."
COLLECT_ARGS=(
  --broadcaster jynxzi
  --date-from "$DATE_FROM"
  --date-to "$DATE_TO"
  --progress-file "$PROGRESS_PATH"
  --output "$LABELS_PATH"
)
if [[ "$SKIP_COVERED" -eq 1 ]]; then
  COLLECT_ARGS+=(--skip-covered)
fi
if [[ "$MERGE_OUTPUT" -eq 1 ]]; then
  COLLECT_ARGS+=(--merge-output)
fi
"$PYTHON_BIN" ml/collect_twitch_clip_labels.py "${COLLECT_ARGS[@]}"

SHOULD_TRAIN="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path

labels = Path("ml/artifacts/jynxzi_twitch_clip_labels.jsonl")
dataset = Path("ml/artifacts/jynxzi_vod_training_rows.jsonl")
artifact = Path("ml/artifacts/jynxzi_ranker.json")

if not labels.exists():
    print("yes")
elif not dataset.exists() or not artifact.exists():
    print("yes")
else:
    labels_mtime = labels.stat().st_mtime
    dataset_mtime = dataset.stat().st_mtime
    artifact_mtime = artifact.stat().st_mtime
    print("yes" if labels_mtime > min(dataset_mtime, artifact_mtime) else "no")
PY
)"

SHOULD_EVALUATE="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path

dataset = Path("ml/artifacts/jynxzi_vod_training_rows.jsonl")
artifact = Path("ml/artifacts/jynxzi_ranker.json")
eval_dataset = Path("ml/artifacts/jynxzi_vod_eval_dataset.jsonl")
eval_summary = Path("ml/artifacts/jynxzi_vod_eval_dataset_summary.json")
ranker_eval = Path("ml/artifacts/jynxzi_ranker_eval.json")
pipeline_eval = Path("ml/artifacts/jynxzi_pipeline_eval.json")

if not dataset.exists() or not artifact.exists():
    print("no")
else:
    upstream_mtime = max(dataset.stat().st_mtime, artifact.stat().st_mtime)
    outputs = [eval_dataset, eval_summary, ranker_eval, pipeline_eval]
    if any(not path.exists() for path in outputs):
        print("yes")
    else:
        oldest_output = min(path.stat().st_mtime for path in outputs)
        print("yes" if oldest_output < upstream_mtime else "no")
PY
)"

if [[ "$SHOULD_TRAIN" != "yes" && "$SHOULD_EVALUATE" != "yes" ]]; then
  echo "Keine neuen Label-Daten seit dem letzten Dataset-/Modellstand. Überspringe Training und Evaluation."
  echo "Labels:         $LABELS_PATH"
  echo "Dataset:        $DATASET_PATH"
  echo "Artifact:       $ARTIFACT_JSON_PATH"
  echo "Eval dataset:   $EVAL_DATASET_PATH"
  echo "Ranker eval:    $RANKER_EVAL_PATH"
  echo "Pipeline eval:  $PIPELINE_EVAL_PATH"
  exit 0
fi

if [[ "$SHOULD_TRAIN" == "yes" ]]; then
  echo "Training VOD-first ranker for jynxzi..."
  if ! run_vod_training; then
    DATASET_STATE="$(dataset_label_state "$DATASET_PATH")"
    SHORT_FORM_STATE="$(dataset_label_state "$SHORT_FORM_PATH")"
    echo "Initial VOD-first training failed." >&2
    echo "Dataset state:     $DATASET_STATE" >&2
    echo "Short-form state:  $SHORT_FORM_STATE" >&2

    if [[ "$DATASET_STATE" == one_class_* || "$SHORT_FORM_STATE" == missing:* || "$SHORT_FORM_STATE" == empty:* || "$SHORT_FORM_STATE" == one_class_* ]]; then
      generate_short_form_backfill
      SHORT_FORM_STATE="$(dataset_label_state "$SHORT_FORM_PATH")"
      echo "Short-form backfill state: $SHORT_FORM_STATE"
      if [[ "$SHORT_FORM_STATE" != balanced:* ]]; then
        echo "Short-form backfill did not produce both positive and negative examples. Aborting." >&2
        exit 1
      fi
      echo "Retrying VOD-first training with short-form supplemental rows..."
      run_vod_training
    else
      exit 1
    fi
  fi
else
  echo "Training übersprungen. Nutze vorhandenes Dataset und Artefakt für die neue Evaluation."
fi

echo "Building eval dataset..."
"$PYTHON_BIN" ml/build_eval_dataset.py \
  --dataset "$DATASET_PATH" \
  --output "$EVAL_DATASET_PATH" \
  --summary-out "$EVAL_SUMMARY_PATH"

echo "Evaluating trained ranker against heuristic baseline..."
"$PYTHON_BIN" ml/evaluate_ranker.py \
  --dataset "$EVAL_DATASET_PATH" \
  --profile jynxzi \
  --artifact-dir "$ARTIFACT_DIR" \
  > "$RANKER_EVAL_PATH"

echo "Evaluating heuristic pipeline ranking..."
"$PYTHON_BIN" ml/evaluate_pipeline.py \
  --dataset "$EVAL_DATASET_PATH" \
  --profile jynxzi \
  > "$PIPELINE_EVAL_PATH"

echo "Evaluating boundary supervision coverage..."
"$PYTHON_BIN" ml/evaluate_boundary_refinement.py \
  --dataset "$DATASET_PATH"

echo "Evaluating rollout recommendation for ENABLE_TRAINED_RANKER_V2..."
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

ranker_eval = Path("ml/artifacts/jynxzi_ranker_eval.json")
if not ranker_eval.exists():
    raise SystemExit("Missing ranker eval output: ml/artifacts/jynxzi_ranker_eval.json")

report = json.loads(ranker_eval.read_text(encoding="utf-8"))
delta = report.get("metrics", {}).get("delta", {})
trained = report.get("metrics", {}).get("trained_runtime", {})
baseline = report.get("metrics", {}).get("baseline_heuristic", {})

ndcg_delta = float(delta.get("ndcg_at_5", 0.0))
precision_delta = float(delta.get("precision_at_5", 0.0))
accuracy_delta = float(delta.get("accuracy", 0.0))
recommend_enable = (
    ndcg_delta >= 0.02
    and precision_delta >= 0.0
    and accuracy_delta >= -0.02
)

summary = {
    "recommend_enable_trained_ranker_v2": recommend_enable,
    "reason": (
        "trained_runtime_outperforms_or_matches_baseline"
        if recommend_enable
        else "keep_heuristic_default_until_eval_is_stronger"
    ),
    "trained_runtime": trained,
    "baseline_heuristic": baseline,
    "delta": delta,
}
print(json.dumps(summary, ensure_ascii=True, indent=2))
PY

echo
echo "Done."
echo "Labels:         $LABELS_PATH"
echo "Dataset:        $DATASET_PATH"
echo "Artifact:       $ARTIFACT_JSON_PATH"
echo "Boundary:       $BOUNDARY_ARTIFACT_JSON_PATH"
echo "Eval dataset:   $EVAL_DATASET_PATH"
echo "Eval summary:   $EVAL_SUMMARY_PATH"
echo "Ranker eval:    $RANKER_EVAL_PATH"
echo "Pipeline eval:  $PIPELINE_EVAL_PATH"
