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
GOLD_EVAL_ROWS_PATH="$ARTIFACT_DIR/jynxzi_gold_eval_rows.jsonl"
GOLD_EVAL_DATASET_PATH="$ARTIFACT_DIR/jynxzi_gold_eval_dataset.jsonl"
GOLD_EVAL_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_gold_eval_summary.json"
THRESHOLD_CALIBRATION_PATH="$ARTIFACT_DIR/jynxzi_threshold_calibration.json"
GOLD_RANKER_EVAL_PATH="$ARTIFACT_DIR/jynxzi_gold_eval.json"
PROXY_RANKER_EVAL_PATH="$ARTIFACT_DIR/jynxzi_proxy_eval.json"
PIPELINE_EVAL_PATH="$ARTIFACT_DIR/jynxzi_pipeline_eval.json"
RANKER_HOLDOUT_DIAGNOSIS_JSON_PATH="$ARTIFACT_DIR/jynxzi_ranker_holdout_diagnosis.json"
RANKER_HOLDOUT_DIAGNOSIS_CSV_PATH="$ARTIFACT_DIR/jynxzi_ranker_holdout_diagnosis.csv"
HARD_NEGATIVE_ROWS_PATH="$ARTIFACT_DIR/jynxzi_ranker_hard_negative_train_rows.jsonl"
HARD_NEGATIVE_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_ranker_hard_negative_train_summary.json"
WORKFLOW_STATE_PATH="$ARTIFACT_DIR/jynxzi_training_workflow_state.json"
WORK_DIR_BASE="${AI_CLIPPING_TRAIN_TMPDIR:-$BACKEND_DIR/tmp/training_runs}"
WORK_DIR="$WORK_DIR_BASE/jynxzi_resume"
VOD_RANKER_SHORT_FORM_MODE="${VOD_RANKER_SHORT_FORM_MODE:-off}"
VOD_RANKER_GOLD_TRAIN_WEIGHT="${VOD_RANKER_GOLD_TRAIN_WEIGHT:-2.0}"
VOD_RANKER_HARD_NEGATIVE_WEIGHT="${VOD_RANKER_HARD_NEGATIVE_WEIGHT:-4.0}"
SHORT_FORM_BACKFILL_ARTIFACT_DIR="$BACKEND_DIR/tmp/short_form_backfill_artifacts"
SHORT_FORM_BACKFILL_ARTIFACT_JSON_PATH="$SHORT_FORM_BACKFILL_ARTIFACT_DIR/jynxzi_ranker.json"
SHORT_FORM_BACKFILL_ARTIFACT_HEALTH_PATH="$SHORT_FORM_BACKFILL_ARTIFACT_DIR/jynxzi_artifact_health.json"
PERSISTED_SHORT_FORM_BACKFILL_ARTIFACT_JSON_PATH="$ARTIFACT_DIR/jynxzi_short_form_backfill_ranker.json"
PERSISTED_SHORT_FORM_BACKFILL_ARTIFACT_HEALTH_PATH="$ARTIFACT_DIR/jynxzi_short_form_backfill_artifact_health.json"

cd "$BACKEND_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv interpreter: $PYTHON_BIN" >&2
  echo "Create backend/.venv and install the backend dependencies before training." >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR" "$MPLCONFIGDIR" "$WORK_DIR_BASE" "$WORK_DIR"

if [[ "${AI_CLIPPING_CACHE_CLEANUP:-1}" != "0" ]]; then
  "$PYTHON_BIN" -m ml.cache_cleanup --backend-root "$BACKEND_DIR"
fi

DATE_FROM=""
DATE_TO=""
SKIP_COVERED=1
MERGE_OUTPUT=1
ARTIFACT_ONLY=0

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
  --artifact-only
  --no-artifact-only
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
    --artifact-only)
      ARTIFACT_ONLY=1
      shift
      ;;
    --no-artifact-only)
      ARTIFACT_ONLY=0
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
  DATE_FROM="$("$PYTHON_BIN" -c 'from datetime import date, timedelta; print((date.today() - timedelta(days=90)).isoformat())')"
fi

if [[ -z "$DATE_TO" ]]; then
  DATE_TO="$("$PYTHON_BIN" -c 'from datetime import date; print(date.today().isoformat())')"
fi

export SKIP_COVERED MERGE_OUTPUT
export ARTIFACT_ONLY
export DATE_FROM DATE_TO
export VOD_RANKER_SHORT_FORM_MODE
export VOD_RANKER_GOLD_TRAIN_WEIGHT
export VOD_RANKER_HARD_NEGATIVE_WEIGHT
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

CURRENT_WORKFLOW_STATE_JSON="$("$PYTHON_BIN" - <<'PY'
import json
import os

from ml.training_workflow_state import build_jynxzi_workflow_state

payload = build_jynxzi_workflow_state(
    date_from=os.environ["DATE_FROM"],
    date_to=os.environ["DATE_TO"],
    skip_covered=os.environ.get("SKIP_COVERED", "1") == "1",
    merge_output=os.environ.get("MERGE_OUTPUT", "1") == "1",
)
print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
PY
)"
export CURRENT_WORKFLOW_STATE_JSON

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
    --include-short-form "$SHORT_FORM_PATH" \
    --short-form-mode "$VOD_RANKER_SHORT_FORM_MODE" \
    --include-gold-train "$GOLD_EVAL_ROWS_PATH" \
    --gold-train-weight "$VOD_RANKER_GOLD_TRAIN_WEIGHT" \
    --include-hard-negative-train "$HARD_NEGATIVE_ROWS_PATH" \
    --hard-negative-weight "$VOD_RANKER_HARD_NEGATIVE_WEIGHT" \
    --workflow-state-path "$WORKFLOW_STATE_PATH" \
    --workflow-state-json "$CURRENT_WORKFLOW_STATE_JSON"
}

export_gold_eval_rows() {
  "$PYTHON_BIN" ml/export_frontend_gold_eval.py \
    --storage-dir "tmp/storage" \
    --creator-id jynxzi \
    --rows-out "$GOLD_EVAL_ROWS_PATH" \
    --output "$GOLD_EVAL_DATASET_PATH" \
    --summary-out "$GOLD_EVAL_SUMMARY_PATH" \
    --merge-existing \
    --preserve-existing-if-empty
}

export_hard_negative_rows() {
  if [[ ! -f "$RANKER_HOLDOUT_DIAGNOSIS_CSV_PATH" ]]; then
    echo "No ranker holdout diagnosis CSV found at $RANKER_HOLDOUT_DIAGNOSIS_CSV_PATH. Skipping hard-negative export."
    return 0
  fi
  "$PYTHON_BIN" ml/export_ranker_hard_negatives.py \
    --diagnosis-csv "$RANKER_HOLDOUT_DIAGNOSIS_CSV_PATH" \
    --gold-rows "$GOLD_EVAL_ROWS_PATH" \
    --rows-out "$HARD_NEGATIVE_ROWS_PATH" \
    --summary-out "$HARD_NEGATIVE_SUMMARY_PATH" \
    --profile jynxzi
}

generate_short_form_backfill() {
  echo "Generating short-form backfill for jynxzi..."
  mkdir -p "$SHORT_FORM_BACKFILL_ARTIFACT_DIR"
  "$PYTHON_BIN" ml/auto_train_short_form.py \
    --only-streamers jynxzi \
    --profile jynxzi \
    --training-mode short-form \
    --artifact-dir "$SHORT_FORM_BACKFILL_ARTIFACT_DIR" \
    --manifest-out "$SHORT_FORM_BACKFILL_ARTIFACT_DIR/discovered_short_form_manifest.csv" \
    --dataset-out "$SHORT_FORM_PATH" \
    --work-dir "$BACKEND_DIR/tmp/short_form_training"
}

persist_short_form_backfill_artifact() {
  if [[ -f "$SHORT_FORM_BACKFILL_ARTIFACT_JSON_PATH" ]]; then
    cp "$SHORT_FORM_BACKFILL_ARTIFACT_JSON_PATH" "$PERSISTED_SHORT_FORM_BACKFILL_ARTIFACT_JSON_PATH"
    echo "Persisted short-form backfill artifact to $PERSISTED_SHORT_FORM_BACKFILL_ARTIFACT_JSON_PATH"
  fi
  if [[ -f "$SHORT_FORM_BACKFILL_ARTIFACT_HEALTH_PATH" ]]; then
    cp "$SHORT_FORM_BACKFILL_ARTIFACT_HEALTH_PATH" "$PERSISTED_SHORT_FORM_BACKFILL_ARTIFACT_HEALTH_PATH"
    echo "Persisted short-form backfill artifact health to $PERSISTED_SHORT_FORM_BACKFILL_ARTIFACT_HEALTH_PATH"
  fi
}

if [[ "$VOD_RANKER_SHORT_FORM_MODE" == "supplemental" && ! -f "$SHORT_FORM_PATH" ]]; then
  echo "Warning: optional short-form dataset not found at $SHORT_FORM_PATH" >&2
  echo "Training will continue with Twitch VOD labels only." >&2
fi

echo "Using training temp base: $WORK_DIR_BASE"
echo "Using resumable work dir: $WORK_DIR"
echo "Using VOD ranker short-form mode: $VOD_RANKER_SHORT_FORM_MODE"
echo "Using VOD ranker gold-train weight: $VOD_RANKER_GOLD_TRAIN_WEIGHT"
echo "Using VOD ranker hard-negative weight: $VOD_RANKER_HARD_NEGATIVE_WEIGHT"
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

echo "Exporting frontend gold eval rows for training merge..."
export_gold_eval_rows

echo "Exporting diagnosis hard-negative rows for training merge..."
export_hard_negative_rows

SHOULD_TRAIN="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

import os

from ml.training_workflow_state import should_retrain

print(
    "yes"
    if should_retrain(
        labels_path=Path("ml/artifacts/jynxzi_twitch_clip_labels.jsonl"),
        dataset_path=Path("ml/artifacts/jynxzi_vod_training_rows.jsonl"),
        artifact_path=Path("ml/artifacts/jynxzi_ranker.json"),
        state_path=Path("ml/artifacts/jynxzi_training_workflow_state.json"),
        current_state=json.loads(os.environ["CURRENT_WORKFLOW_STATE_JSON"]),
        dependency_paths=[
            Path("ml/artifacts/jynxzi_gold_eval_rows.jsonl"),
            Path("ml/artifacts/jynxzi_ranker_hard_negative_train_rows.jsonl"),
        ],
    )
    else "no"
)
PY
)"

SHOULD_EVALUATE="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

import os

from ml.training_workflow_state import should_reevaluate

print(
    "yes"
    if should_reevaluate(
        dataset_path=Path("ml/artifacts/jynxzi_vod_training_rows.jsonl"),
        artifact_path=Path("ml/artifacts/jynxzi_ranker.json"),
        output_paths=[
            Path("ml/artifacts/jynxzi_vod_eval_dataset.jsonl"),
            Path("ml/artifacts/jynxzi_vod_eval_dataset_summary.json"),
            Path("ml/artifacts/jynxzi_gold_eval.json"),
            Path("ml/artifacts/jynxzi_proxy_eval.json"),
            Path("ml/artifacts/jynxzi_pipeline_eval.json"),
        ],
        dependency_paths=[
            Path("tmp/storage/feedback/gold_eval_updates.jsonl"),
        ],
        state_path=Path("ml/artifacts/jynxzi_training_workflow_state.json"),
        current_state=json.loads(os.environ["CURRENT_WORKFLOW_STATE_JSON"]),
    )
    else "no"
)
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
      persist_short_form_backfill_artifact
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

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from ml.training_workflow_state import write_state

write_state(
    Path("ml/artifacts/jynxzi_training_workflow_state.json"),
    json.loads(os.environ["CURRENT_WORKFLOW_STATE_JSON"]),
)
PY

if [[ "$ARTIFACT_ONLY" -eq 1 ]]; then
  echo "Artifact-only mode enabled. Skipping eval, gold eval, and rollout recommendation."
  echo
  echo "Done."
  echo "Labels:         $LABELS_PATH"
  echo "Dataset:        $DATASET_PATH"
  echo "Artifact:       $ARTIFACT_JSON_PATH"
  echo "Boundary:       $BOUNDARY_ARTIFACT_JSON_PATH"
  exit 0
fi

echo "Building eval dataset..."
"$PYTHON_BIN" ml/build_eval_dataset.py \
  --dataset "$DATASET_PATH" \
  --output "$EVAL_DATASET_PATH" \
  --summary-out "$EVAL_SUMMARY_PATH"

echo "Exporting frontend gold eval rows..."
export_gold_eval_rows

echo "Calibrating ranker serving threshold on gold-train rows..."
"$PYTHON_BIN" ml/calibrate_ranker_threshold.py \
  --dataset "$GOLD_EVAL_ROWS_PATH" \
  --profile jynxzi \
  --artifact-dir "$ARTIFACT_DIR" \
  --split-field gold_holdout_split \
  --split-value train \
  --source gold_eval_train \
  --write-artifact \
  --report-out "$THRESHOLD_CALIBRATION_PATH" \
  > /dev/null

echo "Evaluating trained ranker against heuristic baseline on proxy eval rows..."
"$PYTHON_BIN" ml/evaluate_ranker.py \
  --dataset "$EVAL_DATASET_PATH" \
  --profile jynxzi \
  --artifact-dir "$ARTIFACT_DIR" \
  --report-out "$PROXY_RANKER_EVAL_PATH" \
  > /dev/null

echo "Evaluating trained ranker against heuristic baseline on gold eval rows..."
"$PYTHON_BIN" ml/evaluate_ranker.py \
  --dataset "$GOLD_EVAL_ROWS_PATH" \
  --profile jynxzi \
  --artifact-dir "$ARTIFACT_DIR" \
  --report-out "$GOLD_RANKER_EVAL_PATH" \
  > /dev/null

cp "$GOLD_RANKER_EVAL_PATH" "$RANKER_EVAL_PATH"

echo "Diagnosing ranker hard negatives for the next run..."
"$PYTHON_BIN" ml/diagnose_ranker_holdout.py \
  --dataset "$GOLD_EVAL_ROWS_PATH" \
  --profile jynxzi \
  --artifact-dir "$ARTIFACT_DIR" \
  --report-out "$RANKER_HOLDOUT_DIAGNOSIS_JSON_PATH" \
  --csv-out "$RANKER_HOLDOUT_DIAGNOSIS_CSV_PATH" \
  > /dev/null || true
export_hard_negative_rows

echo "Evaluating heuristic pipeline ranking..."
"$PYTHON_BIN" ml/evaluate_pipeline.py \
  --dataset "$EVAL_DATASET_PATH" \
  --profile jynxzi \
  > "$PIPELINE_EVAL_PATH"

echo "Finalizing MLflow tracking artifacts..."
"$PYTHON_BIN" -m ml.mlflow_tracking finalize-run \
  --artifact-path "$ARTIFACT_JSON_PATH" \
  --artifact-health-path "$ARTIFACT_DIR/jynxzi_artifact_health.json" \
  --proxy-eval-path "$PROXY_RANKER_EVAL_PATH" \
  --gold-eval-path "$GOLD_RANKER_EVAL_PATH" \
  --pipeline-eval-path "$PIPELINE_EVAL_PATH" \
  > /dev/null || true

echo "Evaluating boundary supervision coverage..."
"$PYTHON_BIN" ml/evaluate_boundary_refinement.py \
  --dataset "$DATASET_PATH"

echo "Evaluating rollout recommendation for ENABLE_TRAINED_RANKER_V2..."
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

ranker_eval = Path("ml/artifacts/jynxzi_gold_eval.json")
if not ranker_eval.exists():
    raise SystemExit("Missing ranker eval output: ml/artifacts/jynxzi_gold_eval.json")

report = json.loads(ranker_eval.read_text(encoding="utf-8"))
artifact = report.get("artifact", {})
delta = report.get("metrics", {}).get("delta", {})
trained = report.get("metrics", {}).get("trained_runtime", {})
baseline = report.get("metrics", {}).get("baseline_heuristic", {})
artifact_status = str(artifact.get("status") or "missing")
artifact_reason = str(artifact.get("reason") or "artifact_file_missing")

ndcg_delta = float(delta.get("ndcg_at_5", 0.0))
precision_delta = float(delta.get("precision_at_5", 0.0))
accuracy_delta = float(delta.get("accuracy", 0.0))
recommend_enable = False
reason = "inconclusive_ranker_eval"
if artifact_status == "usable":
    recommend_enable = (
        ndcg_delta >= 0.02
        and precision_delta >= 0.0
        and accuracy_delta >= -0.02
    )
    reason = (
        "trained_runtime_outperforms_or_matches_baseline"
        if recommend_enable
        else "keep_heuristic_default_until_eval_is_stronger"
    )
else:
    reason = f"inconclusive_{artifact_status}:{artifact_reason}"

summary = {
    "recommend_enable_trained_ranker_v2": recommend_enable,
    "reason": reason,
    "artifact": artifact,
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
echo "Gold rows:      $GOLD_EVAL_ROWS_PATH"
echo "Gold eval:      $GOLD_RANKER_EVAL_PATH"
echo "Proxy eval:     $PROXY_RANKER_EVAL_PATH"
echo "Ranker eval:    $RANKER_EVAL_PATH"
echo "Pipeline eval:  $PIPELINE_EVAL_PATH"
echo "Hard negatives: $HARD_NEGATIVE_ROWS_PATH"
