#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$BACKEND_DIR/.venv/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$BACKEND_DIR/tmp/matplotlib}"

VOD_INVENTORY_PATH="$BACKEND_DIR/ml/artifacts/jynxzi_twitch_vod_inventory.jsonl"
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
OUTCOME_TRAIN_DATASET_PATH="$ARTIFACT_DIR/jynxzi_outcome_train_dataset.jsonl"
OUTCOME_TRAIN_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_outcome_train_summary.json"
OUTCOME_HOLDOUT_DATASET_PATH="$ARTIFACT_DIR/jynxzi_outcome_holdout_dataset.jsonl"
OUTCOME_HOLDOUT_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_outcome_holdout_summary.json"
RECENT_VOD_HOLDOUT_POLICY_PATH="$ARTIFACT_DIR/jynxzi_recent_vod_holdout_policy.jsonl"
RECENT_VOD_HOLDOUT_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_recent_vod_holdout_summary.json"
TRAINING_HOLDOUT_POLICY_ROWS_PATH="$ARTIFACT_DIR/jynxzi_training_holdout_policy_rows.jsonl"
TRAINING_HOLDOUT_POLICY_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_training_holdout_policy_summary.json"
OUTCOME_FEATURE_SOURCE_PATH="$ARTIFACT_DIR/jynxzi_outcome_feature_source_rows.jsonl"
OUTCOME_RANKER_EVAL_PATH="$ARTIFACT_DIR/jynxzi_outcome_holdout_eval.json"
ROLLOUT_EVAL_SOURCE_PATH="$ARTIFACT_DIR/jynxzi_rollout_eval_source.json"
PROXY_RANKER_EVAL_PATH="$ARTIFACT_DIR/jynxzi_proxy_eval.json"
PIPELINE_EVAL_PATH="$ARTIFACT_DIR/jynxzi_pipeline_eval.json"
RANKER_HOLDOUT_DIAGNOSIS_JSON_PATH="$ARTIFACT_DIR/jynxzi_ranker_holdout_diagnosis.json"
RANKER_HOLDOUT_DIAGNOSIS_CSV_PATH="$ARTIFACT_DIR/jynxzi_ranker_holdout_diagnosis.csv"
ROLLOUT_READINESS_COVERAGE_PATH="$ARTIFACT_DIR/jynxzi_rollout_readiness_coverage.json"
ROLLOUT_READINESS_COLLECTION_PLAN_PATH="$ARTIFACT_DIR/jynxzi_rollout_readiness_collection_plan.json"
ROLLOUT_READINESS_COLLECTION_QUEUE_PATH="$ARTIFACT_DIR/jynxzi_rollout_readiness_collection_candidates.jsonl"
TARGETED_BACKFILL_FEATURE_ROWS_PATH="$ARTIFACT_DIR/jynxzi_targeted_backfill_feature_rows.jsonl"
TARGETED_BACKFILL_FEATURE_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_targeted_backfill_feature_summary.json"
TARGETED_BACKFILL_VOD_MANIFEST_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_targeted_backfill_vod_manifest_summary.json"
TARGETED_BACKFILL_PRE_REMATCH_GUARD_PATH="$ARTIFACT_DIR/jynxzi_targeted_backfill_pre_rematch_guard.json"
HARD_NEGATIVE_ROWS_PATH="$ARTIFACT_DIR/jynxzi_ranker_hard_negative_train_rows.jsonl"
HARD_NEGATIVE_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_ranker_hard_negative_train_summary.json"
SERVING_HARD_NEGATIVE_ROWS_PATH="$ARTIFACT_DIR/jynxzi_serving_hard_negative_train_rows.jsonl"
SERVING_HARD_NEGATIVE_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_serving_hard_negative_train_summary.json"
AUTO_OUTCOME_NEGATIVE_ROWS_PATH="$ARTIFACT_DIR/jynxzi_auto_outcome_negative_rows.jsonl"
AUTO_OUTCOME_NEGATIVE_TRAIN_ROWS_PATH="$ARTIFACT_DIR/jynxzi_auto_outcome_negative_train_rows.jsonl"
AUTO_OUTCOME_NEGATIVE_HOLDOUT_ROWS_PATH="$ARTIFACT_DIR/jynxzi_auto_outcome_negative_holdout_rows.jsonl"
AUTO_OUTCOME_NEGATIVE_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_auto_outcome_negative_summary.json"
AUTO_OUTCOME_NEGATIVE_EVAL_PATH="$ARTIFACT_DIR/jynxzi_auto_outcome_negative_eval.json"
CROSS_PLATFORM_FINGERPRINT_ROWS_PATH="$ARTIFACT_DIR/jynxzi_cross_platform_fingerprint_rows.jsonl"
MM_JUDGE_PROXY_ROWS_PATH="$ARTIFACT_DIR/jynxzi_mm_judge_proxy.jsonl"
MM_JUDGE_SNAPSHOTS_PATH="${MM_JUDGE_SNAPSHOTS_PATH:-$BACKEND_DIR/tmp/storage/feedback/mm_judge_snapshots.jsonl}"
CROSS_PLATFORM_VOD_MANIFEST_PATH="${CROSS_PLATFORM_VOD_MANIFEST_PATH:-$ARTIFACT_DIR/jynxzi_cross_platform_vods.jsonl}"
CROSS_PLATFORM_REPOST_MANIFEST_PATH="${CROSS_PLATFORM_REPOST_MANIFEST_PATH:-$ARTIFACT_DIR/jynxzi_cross_platform_reposts.jsonl}"
CROSS_PLATFORM_FINGERPRINT_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_cross_platform_fingerprint_summary.json"
CROSS_PLATFORM_FINGERPRINT_DIAGNOSTICS_PATH="$ARTIFACT_DIR/jynxzi_cross_platform_fingerprint_diagnostics.jsonl"
CROSS_PLATFORM_FINGERPRINT_ENABLED="${CROSS_PLATFORM_FINGERPRINT_ENABLED:-1}"
CROSS_PLATFORM_FINGERPRINT_MIN_CONFIDENCE="${CROSS_PLATFORM_FINGERPRINT_MIN_CONFIDENCE:-0.50}"
CROSS_PLATFORM_VERIFICATION_MODE="${CROSS_PLATFORM_VERIFICATION_MODE:-transcript}"
CROSS_PLATFORM_TRANSCRIPT_MATCH_WORKERS="${CROSS_PLATFORM_TRANSCRIPT_MATCH_WORKERS:-20}"
CROSS_PLATFORM_SOURCE_MANIFEST_PATH="${CROSS_PLATFORM_SOURCE_MANIFEST_PATH:-$BACKEND_DIR/ml/cross_platform_sources.csv}"
CROSS_PLATFORM_URL_MANIFEST_PATH="${CROSS_PLATFORM_URL_MANIFEST_PATH:-$BACKEND_DIR/ml/viral_clips.txt}"
CROSS_PLATFORM_MAX_REPOSTS="${CROSS_PLATFORM_MAX_REPOSTS:-3600}"
CROSS_PLATFORM_MIN_REPOSTS="${CROSS_PLATFORM_MIN_REPOSTS:-0}"
CROSS_PLATFORM_MIN_MATCHED_ROWS="${CROSS_PLATFORM_MIN_MATCHED_ROWS:-70}"
CROSS_PLATFORM_SEED_MIN_MATCHED_ROWS="${CROSS_PLATFORM_SEED_MIN_MATCHED_ROWS:-70}"
CROSS_PLATFORM_MIN_MATCHED_VODS="${CROSS_PLATFORM_MIN_MATCHED_VODS:-20}"
CROSS_PLATFORM_REFRESH_VODS="${CROSS_PLATFORM_REFRESH_VODS:-0}"
CROSS_PLATFORM_SECOND_PASS_AFTER_VOD_BUILD="${CROSS_PLATFORM_SECOND_PASS_AFTER_VOD_BUILD:-1}"
CROSS_PLATFORM_REFRESH_REPOSTS="${CROSS_PLATFORM_REFRESH_REPOSTS:-0}"
CROSS_PLATFORM_REUSE_EXISTING_ROWS="${CROSS_PLATFORM_REUSE_EXISTING_ROWS:-0}"
CROSS_PLATFORM_HOLDOUT_RATIO="${CROSS_PLATFORM_HOLDOUT_RATIO:-0.20}"
CROSS_PLATFORM_DISCOVER_REPOSTS="${CROSS_PLATFORM_DISCOVER_REPOSTS:-1}"
CROSS_PLATFORM_DOWNLOAD_REPOSTS="${CROSS_PLATFORM_DOWNLOAD_REPOSTS:-1}"
CROSS_PLATFORM_FULL_VOD_AUDIO_FINGERPRINTS="${CROSS_PLATFORM_FULL_VOD_AUDIO_FINGERPRINTS:-1}"
CROSS_PLATFORM_FULL_VOD_AUDIO_MAX_VODS="${CROSS_PLATFORM_FULL_VOD_AUDIO_MAX_VODS:-0}"
CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS="${CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS:-2}"
CROSS_PLATFORM_KEEP_FULL_VOD_AUDIO="${CROSS_PLATFORM_KEEP_FULL_VOD_AUDIO:-0}"
CROSS_PLATFORM_COARSE_VOD_TILING="${CROSS_PLATFORM_COARSE_VOD_TILING:-0}"
CROSS_PLATFORM_COARSE_VOD_TILE_SEC="${CROSS_PLATFORM_COARSE_VOD_TILE_SEC:-360}"
CROSS_PLATFORM_COARSE_VOD_STRIDE_SEC="${CROSS_PLATFORM_COARSE_VOD_STRIDE_SEC:-180}"
CROSS_PLATFORM_COARSE_VOD_MAX_TILES_PER_VOD="${CROSS_PLATFORM_COARSE_VOD_MAX_TILES_PER_VOD:-0}"
CROSS_PLATFORM_COARSE_VOD_MAX_VODS="${CROSS_PLATFORM_COARSE_VOD_MAX_VODS:-0}"
CROSS_PLATFORM_COARSE_VOD_HORIZON_PADDING_SEC="${CROSS_PLATFORM_COARSE_VOD_HORIZON_PADDING_SEC:-900}"
ROLLOUT_REVIEW_QUEUE_BASE="$ARTIFACT_DIR/jynxzi_rollout_review_queue"
ROLLOUT_REVIEW_QUEUE_CSV_PATH="$ARTIFACT_DIR/jynxzi_rollout_review_queue.csv"
ROLLOUT_REVIEW_QUEUE_FEATURES_PATH="$ARTIFACT_DIR/jynxzi_rollout_review_queue.features.jsonl"
ROLLOUT_REVIEW_QUEUE_SUMMARY_PATH="$ARTIFACT_DIR/jynxzi_rollout_review_queue.summary.json"
WORKFLOW_STATE_PATH="$ARTIFACT_DIR/jynxzi_training_workflow_state.json"
WORK_DIR_BASE="${AI_CLIPPING_TRAIN_TMPDIR:-$BACKEND_DIR/tmp/training_runs}"
WORK_DIR="$WORK_DIR_BASE/jynxzi_resume"
STAGED_DATASET_PATH="$WORK_DIR/jynxzi_vod_training_rows.staged.jsonl"
EMPTY_TWITCH_LABELS_PATH="$WORK_DIR/empty_twitch_clip_labels.jsonl"
VOD_RANKER_SHORT_FORM_MODE="${VOD_RANKER_SHORT_FORM_MODE:-off}"
VOD_RANKER_GOLD_TRAIN_WEIGHT="${VOD_RANKER_GOLD_TRAIN_WEIGHT:-2.0}"
VOD_RANKER_USE_GOLD_TRAIN="${VOD_RANKER_USE_GOLD_TRAIN:-0}"
VOD_RANKER_HARD_NEGATIVE_WEIGHT="${VOD_RANKER_HARD_NEGATIVE_WEIGHT:-4.0}"
VOD_RANKER_SERVING_HARD_NEGATIVE_WEIGHT="${VOD_RANKER_SERVING_HARD_NEGATIVE_WEIGHT:-2.0}"
VOD_RANKER_USE_GOLD_HARD_NEGATIVES="${VOD_RANKER_USE_GOLD_HARD_NEGATIVES:-0}"
VOD_RANKER_USE_USER_LABELS="${VOD_RANKER_USE_USER_LABELS:-0}"
VOD_RANKER_ROLLOUT_REVIEW_NEGATIVE_WEIGHT="${VOD_RANKER_ROLLOUT_REVIEW_NEGATIVE_WEIGHT:-4.0}"
VOD_RANKER_OUTCOME_POSITIVE_WEIGHT="${VOD_RANKER_OUTCOME_POSITIVE_WEIGHT:-0.55}"
VOD_RANKER_PRIMARY_LABEL_POLICY="${VOD_RANKER_PRIMARY_LABEL_POLICY:-outcome_only}"
VOD_RANKER_REQUIRE_CROSS_PLATFORM_OUTCOME="${VOD_RANKER_REQUIRE_CROSS_PLATFORM_OUTCOME:-1}"
VOD_RANKER_MIN_CROSS_PLATFORM_TRAIN_ROWS="${VOD_RANKER_MIN_CROSS_PLATFORM_TRAIN_ROWS:-70}"
VOD_RANKER_USE_MM_JUDGE_PROXY="${VOD_RANKER_USE_MM_JUDGE_PROXY:-1}"
VOD_RANKER_MM_JUDGE_WEIGHT="${VOD_RANKER_MM_JUDGE_WEIGHT:-0.4}"
VOD_RANKER_AUTO_OUTCOME_HARD_NEGATIVE_WEIGHT="${VOD_RANKER_AUTO_OUTCOME_HARD_NEGATIVE_WEIGHT:-1.0}"
VOD_RANKER_AUTO_OUTCOME_BACKGROUND_NEGATIVE_WEIGHT="${VOD_RANKER_AUTO_OUTCOME_BACKGROUND_NEGATIVE_WEIGHT:-0.25}"
VOD_RANKER_ROLLOUT_THRESHOLD_MIN_RECALL="${VOD_RANKER_ROLLOUT_THRESHOLD_MIN_RECALL:-0.85}"
VOD_RANKER_ROLLOUT_THRESHOLD_MAX_SELECTED_RATE="${VOD_RANKER_ROLLOUT_THRESHOLD_MAX_SELECTED_RATE:-0.75}"
VOD_RANKER_ROLLOUT_THRESHOLD_TARGET_PRECISION="${VOD_RANKER_ROLLOUT_THRESHOLD_TARGET_PRECISION:-0.85}"
AUTO_OUTCOME_NEGATIVE_ENABLED="${AUTO_OUTCOME_NEGATIVE_ENABLED:-1}"
AUTO_OUTCOME_NEGATIVE_MIN_AGE_DAYS="${AUTO_OUTCOME_NEGATIVE_MIN_AGE_DAYS:-0}"
AUTO_OUTCOME_NEGATIVE_HOLDOUT_RATIO="${AUTO_OUTCOME_NEGATIVE_HOLDOUT_RATIO:-0.20}"
AUTO_OUTCOME_NEGATIVE_POSITIVE_EXCLUSION_SEC="${AUTO_OUTCOME_NEGATIVE_POSITIVE_EXCLUSION_SEC:-180}"
AUTO_OUTCOME_NEGATIVE_HARD_SCORE_THRESHOLD="${AUTO_OUTCOME_NEGATIVE_HARD_SCORE_THRESHOLD:-7.5}"
AUTO_OUTCOME_NEGATIVE_MAX_ROWS="${AUTO_OUTCOME_NEGATIVE_MAX_ROWS:-2000}"
SERVING_HARD_NEGATIVE_ENABLED="${SERVING_HARD_NEGATIVE_ENABLED:-1}"
SERVING_HARD_NEGATIVE_PRE_JUDGE_TOP_K="${SERVING_HARD_NEGATIVE_PRE_JUDGE_TOP_K:-500}"
SERVING_HARD_NEGATIVE_MAX_PER_VOD="${SERVING_HARD_NEGATIVE_MAX_PER_VOD:-20}"
SERVING_HARD_NEGATIVE_MAX_ROWS="${SERVING_HARD_NEGATIVE_MAX_ROWS:-0}"
SERVING_HARD_NEGATIVE_POSITIVE_EXCLUSION_OVERLAP="${SERVING_HARD_NEGATIVE_POSITIVE_EXCLUSION_OVERLAP:-0.25}"
SERVING_HARD_NEGATIVE_MIN_RANK_SCORE="${SERVING_HARD_NEGATIVE_MIN_RANK_SCORE:--999.0}"
ROLLOUT_REVIEW_THRESHOLD="${ROLLOUT_REVIEW_THRESHOLD:-3.0}"
ROLLOUT_REVIEW_THRESHOLD_MARGIN="${ROLLOUT_REVIEW_THRESHOLD_MARGIN:-1.5}"
ROLLOUT_DISAGREEMENT_HEURISTIC_THRESHOLD="${ROLLOUT_DISAGREEMENT_HEURISTIC_THRESHOLD:-8.0}"
ROLLOUT_DISAGREEMENT_MIN_GAP="${ROLLOUT_DISAGREEMENT_MIN_GAP:-4.0}"
ROLLOUT_MAX_DISAGREEMENT_PER_VOD="${ROLLOUT_MAX_DISAGREEMENT_PER_VOD:-2}"
ROLLOUT_REVIEW_TOP_K_PER_VOD="${ROLLOUT_REVIEW_TOP_K_PER_VOD:-10}"
ROLLOUT_REVIEW_MAX_TOTAL="${ROLLOUT_REVIEW_MAX_TOTAL:-200}"
ROLLOUT_REVIEW_MIN_GAP_SEC="${ROLLOUT_REVIEW_MIN_GAP_SEC:-20}"
SHORT_FORM_BACKFILL_ARTIFACT_DIR="$BACKEND_DIR/tmp/short_form_backfill_artifacts"
SHORT_FORM_BACKFILL_ARTIFACT_JSON_PATH="$SHORT_FORM_BACKFILL_ARTIFACT_DIR/jynxzi_ranker.json"
SHORT_FORM_BACKFILL_ARTIFACT_HEALTH_PATH="$SHORT_FORM_BACKFILL_ARTIFACT_DIR/jynxzi_artifact_health.json"
PERSISTED_SHORT_FORM_BACKFILL_ARTIFACT_JSON_PATH="$ARTIFACT_DIR/jynxzi_short_form_backfill_ranker.json"
PERSISTED_SHORT_FORM_BACKFILL_ARTIFACT_HEALTH_PATH="$ARTIFACT_DIR/jynxzi_short_form_backfill_artifact_health.json"
OUTCOME_HOLDOUT_TARGET_POSITIVE_ROWS="${OUTCOME_HOLDOUT_TARGET_POSITIVE_ROWS:-300}"
RECENT_VOD_HOLDOUT_ENABLED="${RECENT_VOD_HOLDOUT_ENABLED:-1}"
RECENT_VOD_HOLDOUT_COUNT="${RECENT_VOD_HOLDOUT_COUNT:-10}"
RECENT_VOD_HOLDOUT_REFRESH="${RECENT_VOD_HOLDOUT_REFRESH:-0}"
ROLLOUT_READINESS_TARGET_NEW_VOD_GROUPS="${ROLLOUT_READINESS_TARGET_NEW_VOD_GROUPS:-20}"
ROLLOUT_READINESS_TARGET_AVAILABLE_GROUPS="${ROLLOUT_READINESS_TARGET_AVAILABLE_GROUPS:-40}"
ROLLOUT_READINESS_MAX_CANDIDATES_PER_VOD="${ROLLOUT_READINESS_MAX_CANDIDATES_PER_VOD:-3}"
TARGETED_BACKFILL_REQUIRE_ALL_VODS="${TARGETED_BACKFILL_REQUIRE_ALL_VODS:-1}"
JYNXZI_TRAIN_DEFAULT_LOOKBACK_DAYS="${JYNXZI_TRAIN_DEFAULT_LOOKBACK_DAYS:-180}"

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
  DATE_FROM="$("$PYTHON_BIN" - <<'PY'
from datetime import date, timedelta
import os

try:
    days = max(int(os.environ.get("JYNXZI_TRAIN_DEFAULT_LOOKBACK_DAYS") or "180"), 1)
except ValueError:
    days = 180
print((date.today() - timedelta(days=days)).isoformat())
PY
)"
fi

if [[ -z "$DATE_TO" ]]; then
  DATE_TO="$("$PYTHON_BIN" -c 'from datetime import date; print(date.today().isoformat())')"
fi

export SKIP_COVERED MERGE_OUTPUT
export ARTIFACT_ONLY
export DATE_FROM DATE_TO
export VOD_RANKER_SHORT_FORM_MODE
export VOD_RANKER_GOLD_TRAIN_WEIGHT
export VOD_RANKER_USE_GOLD_TRAIN
export VOD_RANKER_HARD_NEGATIVE_WEIGHT
export VOD_RANKER_SERVING_HARD_NEGATIVE_WEIGHT
export VOD_RANKER_USE_GOLD_HARD_NEGATIVES
export VOD_RANKER_USE_USER_LABELS
export VOD_RANKER_ROLLOUT_REVIEW_NEGATIVE_WEIGHT
export VOD_RANKER_OUTCOME_POSITIVE_WEIGHT
export VOD_RANKER_PRIMARY_LABEL_POLICY
export VOD_RANKER_REQUIRE_CROSS_PLATFORM_OUTCOME
export VOD_RANKER_MIN_CROSS_PLATFORM_TRAIN_ROWS
export VOD_RANKER_USE_MM_JUDGE_PROXY
export VOD_RANKER_MM_JUDGE_WEIGHT
export VOD_RANKER_AUTO_OUTCOME_HARD_NEGATIVE_WEIGHT
export VOD_RANKER_AUTO_OUTCOME_BACKGROUND_NEGATIVE_WEIGHT
export AUTO_OUTCOME_NEGATIVE_ENABLED
export AUTO_OUTCOME_NEGATIVE_MIN_AGE_DAYS
export AUTO_OUTCOME_NEGATIVE_HOLDOUT_RATIO
export AUTO_OUTCOME_NEGATIVE_POSITIVE_EXCLUSION_SEC
export AUTO_OUTCOME_NEGATIVE_HARD_SCORE_THRESHOLD
export AUTO_OUTCOME_NEGATIVE_MAX_ROWS
export SERVING_HARD_NEGATIVE_ENABLED
export SERVING_HARD_NEGATIVE_ROWS_PATH
export SERVING_HARD_NEGATIVE_SUMMARY_PATH
export SERVING_HARD_NEGATIVE_PRE_JUDGE_TOP_K
export SERVING_HARD_NEGATIVE_MAX_PER_VOD
export SERVING_HARD_NEGATIVE_MAX_ROWS
export SERVING_HARD_NEGATIVE_POSITIVE_EXCLUSION_OVERLAP
export SERVING_HARD_NEGATIVE_MIN_RANK_SCORE
export OUTCOME_HOLDOUT_TARGET_POSITIVE_ROWS
export RECENT_VOD_HOLDOUT_ENABLED
export RECENT_VOD_HOLDOUT_COUNT
export RECENT_VOD_HOLDOUT_REFRESH
export RECENT_VOD_HOLDOUT_POLICY_PATH
export RECENT_VOD_HOLDOUT_SUMMARY_PATH
export TRAINING_HOLDOUT_POLICY_ROWS_PATH
export TRAINING_HOLDOUT_POLICY_SUMMARY_PATH
export OUTCOME_FEATURE_SOURCE_PATH
export VOD_INVENTORY_PATH
export EMPTY_TWITCH_LABELS_PATH
export ROLLOUT_READINESS_COLLECTION_QUEUE_PATH
export ROLLOUT_READINESS_COLLECTION_PLAN_PATH
export TARGETED_BACKFILL_FEATURE_ROWS_PATH
export TARGETED_BACKFILL_REQUIRE_ALL_VODS
export CROSS_PLATFORM_FINGERPRINT_ENABLED
export CROSS_PLATFORM_FINGERPRINT_MIN_CONFIDENCE
export CROSS_PLATFORM_VERIFICATION_MODE
export CROSS_PLATFORM_TRANSCRIPT_MATCH_WORKERS
export CROSS_PLATFORM_VOD_MANIFEST_PATH
export CROSS_PLATFORM_REPOST_MANIFEST_PATH
export CROSS_PLATFORM_FINGERPRINT_ROWS_PATH
export MM_JUDGE_PROXY_ROWS_PATH
export MM_JUDGE_SNAPSHOTS_PATH
export CROSS_PLATFORM_FINGERPRINT_SUMMARY_PATH
export CROSS_PLATFORM_FINGERPRINT_DIAGNOSTICS_PATH
export CROSS_PLATFORM_SOURCE_MANIFEST_PATH
export CROSS_PLATFORM_URL_MANIFEST_PATH
export CROSS_PLATFORM_MAX_REPOSTS
export CROSS_PLATFORM_MIN_REPOSTS
export CROSS_PLATFORM_MIN_MATCHED_ROWS
export CROSS_PLATFORM_SEED_MIN_MATCHED_ROWS
export CROSS_PLATFORM_MIN_MATCHED_VODS
export CROSS_PLATFORM_REFRESH_VODS
export CROSS_PLATFORM_SECOND_PASS_AFTER_VOD_BUILD
export CROSS_PLATFORM_REFRESH_REPOSTS
export CROSS_PLATFORM_REUSE_EXISTING_ROWS
export CROSS_PLATFORM_HOLDOUT_RATIO
export CROSS_PLATFORM_DISCOVER_REPOSTS
export CROSS_PLATFORM_DOWNLOAD_REPOSTS
export CROSS_PLATFORM_FULL_VOD_AUDIO_FINGERPRINTS
export CROSS_PLATFORM_FULL_VOD_AUDIO_MAX_VODS
export CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS
export CROSS_PLATFORM_KEEP_FULL_VOD_AUDIO
export CROSS_PLATFORM_COARSE_VOD_TILING
export CROSS_PLATFORM_COARSE_VOD_TILE_SEC
export CROSS_PLATFORM_COARSE_VOD_STRIDE_SEC
export CROSS_PLATFORM_COARSE_VOD_MAX_TILES_PER_VOD
export CROSS_PLATFORM_COARSE_VOD_MAX_VODS
export CROSS_PLATFORM_COARSE_VOD_HORIZON_PADDING_SEC
export JYNXZI_TRAIN_DEFAULT_LOOKBACK_DAYS
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
  local dataset_out="${1:-$DATASET_PATH}"
  mkdir -p "$(dirname "$EMPTY_TWITCH_LABELS_PATH")"
  : > "$EMPTY_TWITCH_LABELS_PATH"
  TRAIN_ARGS=(
    --labels "$EMPTY_TWITCH_LABELS_PATH" \
    --profile jynxzi \
    --dataset-out "$dataset_out" \
    --artifact-dir "$ARTIFACT_DIR" \
    --work-dir "$WORK_DIR" \
    --include-short-form "$SHORT_FORM_PATH" \
    --short-form-mode "$VOD_RANKER_SHORT_FORM_MODE" \
    --holdout-policy-rows "$TRAINING_HOLDOUT_POLICY_ROWS_PATH" \
    --include-outcome-positive-train "$OUTCOME_TRAIN_DATASET_PATH" \
    --outcome-positive-weight "$VOD_RANKER_OUTCOME_POSITIVE_WEIGHT" \
    --include-auto-outcome-negative-train "$AUTO_OUTCOME_NEGATIVE_TRAIN_ROWS_PATH" \
    --auto-outcome-hard-negative-weight "$VOD_RANKER_AUTO_OUTCOME_HARD_NEGATIVE_WEIGHT" \
    --auto-outcome-background-negative-weight "$VOD_RANKER_AUTO_OUTCOME_BACKGROUND_NEGATIVE_WEIGHT" \
    --include-cross-platform-outcome-train "$CROSS_PLATFORM_FINGERPRINT_ROWS_PATH" \
    --primary-label-policy "$VOD_RANKER_PRIMARY_LABEL_POLICY" \
    --min-cross-platform-train-rows "$VOD_RANKER_MIN_CROSS_PLATFORM_TRAIN_ROWS" \
    --workflow-state-path "$WORKFLOW_STATE_PATH" \
    --workflow-state-json "$CURRENT_WORKFLOW_STATE_JSON"
  )
  if [[ "$VOD_RANKER_USE_MM_JUDGE_PROXY" == "1" ]]; then
    TRAIN_ARGS+=(
      --include-mm-judge-train "$MM_JUDGE_PROXY_ROWS_PATH" \
      --mm-judge-weight "$VOD_RANKER_MM_JUDGE_WEIGHT"
    )
  fi
  if [[ -s "$ROLLOUT_READINESS_COLLECTION_QUEUE_PATH" ]]; then
    TRAIN_ARGS+=(
      --targeted-backfill-labels "$ROLLOUT_READINESS_COLLECTION_QUEUE_PATH"
    )
  fi
  if [[ "$VOD_RANKER_REQUIRE_CROSS_PLATFORM_OUTCOME" == "1" ]]; then
    TRAIN_ARGS+=(--require-cross-platform-outcome)
  fi
  if [[ "$VOD_RANKER_USE_USER_LABELS" == "1" && "$VOD_RANKER_USE_GOLD_TRAIN" == "1" ]]; then
    TRAIN_ARGS+=(
      --include-gold-train "$GOLD_EVAL_ROWS_PATH"
      --gold-train-weight "$VOD_RANKER_GOLD_TRAIN_WEIGHT"
      --rollout-review-negative-weight "$VOD_RANKER_ROLLOUT_REVIEW_NEGATIVE_WEIGHT"
    )
  fi
  if [[ "$VOD_RANKER_USE_USER_LABELS" == "1" && "$VOD_RANKER_USE_GOLD_HARD_NEGATIVES" == "1" ]]; then
    TRAIN_ARGS+=(
      --include-hard-negative-train "$HARD_NEGATIVE_ROWS_PATH"
      --hard-negative-weight "$VOD_RANKER_HARD_NEGATIVE_WEIGHT"
    )
  fi
  if [[ "$SERVING_HARD_NEGATIVE_ENABLED" == "1" && -s "$SERVING_HARD_NEGATIVE_ROWS_PATH" ]]; then
    TRAIN_ARGS+=(
      --include-serving-hard-negative-train "$SERVING_HARD_NEGATIVE_ROWS_PATH"
      --serving-hard-negative-weight "$VOD_RANKER_SERVING_HARD_NEGATIVE_WEIGHT"
    )
  fi
  "$PYTHON_BIN" ml/train_vod_ranker.py "${TRAIN_ARGS[@]}"
}

promote_staged_vod_dataset() {
  if [[ ! -s "$STAGED_DATASET_PATH" ]]; then
    echo "Staged VOD dataset missing or empty: $STAGED_DATASET_PATH" >&2
    return 1
  fi
  mkdir -p "$(dirname "$DATASET_PATH")"
  mv "$STAGED_DATASET_PATH" "$DATASET_PATH"
}

log_cross_platform_match_delta() {
  local phase="$1"
  local before_path="$2"
  "$PYTHON_BIN" -m ml.cross_platform_match_delta \
    --phase "$phase" \
    --before "$before_path" \
    --after "$CROSS_PLATFORM_FINGERPRINT_ROWS_PATH" \
    --summary "$CROSS_PLATFORM_FINGERPRINT_SUMMARY_PATH"
}

prepare_cross_platform_fingerprint_rows() {
  if [[ "$CROSS_PLATFORM_FINGERPRINT_ENABLED" != "1" ]]; then
    echo "Cross-platform fingerprint preparation disabled."
    return 0
  fi
  local min_matched_rows="${1:-$CROSS_PLATFORM_MIN_MATCHED_ROWS}"
  local min_matched_vods="${2:-$CROSS_PLATFORM_MIN_MATCHED_VODS}"
  local refresh_vods="${3:-$CROSS_PLATFORM_REFRESH_VODS}"
  local reuse_existing="${4:-$CROSS_PLATFORM_REUSE_EXISTING_ROWS}"
  local phase="initial"
  if [[ "$refresh_vods" == "1" ]]; then
    phase="post_vod_backfill"
  elif [[ "$reuse_existing" == "1" ]]; then
    phase="reuse_existing"
  fi
  local before_path="$WORK_DIR/cross_platform_fingerprint_rows.before.${phase}.$$.jsonl"
  mkdir -p "$WORK_DIR"
  if [[ -f "$CROSS_PLATFORM_FINGERPRINT_ROWS_PATH" ]]; then
    cp "$CROSS_PLATFORM_FINGERPRINT_ROWS_PATH" "$before_path"
  else
    : > "$before_path"
  fi
  "$PYTHON_BIN" -m ml.prepare_cross_platform_fingerprint_rows \
    --vods "$CROSS_PLATFORM_VOD_MANIFEST_PATH" \
    --reposts "$CROSS_PLATFORM_REPOST_MANIFEST_PATH" \
    --output "$CROSS_PLATFORM_FINGERPRINT_ROWS_PATH" \
    --summary "$CROSS_PLATFORM_FINGERPRINT_SUMMARY_PATH" \
    --diagnostic-output "$CROSS_PLATFORM_FINGERPRINT_DIAGNOSTICS_PATH" \
    --min-confidence "$CROSS_PLATFORM_FINGERPRINT_MIN_CONFIDENCE" \
    --verification-mode "$CROSS_PLATFORM_VERIFICATION_MODE" \
    --profile jynxzi \
    --vod-cache-root "$BACKEND_DIR/tmp/vod_feature_cache" \
    --repost-source-manifest "$CROSS_PLATFORM_SOURCE_MANIFEST_PATH" \
    --repost-url-manifest "$CROSS_PLATFORM_URL_MANIFEST_PATH" \
    --max-reposts "$CROSS_PLATFORM_MAX_REPOSTS" \
    --min-reposts "$CROSS_PLATFORM_MIN_REPOSTS" \
    --min-matched-rows "$min_matched_rows" \
    --transcript-match-workers "$CROSS_PLATFORM_TRANSCRIPT_MATCH_WORKERS" \
    --min-matched-vods "$min_matched_vods" \
    "$([[ "$refresh_vods" == "1" ]] && echo --refresh-vods || echo --no-refresh-vods)" \
    "$([[ "$CROSS_PLATFORM_REFRESH_REPOSTS" == "1" ]] && echo --refresh-reposts || echo --no-refresh-reposts)" \
    "$([[ "$reuse_existing" == "1" ]] && echo --reuse-existing-output || echo --no-reuse-existing-output)" \
    "$([[ "$CROSS_PLATFORM_FULL_VOD_AUDIO_FINGERPRINTS" == "1" ]] && echo --full-vod-audio-fingerprints || echo --no-full-vod-audio-fingerprints)" \
    --full-vod-audio-labels "$EMPTY_TWITCH_LABELS_PATH" \
    --full-vod-audio-vod-inventory "$VOD_INVENTORY_PATH" \
    --full-vod-audio-selected-vods "$ROLLOUT_READINESS_COLLECTION_QUEUE_PATH" \
    --full-vod-audio-max-vods "$CROSS_PLATFORM_FULL_VOD_AUDIO_MAX_VODS" \
    --full-vod-audio-workers "$CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS" \
    "$([[ "$CROSS_PLATFORM_KEEP_FULL_VOD_AUDIO" == "1" ]] && echo --keep-full-vod-audio || echo --no-keep-full-vod-audio)" \
    "$([[ "$refresh_vods" == "1" && "$CROSS_PLATFORM_COARSE_VOD_TILING" == "1" ]] && echo --coarse-vod-tiling || echo --no-coarse-vod-tiling)" \
    --coarse-vod-labels "$EMPTY_TWITCH_LABELS_PATH" \
    --coarse-vod-selected-vods "$ROLLOUT_READINESS_COLLECTION_QUEUE_PATH" \
    --coarse-vod-tile-sec "$CROSS_PLATFORM_COARSE_VOD_TILE_SEC" \
    --coarse-vod-stride-sec "$CROSS_PLATFORM_COARSE_VOD_STRIDE_SEC" \
    --coarse-vod-max-tiles-per-vod "$CROSS_PLATFORM_COARSE_VOD_MAX_TILES_PER_VOD" \
    --coarse-vod-max-vods "$CROSS_PLATFORM_COARSE_VOD_MAX_VODS" \
    --coarse-vod-horizon-padding-sec "$CROSS_PLATFORM_COARSE_VOD_HORIZON_PADDING_SEC" \
    --fail-on-blocked \
    --fail-on-insufficient \
    "$([[ "$CROSS_PLATFORM_DISCOVER_REPOSTS" == "1" ]] && echo --discover-reposts || echo --no-discover-reposts)" \
    "$([[ "$CROSS_PLATFORM_DOWNLOAD_REPOSTS" == "1" ]] && echo --download-reposts || echo --no-download-reposts)"
  log_cross_platform_match_delta "$phase" "$before_path"
  rm -f "$before_path"
}

import_mm_judge_proxy_rows() {
  if [[ "$VOD_RANKER_USE_MM_JUDGE_PROXY" != "1" ]]; then
    echo "MM-judge proxy import disabled."
    return 0
  fi
  "$PYTHON_BIN" ml/import_mm_judge_labels.py \
    --snapshots "$MM_JUDGE_SNAPSHOTS_PATH" \
    --output "$MM_JUDGE_PROXY_ROWS_PATH"
}

export_gold_eval_rows() {
  if [[ "$VOD_RANKER_USE_USER_LABELS" != "1" ]]; then
    reset_user_label_artifacts
    echo "User-labeled gold eval export disabled; removed gold-eval artifacts."
    return 0
  fi
  "$PYTHON_BIN" ml/export_frontend_gold_eval.py \
    --storage-dir "tmp/storage" \
    --creator-id jynxzi \
    --rows-out "$GOLD_EVAL_ROWS_PATH" \
    --output "$GOLD_EVAL_DATASET_PATH" \
    --summary-out "$GOLD_EVAL_SUMMARY_PATH"
}

reset_user_label_artifacts() {
  "$PYTHON_BIN" - \
    "$GOLD_EVAL_ROWS_PATH" \
    "$GOLD_EVAL_DATASET_PATH" \
    "$GOLD_EVAL_SUMMARY_PATH" \
    "$GOLD_RANKER_EVAL_PATH" \
    "$HARD_NEGATIVE_ROWS_PATH" \
    "$HARD_NEGATIVE_SUMMARY_PATH" \
    "$RANKER_HOLDOUT_DIAGNOSIS_JSON_PATH" \
    "$RANKER_HOLDOUT_DIAGNOSIS_CSV_PATH" \
    "$ROLLOUT_REVIEW_QUEUE_CSV_PATH" \
    "$ROLLOUT_REVIEW_QUEUE_FEATURES_PATH" \
    "$ROLLOUT_REVIEW_QUEUE_SUMMARY_PATH" <<'PY'
import csv
import json
import sys
from pathlib import Path

(
    gold_rows_path,
    gold_dataset_path,
    gold_summary_path,
    gold_eval_report_path,
    hard_negative_rows_path,
    hard_negative_summary_path,
    holdout_diagnosis_json_path,
    holdout_diagnosis_csv_path,
    rollout_queue_csv_path,
    rollout_queue_features_path,
    rollout_queue_summary_path,
) = [Path(value) for value in sys.argv[1:]]

for path in (
    gold_rows_path,
    gold_dataset_path,
    gold_summary_path,
    gold_eval_report_path,
    hard_negative_rows_path,
    hard_negative_summary_path,
    holdout_diagnosis_json_path,
    holdout_diagnosis_csv_path,
    rollout_queue_csv_path,
    rollout_queue_features_path,
    rollout_queue_summary_path,
):
    path.parent.mkdir(parents=True, exist_ok=True)

for path in (gold_rows_path, gold_dataset_path, gold_summary_path, gold_eval_report_path):
    path.unlink(missing_ok=True)

for path in (hard_negative_rows_path, rollout_queue_features_path):
    path.write_text("", encoding="utf-8")

hard_negative_summary_path.write_text(
    json.dumps(
        {
            "profile": "jynxzi",
            "exported_rows": 0,
            "carried_forward_rows": 0,
            "selected_false_positive_rows": 0,
            "dataset_sources": {},
            "status": "skipped",
            "reason": "user_labeled_rows_disabled",
        },
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
holdout_diagnosis_json_path.write_text(
    json.dumps(
        {
            "profile": "jynxzi",
            "rows": 0,
            "metrics": {},
            "status": "skipped",
            "reason": "user_labeled_rows_disabled",
        },
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
with holdout_diagnosis_csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["status", "reason"])
    writer.writerow(["skipped", "user_labeled_rows_disabled"])
with rollout_queue_csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["status", "reason"])
    writer.writerow(["skipped", "user_labeled_rows_disabled"])
rollout_queue_summary_path.write_text(
    json.dumps(
        {
            "rows": 0,
            "review_required": 0,
            "status": "skipped",
            "reason": "user_labeled_rows_disabled",
        },
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "event": "user_labeled_artifacts_reset",
            "gold_rows": str(gold_rows_path),
            "hard_negative_rows": str(hard_negative_rows_path),
            "rollout_review_queue": str(rollout_queue_csv_path),
        },
        ensure_ascii=True,
    )
)
PY
}

build_twitch_vod_inventory() {
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from config import settings
from ml.collect_twitch_clip_labels import TwitchClient, _get_user, _get_vods
from ml.prepare_cross_platform_fingerprint_rows import _read_json, _safe_float

output_path = Path(os.environ["VOD_INVENTORY_PATH"])
vod_cache_root = Path(os.environ.get("CROSS_PLATFORM_VOD_CACHE_ROOT") or "tmp/vod_feature_cache")
date_from = os.environ["DATE_FROM"]
date_to = os.environ["DATE_TO"]
profile = "jynxzi"

rows_by_vod: dict[str, dict] = {}

def add_row(row: dict) -> None:
    vod_id = str(row.get("vod_id") or "").strip()
    if not vod_id:
        return
    existing = rows_by_vod.get(vod_id, {})
    row_source = str(row.get("inventory_source") or "unknown")
    existing_sources = existing.get("inventory_sources")
    if isinstance(existing_sources, list):
        sources = {str(source) for source in existing_sources if str(source).strip()}
    elif existing.get("inventory_source"):
        sources = {str(existing.get("inventory_source"))}
    else:
        sources = set()
    sources.add(row_source)
    merged = {**existing, **row}
    for key in ("created_at", "title", "vod_duration_seconds"):
        existing_value = existing.get(key)
        row_value = row.get(key)
        if existing_value not in (None, "", 0, 0.0) and row_value in (None, "", 0, 0.0):
            merged[key] = existing_value
    merged["vod_id"] = vod_id
    merged["vod_url"] = str(merged.get("vod_url") or f"https://www.twitch.tv/videos/{vod_id}")
    merged["broadcaster"] = str(merged.get("broadcaster") or profile)
    merged["streamer"] = str(merged.get("streamer") or profile)
    merged["inventory_sources"] = sorted(sources)
    if "twitch_vod_api" in sources:
        merged["inventory_source"] = "twitch_vod_api"
    else:
        merged["inventory_source"] = str(merged.get("inventory_source") or row_source or "unknown")
    rows_by_vod[vod_id] = merged

api_error = ""
api_rows = 0
if settings.twitch_client_id and settings.twitch_client_secret:
    try:
        client = TwitchClient(settings.twitch_client_id, settings.twitch_client_secret)
        user = _get_user(client, profile)
        for vod in _get_vods(client, str(user["id"]), date_from, date_to):
            vod_id = str(vod.get("id") or "").strip()
            if not vod_id:
                continue
            api_rows += 1
            add_row(
                {
                    "vod_id": vod_id,
                    "vod_url": str(vod.get("url") or f"https://www.twitch.tv/videos/{vod_id}"),
                    "vod_duration_seconds": int(vod.get("duration_seconds") or 0),
                    "created_at": str(vod.get("created_at") or ""),
                    "title": str(vod.get("title") or ""),
                    "broadcaster": profile,
                    "streamer": profile,
                    "inventory_source": "twitch_vod_api",
                }
            )
    except Exception as exc:
        api_error = f"{type(exc).__name__}: {exc}"
else:
    api_error = "missing_twitch_credentials"

cache_rows = 0
if vod_cache_root.exists():
    for vod_dir in sorted(path for path in vod_cache_root.iterdir() if path.is_dir()):
        vod_id = vod_dir.name
        metadata_candidates = [
            vod_dir / "full_vod_audio" / "source.metadata.json",
            vod_dir / "chat" / "chat.metadata.json",
            vod_dir / "audio.metadata.json",
        ]
        payloads = [_read_json(path) for path in metadata_candidates if path.exists()]
        if not payloads:
            continue
        source = {}
        for payload in payloads:
            if isinstance(payload.get("source"), dict):
                source.update(payload.get("source") or {})
        duration = max(
            _safe_float(payload.get("vod_duration_seconds"))
            or _safe_float(payload.get("duration_seconds"))
            or _safe_float(payload.get("expanded_end"))
            or _safe_float(payload.get("fingerprint_duration"))
            or _safe_float(payload.get("last_t"))
            for payload in payloads
        )
        created_at = next((str(payload.get("created_at") or "") for payload in payloads if payload.get("created_at")), "")
        vod_url = next(
            (
                str(payload.get("vod_url") or "")
                for payload in payloads
                if str(payload.get("vod_url") or "").strip()
            ),
            "",
        )
        if not vod_url:
            vod_url = str(source.get("vod_url") or f"https://www.twitch.tv/videos/{vod_id}")
        cache_rows += 1
        add_row(
            {
                "vod_id": vod_id,
                "vod_url": vod_url,
                "vod_duration_seconds": round(duration, 3),
                "created_at": created_at,
                "broadcaster": profile,
                "streamer": profile,
                "inventory_source": "vod_feature_cache",
            }
        )

rows = sorted(rows_by_vod.values(), key=lambda row: str(row.get("created_at") or ""), reverse=True)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
print(
    json.dumps(
        {
            "event": "twitch_vod_inventory_built",
            "path": str(output_path),
            "date_from": date_from,
            "date_to": date_to,
            "rows": len(rows),
            "api_rows": api_rows,
            "cache_rows": cache_rows,
            "api_error": api_error,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
)
PY
}

build_rollout_readiness_collection_queue() {
  "$PYTHON_BIN" - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

labels_path = Path(os.environ["EMPTY_TWITCH_LABELS_PATH"])
cross_platform_path = Path(os.environ["CROSS_PLATFORM_FINGERPRINT_ROWS_PATH"])
feature_source_path = Path(os.environ["OUTCOME_FEATURE_SOURCE_PATH"])
queue_path = Path(os.environ["ROLLOUT_READINESS_COLLECTION_QUEUE_PATH"])
plan_path = Path(os.environ["ROLLOUT_READINESS_COLLECTION_PLAN_PATH"])
target_groups = max(int(os.environ.get("ROLLOUT_READINESS_TARGET_NEW_VOD_GROUPS") or "20"), 0)
max_per_vod = max(int(os.environ.get("ROLLOUT_READINESS_MAX_CANDIDATES_PER_VOD") or "3"), 1)

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows

def vod_id(row: dict) -> str:
    raw = str(row.get("vod_id") or "").strip()
    if raw:
        return raw
    for key in ("vod_url", "source_url", "review_url", "url"):
        text = str(row.get(key) or "")
        marker = "twitch.tv/videos/"
        if marker in text:
            return text.split(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]
    return ""

def window(row: dict) -> tuple[float, float] | None:
    for start_key, end_key in (("window_start", "window_end"), ("final_start", "final_end")):
        try:
            start = float(row.get(start_key))
            end = float(row.get(end_key))
        except (TypeError, ValueError):
            continue
        if end > start:
            return start, end
    return None

labels: list[dict] = []
cross_rows = read_jsonl(cross_platform_path)
feature_rows = read_jsonl(feature_source_path)
covered_vods = {vod_id(row) for row in cross_rows if vod_id(row)}
covered_feature_vods = {
    vod_id(row)
    for row in feature_rows
    if vod_id(row) and isinstance(row.get("features"), dict) and row.get("features")
}
blocked_vods = covered_vods | covered_feature_vods

by_vod: dict[str, list[dict]] = defaultdict(list)
for row in labels:
    if str(row.get("label_class") or "").strip().lower() == "ambiguous":
        continue
    vid = vod_id(row)
    if not vid or vid in blocked_vods or window(row) is None:
        continue
    by_vod[vid].append(row)

selected: list[dict] = []
selected_groups: list[str] = []
for vid, rows in sorted(by_vod.items(), key=lambda item: (-len(item[1]), item[0])):
    if len(selected_groups) >= target_groups:
        break
    selected_groups.append(vid)
    for row in sorted(rows, key=lambda item: float(item.get("view_count") or item.get("views") or 0), reverse=True)[:max_per_vod]:
        start, end = window(row) or (0.0, 1.0)
        copied = dict(row)
        copied["vod_id"] = vid
        copied["vod_url"] = str(copied.get("vod_url") or f"https://www.twitch.tv/videos/{vid}")
        copied["window_start"] = round(start, 2)
        copied["window_end"] = round(end, 2)
        copied["label"] = 1
        copied["label_class"] = str(copied.get("label_class") or "positive")
        copied["targeted_feature_backfill"] = True
        copied["targeted_feature_backfill_source"] = "rollout_readiness_collection"
        selected.append(copied)

queue_path.parent.mkdir(parents=True, exist_ok=True)
queue_path.write_text(
    "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in selected),
    encoding="utf-8",
)
plan = {
    "event": "rollout_readiness_collection_queue_built",
    "queue_path": str(queue_path),
    "labels_path": str(labels_path),
    "input_label_rows": len(labels),
    "cross_platform_rows": len(cross_rows),
    "feature_source_rows": len(feature_rows),
    "blocked_vods": len(blocked_vods),
    "candidate_vods": len(by_vod),
    "target_new_vod_groups": target_groups,
    "max_candidates_per_vod": max_per_vod,
    "selected_vods": len(selected_groups),
    "selected_rows": len(selected),
    "selected_vod_ids": selected_groups,
}
plan_path.parent.mkdir(parents=True, exist_ok=True)
plan_path.write_text(json.dumps(plan, ensure_ascii=True, indent=2), encoding="utf-8")
print(json.dumps(plan, ensure_ascii=True, sort_keys=True))
PY
}

build_targeted_backfill_feature_cache() {
  "$PYTHON_BIN" -m ml.targeted_backfill build-cache \
    --queue "$ROLLOUT_READINESS_COLLECTION_QUEUE_PATH" \
    --output "$TARGETED_BACKFILL_FEATURE_ROWS_PATH" \
    --summary "$TARGETED_BACKFILL_FEATURE_SUMMARY_PATH" \
    --work-dir "$WORK_DIR" \
    --profile jynxzi \
    --require-all-vods "$TARGETED_BACKFILL_REQUIRE_ALL_VODS"
}

refresh_cross_platform_vod_manifest_from_cache() {
  "$PYTHON_BIN" -m ml.targeted_backfill refresh-vod-manifest \
    --vod-cache-root "$BACKEND_DIR/tmp/vod_feature_cache" \
    --manifest "$CROSS_PLATFORM_VOD_MANIFEST_PATH" \
    --summary "$TARGETED_BACKFILL_VOD_MANIFEST_SUMMARY_PATH"
}

guard_targeted_backfill_before_rematch() {
  "$PYTHON_BIN" -m ml.targeted_backfill guard-rematch \
    --queue "$ROLLOUT_READINESS_COLLECTION_QUEUE_PATH" \
    --feature-rows "$TARGETED_BACKFILL_FEATURE_ROWS_PATH" \
    --vod-manifest "$CROSS_PLATFORM_VOD_MANIFEST_PATH" \
    --summary "$TARGETED_BACKFILL_PRE_REMATCH_GUARD_PATH" \
    --profile jynxzi \
    --require-all-vods "$TARGETED_BACKFILL_REQUIRE_ALL_VODS"
}

export_hard_negative_rows() {
  if [[ "$VOD_RANKER_USE_USER_LABELS" != "1" ]]; then
    reset_user_label_artifacts
    echo "User-labeled hard-negative export disabled."
    return 0
  fi
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

export_serving_hard_negative_rows() {
  if [[ "$SERVING_HARD_NEGATIVE_ENABLED" != "1" ]]; then
    echo "Serving hard-negative export disabled."
    return 0
  fi
  "$PYTHON_BIN" ml/export_serving_hard_negatives.py \
    --storage-dir "$BACKEND_DIR/tmp/storage" \
    --positive-rows "$CROSS_PLATFORM_FINGERPRINT_ROWS_PATH" \
    --holdout-policy-rows "$TRAINING_HOLDOUT_POLICY_ROWS_PATH" \
    --rows-out "$SERVING_HARD_NEGATIVE_ROWS_PATH" \
    --summary-out "$SERVING_HARD_NEGATIVE_SUMMARY_PATH" \
    --profile jynxzi \
    --pre-judge-top-k "$SERVING_HARD_NEGATIVE_PRE_JUDGE_TOP_K" \
    --max-per-vod "$SERVING_HARD_NEGATIVE_MAX_PER_VOD" \
    --max-rows "$SERVING_HARD_NEGATIVE_MAX_ROWS" \
    --positive-exclusion-overlap "$SERVING_HARD_NEGATIVE_POSITIVE_EXCLUSION_OVERLAP" \
    --min-rank-score "$SERVING_HARD_NEGATIVE_MIN_RANK_SCORE"
}

refresh_outcome_feature_source() {
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from ml.row_identity import row_identity_keys

output_path = Path(os.environ["OUTCOME_FEATURE_SOURCE_PATH"])
input_paths = [
    Path("ml/artifacts/jynxzi_outcome_feature_source_rows.jsonl"),
    Path(os.environ["TARGETED_BACKFILL_FEATURE_ROWS_PATH"]),
    Path("ml/artifacts/jynxzi_vod_training_rows.jsonl"),
    Path("ml/artifacts/jynxzi_vod_eval_dataset.jsonl"),
    Path("ml/artifacts/jynxzi_outcome_train_dataset.jsonl"),
    Path("ml/artifacts/jynxzi_outcome_holdout_dataset.jsonl"),
    Path("ml/artifacts/jynxzi_outcome_train_dataset.pre_candidate_source_fix.jsonl"),
    Path("ml/artifacts/jynxzi_outcome_holdout_dataset.pre_candidate_source_fix.jsonl"),
]

rows_by_identity: dict[str, dict] = {}
rows_by_candidate: dict[str, dict] = {}
input_rows = 0
feature_rows = 0
for path in input_paths:
    if not path.exists():
        continue
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        input_rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or not isinstance(row.get("features"), dict) or not row["features"]:
            continue
        identities = sorted(row_identity_keys(row))
        if not identities:
            continue
        feature_rows += 1
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id:
            rows_by_candidate.setdefault(candidate_id, row)
        for identity in identities:
            rows_by_identity.setdefault(identity, row)

unique_rows: dict[str, dict] = {}
for row in rows_by_identity.values():
    candidate_id = str(row.get("candidate_id") or "").strip()
    if candidate_id:
        key = f"candidate:{candidate_id}"
    else:
        identities = sorted(row_identity_keys(row))
        key = identities[0] if identities else str(id(row))
    unique_rows.setdefault(key, row)

output_path.parent.mkdir(parents=True, exist_ok=True)
ordered = sorted(
    unique_rows.values(),
    key=lambda row: (
        str(row.get("vod_id") or ""),
        float(row.get("window_start") or row.get("start_time") or 0.0),
        str(row.get("candidate_id") or ""),
    ),
)
output_path.write_text(
    "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in ordered),
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "event": "outcome_feature_source_refreshed",
            "path": str(output_path),
            "input_rows": input_rows,
            "feature_rows": feature_rows,
            "output_rows": len(ordered),
        },
        ensure_ascii=True,
    )
)
PY
}

export_auto_outcome_negative_rows() {
  if [[ "$AUTO_OUTCOME_NEGATIVE_ENABLED" != "1" ]]; then
    echo "Auto-outcome negative export disabled."
    return 0
  fi
  "$PYTHON_BIN" ml/export_auto_outcome_negatives.py \
    --candidate-rows "$EVAL_DATASET_PATH" \
    --candidate-rows "$DATASET_PATH" \
    --positive-rows "$CROSS_PLATFORM_FINGERPRINT_ROWS_PATH" \
    --rows-out "$AUTO_OUTCOME_NEGATIVE_ROWS_PATH" \
    --train-out "$AUTO_OUTCOME_NEGATIVE_TRAIN_ROWS_PATH" \
    --holdout-out "$AUTO_OUTCOME_NEGATIVE_HOLDOUT_ROWS_PATH" \
    --summary-out "$AUTO_OUTCOME_NEGATIVE_SUMMARY_PATH" \
    --profile jynxzi \
    --min-age-days "$AUTO_OUTCOME_NEGATIVE_MIN_AGE_DAYS" \
    --holdout-ratio "$AUTO_OUTCOME_NEGATIVE_HOLDOUT_RATIO" \
    --positive-exclusion-sec "$AUTO_OUTCOME_NEGATIVE_POSITIVE_EXCLUSION_SEC" \
    --hard-score-threshold "$AUTO_OUTCOME_NEGATIVE_HARD_SCORE_THRESHOLD" \
    --hard-sample-weight "$VOD_RANKER_AUTO_OUTCOME_HARD_NEGATIVE_WEIGHT" \
    --background-sample-weight "$VOD_RANKER_AUTO_OUTCOME_BACKGROUND_NEGATIVE_WEIGHT" \
    --max-rows "$AUTO_OUTCOME_NEGATIVE_MAX_ROWS"
}

build_outcome_dataset() {
  local split="$1"
  local auto_negative_rows="$2"
  local output_path="$3"
  local summary_path="$4"
  local target_positive_rows="${5:-0}"
  "$PYTHON_BIN" ml/build_outcome_holdout_dataset.py \
    --positive-rows "$CROSS_PLATFORM_FINGERPRINT_ROWS_PATH" \
    --auto-negative-rows "$auto_negative_rows" \
    --split "$split" \
    --output "$output_path" \
    --summary-out "$summary_path" \
    --target-positive-rows "$target_positive_rows" \
    --feature-source-rows "$OUTCOME_FEATURE_SOURCE_PATH" \
    --cross-platform-holdout-ratio "$CROSS_PLATFORM_HOLDOUT_RATIO"
}

build_outcome_train_dataset() {
  build_outcome_dataset train "$AUTO_OUTCOME_NEGATIVE_TRAIN_ROWS_PATH" "$OUTCOME_TRAIN_DATASET_PATH" "$OUTCOME_TRAIN_SUMMARY_PATH" "$VOD_RANKER_MIN_CROSS_PLATFORM_TRAIN_ROWS"
}

build_outcome_holdout_dataset() {
  build_outcome_dataset holdout "$AUTO_OUTCOME_NEGATIVE_HOLDOUT_ROWS_PATH" "$OUTCOME_HOLDOUT_DATASET_PATH" "$OUTCOME_HOLDOUT_SUMMARY_PATH" "$OUTCOME_HOLDOUT_TARGET_POSITIVE_ROWS"
}

build_recent_vod_holdout_policy() {
  local refresh_args=()
  if [[ "$RECENT_VOD_HOLDOUT_REFRESH" == "1" ]]; then
    refresh_args+=(--refresh)
  fi
  "$PYTHON_BIN" -m ml.build_recent_vod_holdout \
    --profile jynxzi \
    --vod-inventory "$VOD_INVENTORY_PATH" \
    --output "$RECENT_VOD_HOLDOUT_POLICY_PATH" \
    --summary-output "$RECENT_VOD_HOLDOUT_SUMMARY_PATH" \
    --count "$([[ "$RECENT_VOD_HOLDOUT_ENABLED" == "1" ]] && echo "$RECENT_VOD_HOLDOUT_COUNT" || echo 0)" \
    "${refresh_args[@]}"
}

build_training_holdout_policy_rows() {
  "$PYTHON_BIN" -m ml.build_training_holdout_policy \
    --input "$OUTCOME_HOLDOUT_DATASET_PATH" \
    --input "$RECENT_VOD_HOLDOUT_POLICY_PATH" \
    --output "$TRAINING_HOLDOUT_POLICY_ROWS_PATH" \
    --summary-output "$TRAINING_HOLDOUT_POLICY_SUMMARY_PATH"
}

select_rollout_eval_report() {
  "$PYTHON_BIN" - <<'PY'
import json
import shutil
from pathlib import Path

outcome_eval = Path("ml/artifacts/jynxzi_outcome_holdout_eval.json")
gold_eval = Path("ml/artifacts/jynxzi_gold_eval.json")
ranker_eval = Path("ml/artifacts/jynxzi_ranker_eval.json")
source_out = Path("ml/artifacts/jynxzi_rollout_eval_source.json")

def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

outcome = load(outcome_eval)
gold = load(gold_eval)
outcome_rows = int(outcome.get("rows") or 0)
outcome_positive = int(outcome.get("positive_rows") or 0)
outcome_negative = int(outcome.get("negative_rows") or 0)
use_outcome = outcome_rows > 0 and outcome_positive > 0 and outcome_negative > 0
source = "outcome_holdout" if use_outcome else "gold_eval"
selected = outcome_eval if use_outcome else gold_eval
if selected.exists():
    shutil.copyfile(selected, ranker_eval)
source_out.write_text(
    json.dumps(
        {
            "selected_source": source,
            "selected_path": str(selected),
            "outcome_rows": outcome_rows,
            "outcome_positive_rows": outcome_positive,
            "outcome_negative_rows": outcome_negative,
            "gold_rows": int(gold.get("rows") or 0),
            "reason": "outcome_holdout_has_both_classes" if use_outcome else "outcome_holdout_missing_or_one_class",
        },
        ensure_ascii=True,
        indent=2,
    ),
    encoding="utf-8",
)
PY
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
echo "Using VOD ranker gold-train merge: $VOD_RANKER_USE_GOLD_TRAIN"
echo "Using VOD ranker gold hard-negative merge: $VOD_RANKER_USE_GOLD_HARD_NEGATIVES"
echo "Using user-labeled gold/manual rows: $VOD_RANKER_USE_USER_LABELS"
echo "Using VOD ranker primary label policy: $VOD_RANKER_PRIMARY_LABEL_POLICY"
echo "Using VOD ranker required cross-platform rows: $VOD_RANKER_MIN_CROSS_PLATFORM_TRAIN_ROWS"
echo "Using cross-platform seed minimum rows: $CROSS_PLATFORM_SEED_MIN_MATCHED_ROWS"
echo "Using cross-platform strict minimum rows/vods: $CROSS_PLATFORM_MIN_MATCHED_ROWS/$CROSS_PLATFORM_MIN_MATCHED_VODS"
echo "Using cross-platform verification mode: $CROSS_PLATFORM_VERIFICATION_MODE"
echo "Using cross-platform transcript match workers: $CROSS_PLATFORM_TRANSCRIPT_MATCH_WORKERS"
echo "Using cross-platform second pass after VOD build: $CROSS_PLATFORM_SECOND_PASS_AFTER_VOD_BUILD"
echo "Using cross-platform full-VOD audio fingerprints: $CROSS_PLATFORM_FULL_VOD_AUDIO_FINGERPRINTS max_vods=${CROSS_PLATFORM_FULL_VOD_AUDIO_MAX_VODS} workers=${CROSS_PLATFORM_FULL_VOD_AUDIO_WORKERS} keep_audio=${CROSS_PLATFORM_KEEP_FULL_VOD_AUDIO}"
echo "Using cross-platform coarse VOD tiling: $CROSS_PLATFORM_COARSE_VOD_TILING tile=${CROSS_PLATFORM_COARSE_VOD_TILE_SEC}s stride=${CROSS_PLATFORM_COARSE_VOD_STRIDE_SEC}s max_tiles=${CROSS_PLATFORM_COARSE_VOD_MAX_TILES_PER_VOD}"
echo "Using targeted backfill require-all-vods: $TARGETED_BACKFILL_REQUIRE_ALL_VODS"
echo "Using recent VOD holdout: enabled=$RECENT_VOD_HOLDOUT_ENABLED count=$RECENT_VOD_HOLDOUT_COUNT refresh=$RECENT_VOD_HOLDOUT_REFRESH"
echo "Using VOD ranker MM-judge proxy: $VOD_RANKER_USE_MM_JUDGE_PROXY weight=$VOD_RANKER_MM_JUDGE_WEIGHT"
echo "Using VOD ranker rollout-review-negative weight: $VOD_RANKER_ROLLOUT_REVIEW_NEGATIVE_WEIGHT"
echo "Using rollout threshold recall target: $VOD_RANKER_ROLLOUT_THRESHOLD_MIN_RECALL"
echo "Trainiere VOD von ${DATE_FROM} bis ${DATE_TO}"
if [[ "$VOD_RANKER_USE_USER_LABELS" != "1" ]]; then
  echo "Resetting generated user-labeled artifacts; model will use cross-platform outcomes and automatic negatives only."
  reset_user_label_artifacts
fi
echo "Building Twitch VOD inventory for cross-platform matching..."
build_twitch_vod_inventory
echo "Building latest-VOD holdout policy..."
build_recent_vod_holdout_policy
mkdir -p "$(dirname "$EMPTY_TWITCH_LABELS_PATH")"
: > "$EMPTY_TWITCH_LABELS_PATH"
echo "Twitch clip-label collection removed; training uses cross-platform outcomes from VOD inventory."

echo "Preparing cross-platform outcome rows..."
prepare_cross_platform_fingerprint_rows "$CROSS_PLATFORM_SEED_MIN_MATCHED_ROWS" 0 0 "$CROSS_PLATFORM_REUSE_EXISTING_ROWS"

echo "Preparing gold eval artifacts for training merge..."
export_gold_eval_rows

echo "Skipping Twitch clip-label pruning; clip labels are not part of this training flow."

echo "Building rollout-readiness feature-backfill queue..."
build_rollout_readiness_collection_queue

echo "Building targeted feature-backfill cache before outcome/rematch stages..."
build_targeted_backfill_feature_cache

echo "Preparing diagnosis hard-negative rows for training merge..."
export_hard_negative_rows

echo "Exporting serving hard-negative rows for training merge..."
export_serving_hard_negative_rows

echo "Exporting auto-outcome negative rows for training and holdout..."
export_auto_outcome_negative_rows

echo "Refreshing outcome feature source rows..."
refresh_outcome_feature_source

echo "Building label-light outcome train dataset..."
build_outcome_train_dataset

echo "Building label-light outcome holdout dataset..."
build_outcome_holdout_dataset

echo "Building merged training holdout policy rows..."
build_training_holdout_policy_rows

echo "Importing multimodal judge proxy rows..."
import_mm_judge_proxy_rows

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
            Path("ml/artifacts/jynxzi_outcome_train_dataset.jsonl"),
            Path("ml/artifacts/jynxzi_outcome_holdout_dataset.jsonl"),
            Path(os.environ["RECENT_VOD_HOLDOUT_POLICY_PATH"]),
            Path(os.environ["TRAINING_HOLDOUT_POLICY_ROWS_PATH"]),
            Path(os.environ["OUTCOME_FEATURE_SOURCE_PATH"]),
            Path(os.environ["TARGETED_BACKFILL_FEATURE_ROWS_PATH"]),
            Path("ml/artifacts/jynxzi_auto_outcome_negative_train_rows.jsonl"),
            Path(os.environ["CROSS_PLATFORM_VOD_MANIFEST_PATH"]),
            Path(os.environ["CROSS_PLATFORM_REPOST_MANIFEST_PATH"]),
            Path(os.environ["CROSS_PLATFORM_SOURCE_MANIFEST_PATH"]),
            Path(os.environ["CROSS_PLATFORM_URL_MANIFEST_PATH"]),
            Path(os.environ["CROSS_PLATFORM_FINGERPRINT_ROWS_PATH"]),
            Path(os.environ["SERVING_HARD_NEGATIVE_ROWS_PATH"]),
            Path("ml/artifacts/jynxzi_cross_platform_fingerprint_summary.json"),
            Path(os.environ["MM_JUDGE_PROXY_ROWS_PATH"]),
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

dependency_paths = [
    Path(os.environ["OUTCOME_FEATURE_SOURCE_PATH"]),
    Path(os.environ["RECENT_VOD_HOLDOUT_POLICY_PATH"]),
    Path(os.environ["TRAINING_HOLDOUT_POLICY_ROWS_PATH"]),
    Path(os.environ["TARGETED_BACKFILL_FEATURE_ROWS_PATH"]),
    Path(os.environ["CROSS_PLATFORM_FINGERPRINT_ROWS_PATH"]),
    Path(os.environ["SERVING_HARD_NEGATIVE_ROWS_PATH"]),
    Path("ml/artifacts/jynxzi_cross_platform_fingerprint_summary.json"),
]
if os.environ.get("VOD_RANKER_USE_USER_LABELS") == "1":
    dependency_paths.append(Path("tmp/storage/feedback/gold_eval_updates.jsonl"))

print(
    "yes"
    if should_reevaluate(
        dataset_path=Path("ml/artifacts/jynxzi_vod_training_rows.jsonl"),
        artifact_path=Path("ml/artifacts/jynxzi_ranker.json"),
        output_paths=[
            Path("ml/artifacts/jynxzi_vod_eval_dataset.jsonl"),
            Path("ml/artifacts/jynxzi_vod_eval_dataset_summary.json"),
            Path("ml/artifacts/jynxzi_gold_eval.json"),
            Path("ml/artifacts/jynxzi_outcome_train_dataset.jsonl"),
            Path("ml/artifacts/jynxzi_outcome_train_summary.json"),
            Path("ml/artifacts/jynxzi_outcome_holdout_dataset.jsonl"),
            Path("ml/artifacts/jynxzi_outcome_holdout_summary.json"),
            Path("ml/artifacts/jynxzi_outcome_holdout_eval.json"),
            Path("ml/artifacts/jynxzi_rollout_eval_source.json"),
            Path("ml/artifacts/jynxzi_proxy_eval.json"),
            Path("ml/artifacts/jynxzi_auto_outcome_negative_eval.json"),
            Path("ml/artifacts/jynxzi_pipeline_eval.json"),
        ],
        dependency_paths=dependency_paths,
        state_path=Path("ml/artifacts/jynxzi_training_workflow_state.json"),
        current_state=json.loads(os.environ["CURRENT_WORKFLOW_STATE_JSON"]),
    )
    else "no"
)
PY
)"

if [[ "$SHOULD_TRAIN" != "yes" && "$SHOULD_EVALUATE" != "yes" ]]; then
  echo "Keine neuen Label-Daten seit dem letzten Dataset-/Modellstand. Überspringe Training und Evaluation."
  echo "VOD inventory:  $VOD_INVENTORY_PATH"
  echo "Dataset:        $DATASET_PATH"
  echo "Artifact:       $ARTIFACT_JSON_PATH"
  echo "Eval dataset:   $EVAL_DATASET_PATH"
  echo "Ranker eval:    $RANKER_EVAL_PATH"
  echo "Auto-negative:  $AUTO_OUTCOME_NEGATIVE_ROWS_PATH"
  echo "Recent holdout: $RECENT_VOD_HOLDOUT_POLICY_PATH"
  echo "Holdout policy: $TRAINING_HOLDOUT_POLICY_ROWS_PATH"
  echo "Pipeline eval:  $PIPELINE_EVAL_PATH"
  exit 0
fi

if [[ "$SHOULD_TRAIN" == "yes" ]]; then
  echo "Training VOD-first ranker for jynxzi..."
  rm -f "$STAGED_DATASET_PATH"
  if run_vod_training "$STAGED_DATASET_PATH"; then
    promote_staged_vod_dataset
    refresh_outcome_feature_source
  else
    DATASET_STATE="$(dataset_label_state "$STAGED_DATASET_PATH")"
    SHORT_FORM_STATE="$(dataset_label_state "$SHORT_FORM_PATH")"
    echo "Initial VOD-first training failed." >&2
    echo "Staged dataset:    $STAGED_DATASET_PATH" >&2
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
      rm -f "$STAGED_DATASET_PATH"
      run_vod_training "$STAGED_DATASET_PATH"
      promote_staged_vod_dataset
      refresh_outcome_feature_source
    else
      exit 1
    fi
  fi

  if [[ "$CROSS_PLATFORM_SECOND_PASS_AFTER_VOD_BUILD" == "1" ]]; then
    CROSS_PLATFORM_DIGEST_BEFORE="$("$PYTHON_BIN" - "$CROSS_PLATFORM_FINGERPRINT_ROWS_PATH" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "")
PY
)"
    echo "Refreshing VOD manifest and rematching cross-platform outcomes after feature-cache backfill..."
    refresh_cross_platform_vod_manifest_from_cache
    guard_targeted_backfill_before_rematch
    prepare_cross_platform_fingerprint_rows "$CROSS_PLATFORM_MIN_MATCHED_ROWS" "$CROSS_PLATFORM_MIN_MATCHED_VODS" 1 0
    CROSS_PLATFORM_DIGEST_AFTER="$("$PYTHON_BIN" - "$CROSS_PLATFORM_FINGERPRINT_ROWS_PATH" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "")
PY
)"
    if [[ "$CROSS_PLATFORM_DIGEST_AFTER" != "$CROSS_PLATFORM_DIGEST_BEFORE" ]]; then
      echo "Cross-platform outcomes changed after VOD-cache backfill. Rebuilding outcome datasets and retraining once."
      export_serving_hard_negative_rows
      export_auto_outcome_negative_rows
      refresh_outcome_feature_source
      build_outcome_train_dataset
      build_outcome_holdout_dataset
      build_training_holdout_policy_rows
      rm -f "$STAGED_DATASET_PATH"
      run_vod_training "$STAGED_DATASET_PATH"
      promote_staged_vod_dataset
      refresh_outcome_feature_source
    else
      echo "Cross-platform outcomes unchanged after VOD-cache backfill."
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
  echo "VOD inventory:  $VOD_INVENTORY_PATH"
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

echo "Preparing gold eval artifacts..."
export_gold_eval_rows

echo "Refreshing label-light outcome train dataset after VOD feature build..."
build_outcome_train_dataset

echo "Building label-light outcome holdout dataset..."
build_outcome_holdout_dataset

echo "Refreshing merged training holdout policy rows..."
build_training_holdout_policy_rows

echo "Calibrating recall-friendly ranker serving threshold on outcome-holdout rows..."
"$PYTHON_BIN" ml/calibrate_ranker_threshold.py \
  --dataset "$OUTCOME_HOLDOUT_DATASET_PATH" \
  --profile jynxzi \
  --artifact-dir "$ARTIFACT_DIR" \
  --source rollout_recall_calibration \
  --selection-mode recall_precision \
  --min-recall "$VOD_RANKER_ROLLOUT_THRESHOLD_MIN_RECALL" \
  --max-selected-rate "$VOD_RANKER_ROLLOUT_THRESHOLD_MAX_SELECTED_RATE" \
  --target-precision "$VOD_RANKER_ROLLOUT_THRESHOLD_TARGET_PRECISION" \
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

if [[ "$VOD_RANKER_USE_USER_LABELS" == "1" ]]; then
  echo "Evaluating trained ranker against heuristic baseline on gold eval rows..."
  "$PYTHON_BIN" ml/evaluate_ranker.py \
    --dataset "$GOLD_EVAL_ROWS_PATH" \
    --profile jynxzi \
    --artifact-dir "$ARTIFACT_DIR" \
    --report-out "$GOLD_RANKER_EVAL_PATH" \
    > /dev/null
else
  echo "Skipping gold eval; user-labeled rows are disabled."
  reset_user_label_artifacts
fi

echo "Evaluating trained ranker on label-light outcome holdout rows..."
"$PYTHON_BIN" ml/evaluate_ranker.py \
  --dataset "$OUTCOME_HOLDOUT_DATASET_PATH" \
  --profile jynxzi \
  --artifact-dir "$ARTIFACT_DIR" \
  --report-out "$OUTCOME_RANKER_EVAL_PATH" \
  > /dev/null

echo "Selecting primary rollout eval report..."
select_rollout_eval_report

echo "Evaluating trained ranker on auto-outcome negative holdout rows..."
"$PYTHON_BIN" ml/evaluate_ranker.py \
  --dataset "$AUTO_OUTCOME_NEGATIVE_HOLDOUT_ROWS_PATH" \
  --profile jynxzi \
  --artifact-dir "$ARTIFACT_DIR" \
  --report-out "$AUTO_OUTCOME_NEGATIVE_EVAL_PATH" \
  > /dev/null || true

if [[ "$VOD_RANKER_USE_USER_LABELS" == "1" ]]; then
  echo "Diagnosing ranker hard negatives for the next run..."
  "$PYTHON_BIN" ml/diagnose_ranker_holdout.py \
    --dataset "$GOLD_EVAL_ROWS_PATH" \
    --profile jynxzi \
    --artifact-dir "$ARTIFACT_DIR" \
    --report-out "$RANKER_HOLDOUT_DIAGNOSIS_JSON_PATH" \
    --csv-out "$RANKER_HOLDOUT_DIAGNOSIS_CSV_PATH" \
    > /dev/null || true
  export_hard_negative_rows
else
  echo "Skipping gold-derived hard-negative diagnosis; user-labeled rows are disabled."
  reset_user_label_artifacts
fi

echo "Diagnosing rollout readiness coverage..."
"$PYTHON_BIN" ml/diagnose_rollout_readiness_coverage.py \
  --dataset "$DATASET_PATH" \
  --profile jynxzi \
  --holdout-dataset "$OUTCOME_HOLDOUT_DATASET_PATH" \
  --report-out "$ROLLOUT_READINESS_COVERAGE_PATH" \
  > /dev/null || true

echo "Evaluating trained product pipeline ranking..."
"$PYTHON_BIN" ml/evaluate_pipeline.py \
  --dataset "$EVAL_DATASET_PATH" \
  --profile jynxzi \
  --artifact-dir "$ARTIFACT_DIR" \
  --evaluation-mode trained_product \
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

ranker_eval = Path("ml/artifacts/jynxzi_ranker_eval.json")
eval_source_path = Path("ml/artifacts/jynxzi_rollout_eval_source.json")
if not ranker_eval.exists():
    raise SystemExit("Missing ranker eval output: ml/artifacts/jynxzi_ranker_eval.json")

report = json.loads(ranker_eval.read_text(encoding="utf-8"))
eval_source = json.loads(eval_source_path.read_text(encoding="utf-8")) if eval_source_path.exists() else {}
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
    "eval_source": eval_source,
    "artifact": artifact,
    "trained_runtime": trained,
    "baseline_heuristic": baseline,
    "delta": delta,
}
print(json.dumps(summary, ensure_ascii=True, indent=2))
PY

if [[ "$VOD_RANKER_USE_USER_LABELS" == "1" ]]; then
  echo "Building two-stage rollout review queue..."
  if ! "$PYTHON_BIN" ml/build_rollout_review_queue.py \
      --dataset "$EVAL_DATASET_PATH" \
      --existing-gold-rows "$GOLD_EVAL_ROWS_PATH" \
      --holdout-rows "$GOLD_EVAL_ROWS_PATH" \
      --profile jynxzi \
      --artifact-dir "$ARTIFACT_DIR" \
      --out "$ROLLOUT_REVIEW_QUEUE_BASE" \
      --top-k-per-vod "$ROLLOUT_REVIEW_TOP_K_PER_VOD" \
      --max-total "$ROLLOUT_REVIEW_MAX_TOTAL" \
      --min-gap-sec "$ROLLOUT_REVIEW_MIN_GAP_SEC" \
      --review-threshold "$ROLLOUT_REVIEW_THRESHOLD" \
      --review-threshold-margin "$ROLLOUT_REVIEW_THRESHOLD_MARGIN" \
      --disagreement-heuristic-threshold "$ROLLOUT_DISAGREEMENT_HEURISTIC_THRESHOLD" \
      --disagreement-min-gap "$ROLLOUT_DISAGREEMENT_MIN_GAP" \
      --max-disagreement-per-vod "$ROLLOUT_MAX_DISAGREEMENT_PER_VOD" \
      > /dev/null; then
    echo "Warning: rollout review queue was not generated." >&2
  fi
else
  echo "Skipping rollout review queue; user-labeled rows are disabled."
  reset_user_label_artifacts
fi

echo
echo "Done."
echo "VOD inventory:  $VOD_INVENTORY_PATH"
echo "Dataset:        $DATASET_PATH"
echo "Artifact:       $ARTIFACT_JSON_PATH"
echo "Boundary:       $BOUNDARY_ARTIFACT_JSON_PATH"
echo "Eval dataset:   $EVAL_DATASET_PATH"
echo "Eval summary:   $EVAL_SUMMARY_PATH"
echo "Gold rows:      $GOLD_EVAL_ROWS_PATH"
echo "Gold eval:      $GOLD_RANKER_EVAL_PATH"
echo "Outcome train:  $OUTCOME_TRAIN_DATASET_PATH"
echo "Outcome holdout:$OUTCOME_HOLDOUT_DATASET_PATH"
echo "Outcome eval:   $OUTCOME_RANKER_EVAL_PATH"
echo "Rollout eval:   $RANKER_EVAL_PATH"
echo "Proxy eval:     $PROXY_RANKER_EVAL_PATH"
echo "Ranker eval:    $RANKER_EVAL_PATH"
echo "Pipeline eval:  $PIPELINE_EVAL_PATH"
echo "Hard negatives: $HARD_NEGATIVE_ROWS_PATH"
echo "Rollout queue:  $ROLLOUT_REVIEW_QUEUE_CSV_PATH"
