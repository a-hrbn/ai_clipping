#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_LOCAL_DOCKER="$SCRIPT_DIR/train_local_docker.sh"

PROFILE="${PROPOSAL_PROFILE:-${CLIP_PROFILE:-jynxzi}}"
STORAGE_DIR="${PROPOSAL_STORAGE_DIR:-/app/backend/tmp/storage}"
POSITIVE_ROWS="${PROPOSAL_POSITIVE_ROWS:-/app/backend/ml/artifacts/${PROFILE}_cross_platform_fingerprint_rows.jsonl}"
HOLDOUT_POLICY_ROWS="${PROPOSAL_HOLDOUT_POLICY_ROWS:-/app/backend/ml/artifacts/${PROFILE}_training_holdout_policy_rows.jsonl}"
TRAINING_ROWS="${PROPOSAL_TRAINING_ROWS:-/app/backend/ml/artifacts/${PROFILE}_proposal_training_rows.jsonl}"
OUTPUT_PATH="${PROPOSAL_OUTPUT:-/app/backend/ml/artifacts/${PROFILE}_proposal_scorer.json}"
MIN_ROWS="${PROPOSAL_MIN_ROWS:-8}"
VOD_CACHE_ROOT="${PROPOSAL_VOD_CACHE_ROOT:-/app/backend/tmp/vod_feature_cache}"
MATERIALIZED_STORAGE_DIR="${PROPOSAL_MATERIALIZED_STORAGE_DIR:-${STORAGE_DIR}/_proposal_training_materialized}"
MATERIALIZE_MAX_VODS="${PROPOSAL_MATERIALIZE_MAX_VODS:-20}"
MATERIALIZE_REFRESH="${PROPOSAL_MATERIALIZE_REFRESH:-false}"
DURATIONS_SEC="${PROPOSAL_DURATIONS_SEC:-12,20,30,45,60}"
STRIDE_SEC="${PROPOSAL_STRIDE_SEC:-5.0}"
MAX_CANDIDATES_PER_VOD="${PROPOSAL_MAX_CANDIDATES_PER_VOD:-5000}"
CONTEXT_PATTERN_SEMANTIC_CLUSTER_SIMILARITY="${PROPOSAL_CONTEXT_PATTERN_SEMANTIC_CLUSTER_SIMILARITY:-0.72}"

show_help() {
  cat <<'EOF'
Usage:
  scripts/train_proposal_local_docker.sh [options] [-- extra train_proposal_pipeline.py args]

Options:
  --profile NAME             Profile/artifact prefix. Default: PROPOSAL_PROFILE, CLIP_PROFILE, then jynxzi.
  --storage-dir PATH         Container storage dir with job debug folders. Default: /app/backend/tmp/storage.
  --positive-rows PATH       Optional container JSONL with cross-platform positive windows.
                             Default: /app/backend/ml/artifacts/<profile>_cross_platform_fingerprint_rows.jsonl.
  --holdout-policy-rows PATH Optional container JSONL with holdout policy rows to exclude from training.
                             Default: /app/backend/ml/artifacts/<profile>_training_holdout_policy_rows.jsonl.
  --training-rows PATH       Optional container JSONL from build_proposal_training_rows.py.
                             Default: /app/backend/ml/artifacts/<profile>_proposal_training_rows.jsonl.
  --output PATH              Container output artifact path.
                             Default: /app/backend/ml/artifacts/<profile>_proposal_scorer.json.
  --min-rows N               Minimum labelled rows required before writing trained weights. Default: 8.
  --vod-cache-root PATH      Container VOD feature cache root with full-VOD transcripts.
                             Default: /app/backend/tmp/vod_feature_cache.
  --materialized-storage-dir PATH
                             Container dir for generated proposal debug jobs.
                             Default: <storage-dir>/_proposal_training_materialized.
  --materialize-max-vods N   Number of positive VODs to materialize before row building. Default: 20.
  --materialize-refresh      Rebuild generated proposal debug jobs even when cached.
  --durations-sec LIST       Dense proposal durations, comma separated. Default: 12,20,30,45,60.
  --stride-sec N             Dense proposal stride in seconds. Default: 5.0.
  --max-candidates-per-vod N Maximum generated proposal candidates per VOD. Default: 5000.
  --context-pattern-semantic-cluster-similarity N
                             Cosine threshold for semantic context-pattern clustering. Default: 0.72.
  --date-from DATE          Accepted for compatibility with train_local_docker.sh; ignored.
  --date-to DATE            Accepted for compatibility with train_local_docker.sh; ignored.
  -h, --help                 Show this help.

Environment:
  PROPOSAL_PROFILE, PROPOSAL_STORAGE_DIR, PROPOSAL_POSITIVE_ROWS, PROPOSAL_TRAINING_ROWS,
  PROPOSAL_HOLDOUT_POLICY_ROWS, PROPOSAL_OUTPUT, PROPOSAL_MIN_ROWS, PROPOSAL_VOD_CACHE_ROOT,
  PROPOSAL_MATERIALIZED_STORAGE_DIR, PROPOSAL_MATERIALIZE_MAX_VODS, PROPOSAL_MATERIALIZE_REFRESH, PROPOSAL_DURATIONS_SEC,
  PROPOSAL_STRIDE_SEC, PROPOSAL_MAX_CANDIDATES_PER_VOD,
  PROPOSAL_CONTEXT_PATTERN_SEMANTIC_CLUSTER_SIMILARITY override the defaults above.

Examples:
  scripts/train_proposal_local_docker.sh
  scripts/train_proposal_local_docker.sh --profile jynxzi --min-rows 20
  scripts/train_proposal_local_docker.sh -- --min-rows 40
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:?Missing value for --profile}"
      POSITIVE_ROWS="${PROPOSAL_POSITIVE_ROWS:-/app/backend/ml/artifacts/${PROFILE}_cross_platform_fingerprint_rows.jsonl}"
      HOLDOUT_POLICY_ROWS="${PROPOSAL_HOLDOUT_POLICY_ROWS:-/app/backend/ml/artifacts/${PROFILE}_training_holdout_policy_rows.jsonl}"
      TRAINING_ROWS="${PROPOSAL_TRAINING_ROWS:-/app/backend/ml/artifacts/${PROFILE}_proposal_training_rows.jsonl}"
      OUTPUT_PATH="${PROPOSAL_OUTPUT:-/app/backend/ml/artifacts/${PROFILE}_proposal_scorer.json}"
      shift 2
      ;;
    --profile=*)
      PROFILE="${1#*=}"
      POSITIVE_ROWS="${PROPOSAL_POSITIVE_ROWS:-/app/backend/ml/artifacts/${PROFILE}_cross_platform_fingerprint_rows.jsonl}"
      HOLDOUT_POLICY_ROWS="${PROPOSAL_HOLDOUT_POLICY_ROWS:-/app/backend/ml/artifacts/${PROFILE}_training_holdout_policy_rows.jsonl}"
      TRAINING_ROWS="${PROPOSAL_TRAINING_ROWS:-/app/backend/ml/artifacts/${PROFILE}_proposal_training_rows.jsonl}"
      OUTPUT_PATH="${PROPOSAL_OUTPUT:-/app/backend/ml/artifacts/${PROFILE}_proposal_scorer.json}"
      shift
      ;;
    --storage-dir)
      STORAGE_DIR="${2:?Missing value for --storage-dir}"
      MATERIALIZED_STORAGE_DIR="${PROPOSAL_MATERIALIZED_STORAGE_DIR:-${STORAGE_DIR}/_proposal_training_materialized}"
      shift 2
      ;;
    --storage-dir=*)
      STORAGE_DIR="${1#*=}"
      MATERIALIZED_STORAGE_DIR="${PROPOSAL_MATERIALIZED_STORAGE_DIR:-${STORAGE_DIR}/_proposal_training_materialized}"
      shift
      ;;
    --positive-rows)
      POSITIVE_ROWS="${2:?Missing value for --positive-rows}"
      shift 2
      ;;
    --positive-rows=*)
      POSITIVE_ROWS="${1#*=}"
      shift
      ;;
    --holdout-policy-rows)
      HOLDOUT_POLICY_ROWS="${2:?Missing value for --holdout-policy-rows}"
      shift 2
      ;;
    --holdout-policy-rows=*)
      HOLDOUT_POLICY_ROWS="${1#*=}"
      shift
      ;;
    --training-rows)
      TRAINING_ROWS="${2:?Missing value for --training-rows}"
      shift 2
      ;;
    --training-rows=*)
      TRAINING_ROWS="${1#*=}"
      shift
      ;;
    --output)
      OUTPUT_PATH="${2:?Missing value for --output}"
      shift 2
      ;;
    --output=*)
      OUTPUT_PATH="${1#*=}"
      shift
      ;;
    --min-rows)
      MIN_ROWS="${2:?Missing value for --min-rows}"
      shift 2
      ;;
    --min-rows=*)
      MIN_ROWS="${1#*=}"
      shift
      ;;
    --vod-cache-root)
      VOD_CACHE_ROOT="${2:?Missing value for --vod-cache-root}"
      shift 2
      ;;
    --vod-cache-root=*)
      VOD_CACHE_ROOT="${1#*=}"
      shift
      ;;
    --materialized-storage-dir)
      MATERIALIZED_STORAGE_DIR="${2:?Missing value for --materialized-storage-dir}"
      shift 2
      ;;
    --materialized-storage-dir=*)
      MATERIALIZED_STORAGE_DIR="${1#*=}"
      shift
      ;;
    --materialize-max-vods)
      MATERIALIZE_MAX_VODS="${2:?Missing value for --materialize-max-vods}"
      shift 2
      ;;
    --materialize-max-vods=*)
      MATERIALIZE_MAX_VODS="${1#*=}"
      shift
      ;;
    --materialize-refresh)
      MATERIALIZE_REFRESH="true"
      shift
      ;;
    --durations-sec)
      DURATIONS_SEC="${2:?Missing value for --durations-sec}"
      shift 2
      ;;
    --durations-sec=*)
      DURATIONS_SEC="${1#*=}"
      shift
      ;;
    --stride-sec)
      STRIDE_SEC="${2:?Missing value for --stride-sec}"
      shift 2
      ;;
    --stride-sec=*)
      STRIDE_SEC="${1#*=}"
      shift
      ;;
    --max-candidates-per-vod)
      MAX_CANDIDATES_PER_VOD="${2:?Missing value for --max-candidates-per-vod}"
      shift 2
      ;;
    --max-candidates-per-vod=*)
      MAX_CANDIDATES_PER_VOD="${1#*=}"
      shift
      ;;
    --context-pattern-semantic-cluster-similarity)
      CONTEXT_PATTERN_SEMANTIC_CLUSTER_SIMILARITY="${2:?Missing value for --context-pattern-semantic-cluster-similarity}"
      shift 2
      ;;
    --context-pattern-semantic-cluster-similarity=*)
      CONTEXT_PATTERN_SEMANTIC_CLUSTER_SIMILARITY="${1#*=}"
      shift
      ;;
    --date-from|--date-to)
      shift 2
      ;;
    --date-from=*|--date-to=*)
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "$TRAIN_LOCAL_DOCKER" ]]; then
  echo "Missing executable training wrapper: $TRAIN_LOCAL_DOCKER" >&2
  exit 1
fi

cd "$REPO_ROOT"

MATERIALIZE_ARGS=()
if [[ "$MATERIALIZE_REFRESH" == "true" ]]; then
  MATERIALIZE_ARGS+=(--materialize-refresh)
fi

AI_CLIPPING_LOCAL_TRAIN_ENTRYPOINT="${AI_CLIPPING_LOCAL_TRAIN_ENTRYPOINT:-python}" \
exec "$TRAIN_LOCAL_DOCKER" \
  -m ml.train_proposal_pipeline \
  --profile "$PROFILE" \
  --storage-dir "$STORAGE_DIR" \
  --positive-rows "$POSITIVE_ROWS" \
  --holdout-policy-rows "$HOLDOUT_POLICY_ROWS" \
  --training-rows "$TRAINING_ROWS" \
  --output "$OUTPUT_PATH" \
  --min-rows "$MIN_ROWS" \
  --vod-cache-root "$VOD_CACHE_ROOT" \
  --materialized-storage-dir "$MATERIALIZED_STORAGE_DIR" \
  --materialize-max-vods "$MATERIALIZE_MAX_VODS" \
  "${MATERIALIZE_ARGS[@]}" \
  --durations-sec "$DURATIONS_SEC" \
  --stride-sec "$STRIDE_SEC" \
  --max-candidates-per-vod "$MAX_CANDIDATES_PER_VOD" \
  --context-pattern-semantic-cluster-similarity "$CONTEXT_PATTERN_SEMANTIC_CLUSTER_SIMILARITY" \
  "${EXTRA_ARGS[@]}"
