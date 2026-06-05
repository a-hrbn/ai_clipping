#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RANKER_SCRIPT="$SCRIPT_DIR/train_local_docker.sh"

PROFILE="${CLIP_PROFILE:-jynxzi}"
DATE_FROM=""
DATE_TO=""
DRY_RUN=0
SKIP_RANKER=0
RANKER_ARGS=()

show_help() {
  cat <<'EOF'
Usage:
  scripts/train_clipfinding_models_docker.sh --date-from DATE --date-to DATE [options]

Trains the clipfinding VOD ranker.

Required:
  --date-from DATE          Date range start. Forwarded to both scripts.
  --date-to DATE            Date range end. Forwarded to both scripts.

Options:
  --profile NAME            Profile/streamer. Default: CLIP_PROFILE, then jynxzi.
  --ranker-arg ARG          Extra single argument for train_local_docker.sh.
                            Repeat for multiple args and values.
  --dry-run                 Print the exact subcommands without running them.
  -h, --help                Show this help.

Examples:
  scripts/train_clipfinding_models_docker.sh --date-from 2026-03-01 --date-to 2026-05-23

  scripts/train_clipfinding_models_docker.sh \
    --date-from 2026-03-01 --date-to 2026-05-23 \
    --ranker-arg --training-vod-workers --ranker-arg 2
EOF
}

shell_join() {
  local out=""
  local arg
  for arg in "$@"; do
    printf -v quoted '%q' "$arg"
    out+="${out:+ }$quoted"
  done
  printf '%s\n' "$out"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --date-from)
      DATE_FROM="${2:?Missing value for --date-from}"
      shift 2
      ;;
    --date-from=*)
      DATE_FROM="${1#*=}"
      shift
      ;;
    --date-to)
      DATE_TO="${2:?Missing value for --date-to}"
      shift 2
      ;;
    --date-to=*)
      DATE_TO="${1#*=}"
      shift
      ;;
    --profile)
      PROFILE="${2:?Missing value for --profile}"
      shift 2
      ;;
    --profile=*)
      PROFILE="${1#*=}"
      shift
      ;;
    --ranker-arg)
      RANKER_ARGS+=("${2:?Missing value for --ranker-arg}")
      shift 2
      ;;
    --ranker-arg=*)
      RANKER_ARGS+=("${1#*=}")
      shift
      ;;
    --skip-ranker)
      SKIP_RANKER=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Use --ranker-arg '$1' to pass through extra ranker options." >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DATE_FROM" || -z "$DATE_TO" ]]; then
  echo "Missing required --date-from/--date-to." >&2
  show_help >&2
  exit 1
fi

if [[ "$SKIP_RANKER" == "1" ]]; then
  echo "Nothing to run: --skip-ranker was set." >&2
  exit 1
fi

if [[ ! -x "$RANKER_SCRIPT" ]]; then
  echo "Missing executable ranker wrapper: $RANKER_SCRIPT" >&2
  exit 1
fi

RANKER_COMMAND=(
  "$RANKER_SCRIPT"
  --date-from "$DATE_FROM"
  --date-to "$DATE_TO"
  --streamer "$PROFILE"
  "${RANKER_ARGS[@]}"
)

cd "$REPO_ROOT"

echo "{\"event\":\"clipfinding_model_training_plan\",\"profile\":\"$PROFILE\",\"date_from\":\"$DATE_FROM\",\"date_to\":\"$DATE_TO\",\"skip_ranker\":$SKIP_RANKER}"
if [[ "$SKIP_RANKER" != "1" ]]; then
  echo "{\"event\":\"ranker_training_command\",\"command\":\"$(shell_join "${RANKER_COMMAND[@]}")\"}"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if [[ "$SKIP_RANKER" != "1" ]]; then
  echo "{\"event\":\"ranker_training_start\"}"
  "${RANKER_COMMAND[@]}"
  echo "{\"event\":\"ranker_training_done\"}"
fi

echo "{\"event\":\"clipfinding_model_training_done\",\"profile\":\"$PROFILE\",\"date_from\":\"$DATE_FROM\",\"date_to\":\"$DATE_TO\"}"
