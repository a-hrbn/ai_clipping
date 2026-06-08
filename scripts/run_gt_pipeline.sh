#!/usr/bin/env bash
# Ground-Truth Pipeline — thin wrapper that runs inside the Docker container.
#
# Edit backend/ml/jynxzi_gt_vods.csv to add VODs, then run:
#
#   ./scripts/run_gt_pipeline.sh                   # run / resume
#   ./scripts/run_gt_pipeline.sh --reset           # start from scratch
#   ./scripts/run_gt_pipeline.sh --help            # show all options
#   ./scripts/run_gt_pipeline.sh --vods ml/other.csv
#
# All Python logic lives in backend/ml/gt_pipeline.py (mounted at /app/backend).
# Logs: backend/tmp/gt_runs/<run_id>/pipeline.log
set -euo pipefail

CONTAINER="ai-clipping-backend"
WORKDIR="/app/backend"

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q "^true$"; then
    echo "ERROR: Container '$CONTAINER' is not running."
    echo "Start it first: docker compose up -d"
    exit 1
fi

exec docker exec -w "$WORKDIR" "$CONTAINER" python ml/gt_pipeline.py "$@"
