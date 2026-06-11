#!/usr/bin/env bash
# GT Flywheel — daily capture job (issue #6 / PRD #3).
#
# Polls Twitch Helix for Jynxzi's new archive VODs, captures each not-yet-seen
# VOD (fingerprint-first → transcript → provenance, then deletes the media), and
# persists the already-captured state so re-runs are idempotent/resumable.
#
# Run / resume manually:
#   ./scripts/run_gt_capture_daily.sh
#   ./scripts/run_gt_capture_daily.sh --broadcaster jynxzi --json
#   ./scripts/run_gt_capture_daily.sh --limit 1            # capture at most one new VOD
#
# Schedule (daily, after Jynxzi's usual stream-end). Add to crontab -e — run at
# 06:00 local so a stream that ended late is still archived by Twitch:
#
#   0 6 * * *  cd /path/to/ai_clipping && ./scripts/run_gt_capture_daily.sh >> backend/tmp/gt_capture_daily.log 2>&1
#
# All Python logic lives in backend/ml/gt_capture_daily.py (mounted at /app/backend).
set -euo pipefail

CONTAINER="ai-clipping-backend-cuda"
WORKDIR="/app/backend"

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q "^true$"; then
    # Fall back to the CPU image if the CUDA box is not up.
    CONTAINER="ai-clipping-backend"
fi

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q "^true$"; then
    echo "ERROR: No backend container running (tried ai-clipping-backend-cuda, ai-clipping-backend)."
    echo "Start it first: docker compose -f docker-compose.cuda.yml up -d"
    exit 1
fi

exec docker exec -w "$WORKDIR" "$CONTAINER" python -m ml.gt_capture_daily "$@"
