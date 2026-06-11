#!/usr/bin/env bash
# GT Flywheel — weekly match+verify job (issue #8 / PRD #3).
#
# Re-scans @Iboclipss for new shorts (reusing the yt-dlp cookie-auth + request
# pacing so bot-protection can't silently zero out discovery), matches only the
# new reposts against every VOD captured by the daily job, auto-mints
# gold/silver/review with the confirm-margin rule, rebuilds the gold-filtered
# eval manifest end-to-end, and prints a gold-accrual summary vs the target.
#
# Run / resume manually:
#   ./scripts/run_gt_match_verify_weekly.sh
#   ./scripts/run_gt_match_verify_weekly.sh --json
#   ./scripts/run_gt_match_verify_weekly.sh --limit 5     # match at most 5 new reposts
#
# Schedule (weekly — reposts lag the VOD by days, so weekly is plenty). Add to
# crontab -e; runs Mondays at 07:00 local, after the daily capture has banked the
# week's VODs:
#
#   0 7 * * 1  cd /path/to/ai_clipping && ./scripts/run_gt_match_verify_weekly.sh >> backend/tmp/gt_match_verify_weekly.log 2>&1
#
# All Python logic lives in backend/ml/gt_match_verify_weekly.py (mounted at /app/backend).
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

exec docker exec -w "$WORKDIR" "$CONTAINER" python -m ml.gt_match_verify_weekly "$@"
