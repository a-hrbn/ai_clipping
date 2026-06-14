#!/usr/bin/env bash
# Ralph — autonomous TDD over the agent-ready issue backlog.
#
# Walks the open, AGENT-READY issues in dependency order and, for each one,
# spawns a fresh unattended `claude` client running the /tdd skill. "Ralph" =
# re-prompt a fresh agent in a loop until the work is done; here the loop is
# bounded per issue and the done-signal is the issue being CLOSED on GitHub
# (which /tdd does only when truly green + committed).
#
# Design (see git log / the grill session that produced this):
#   - Selection : open + `ready-for-agent`, PRDs excluded, ascending by number.
#   - Order     : sequential, shared working tree (slices are dependency-chained;
#                 #77 builds on #76's committed code).
#   - Branch    : one run-branch `ralph/afk-<ts>` off main. Commits land there.
#                 The loop NEVER pushes or opens a PR — a human reviews first.
#   - Retry     : up to RALPH_MAX_ATTEMPTS fresh clients per issue; stop when the
#                 issue goes CLOSED.
#   - Failure   : fail-fast. If an issue exhausts its attempts still open, comment
#                 on it and abort the whole run (downstream slices depend on it).
#   - Autonomy  : --dangerously-skip-permissions + an appended system prompt that
#                 neutralizes /tdd's interactive approval gates.
#   - Tests     : run inside the docker backend container (ML deps live there).
#
# Usage:
#   ./scripts/ralph_tdd.sh                 # auto-select the agent-ready backlog
#   ./scripts/ralph_tdd.sh 76 77           # process exactly these, in this order
#   ./scripts/ralph_tdd.sh --dry-run       # print the resolved plan, spawn nothing
#   ./scripts/ralph_tdd.sh --yes           # skip the confirmation prompt (cron)
#
# Env knobs (defaults in brackets):
#   RALPH_MAX_ATTEMPTS    [3]               retries per issue before giving up
#   RALPH_ATTEMPT_TIMEOUT [40m]             wall-clock cap per claude attempt
#   RALPH_MODEL           [opus]            --model
#   RALPH_EFFORT          [xhigh]           --effort
#   RALPH_LABEL           [ready-for-agent] selection label
#   RALPH_YES             [0]               1 = skip confirmation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib_backend_container.sh
source "$SCRIPT_DIR/lib_backend_container.sh"

MAX_ATTEMPTS="${RALPH_MAX_ATTEMPTS:-3}"
ATTEMPT_TIMEOUT="${RALPH_ATTEMPT_TIMEOUT:-40m}"
MODEL="${RALPH_MODEL:-opus}"
EFFORT="${RALPH_EFFORT:-xhigh}"
LABEL="${RALPH_LABEL:-ready-for-agent}"
ASSUME_YES="${RALPH_YES:-0}"
DRY_RUN=0

# --- arg parsing -------------------------------------------------------------
EXPLICIT_ISSUES=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --yes|-y)  ASSUME_YES=1 ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    [0-9]*)    EXPLICIT_ISSUES+=("$arg") ;;
    *) echo "ralph: unknown argument '$arg' (expected issue numbers, --dry-run, --yes)" >&2; exit 2 ;;
  esac
done

die() { echo "ralph: $*" >&2; exit 1; }

# --- preflight: clean tree, on main, tools present, container up -------------
command -v claude >/dev/null 2>&1 || die "claude CLI not found on PATH."
command -v gh     >/dev/null 2>&1 || die "gh CLI not found on PATH."
command -v timeout >/dev/null 2>&1 || die "coreutils 'timeout' not found on PATH."

cd "$ROOT_DIR"

[ -z "$(git status --porcelain)" ] || die "working tree is dirty — commit or stash first (Ralph needs a clean base so the agent's commits contain only its own work)."

# Resolve the running backend container (cuda preferred, cpu fallback) — its name
# is baked into the autonomy prompt so every attempt runs tests the same way.
CONTAINER="${BACKEND_CONTAINER_CUDA:-ai-clipping-backend-cuda}"
container_running "$CONTAINER" || CONTAINER="${BACKEND_CONTAINER:-ai-clipping-backend}"
if ! container_running "$CONTAINER"; then
  echo "ralph: no backend container running (tried cuda, cpu)." >&2
  echo "       start it first: docker compose -f docker-compose.cuda.yml up -d" >&2
  exit 1
fi

# --- resolve the ordered issue list -----------------------------------------
declare -A TITLE
ISSUES=()
if [ "${#EXPLICIT_ISSUES[@]}" -gt 0 ]; then
  for n in "${EXPLICIT_ISSUES[@]}"; do
    title="$(gh issue view "$n" --json title --jq .title 2>/dev/null)" \
      || die "could not read issue #$n (not found, or gh not authenticated)."
    ISSUES+=("$n"); TITLE[$n]="$title"
  done
else
  # Open + labelled, PRDs (title starts 'PRD:') excluded, ascending by number.
  while IFS=$'\t' read -r n title; do
    [ -n "$n" ] || continue
    ISSUES+=("$n"); TITLE[$n]="$title"
  done < <(gh issue list --state open --label "$LABEL" --limit 200 \
            --json number,title \
            --jq '[.[] | select(.title | startswith("PRD:") | not)]
                  | sort_by(.number) | .[] | "\(.number)\t\(.title)"')
fi

[ "${#ISSUES[@]}" -gt 0 ] || die "no matching issues (label '$LABEL', PRDs excluded). Nothing to do."

# --- run metadata ------------------------------------------------------------
RUN_TS="$(date +%Y%m%d-%H%M%S)"
RUN_BRANCH="ralph/afk-$RUN_TS"
LOG_DIR="$ROOT_DIR/backend/tmp/ralph/$RUN_TS"
mkdir -p "$LOG_DIR"

# --- the autonomy system prompt (neutralizes /tdd's interactive gates) -------
read -r -d '' AUTONOMY_PROMPT <<EOF || true
You are running UNATTENDED in a headless Ralph loop. No human will answer you.

- Every /tdd step that says "confirm with the user", "get user approval", or
  "ask the user" is PRE-APPROVED. Never wait for approval; decide and proceed.
- Read docs/agents/issue-tracker.md, then \`gh issue view #N --comments\` for the
  issue you were given. Treat the issue body as the approved spec. Read the
  comments for context from any prior attempt on this same issue.
- Make reasonable interface and behavior decisions yourself.
- Run tests ONLY inside the backend container, like this:
      $DOCKER exec -w /app/backend $CONTAINER python -m pytest tests/ -q
  (narrow to the relevant test file/node while iterating). Use exactly that
  docker binary ('$DOCKER' — bare 'docker' is disabled in this WSL distro). Do
  NOT create a host venv or run pytest on the host.
- Do real vertical-slice TDD: one test -> minimal code -> repeat. No horizontal
  slicing (do not write all tests first).
- Commit your work as you go. Commit ONLY — do NOT push and do NOT open a PR;
  the surrounding script owns git remotes and a human reviews before pushing.
- Close the issue and cascade to the parent PRD per /tdd's "Close the issue"
  step ONLY when the work is truly done: all tests green, refactor done, change
  committed.
- If you get blocked (missing context, tests you cannot make pass, an external
  dependency), leave a comment on the issue explaining why and STOP. Do NOT
  close an unfinished issue.
EOF

# --- plan + confirmation -----------------------------------------------------
echo "=============================================================="
echo " Ralph TDD run"
echo "   branch        : $RUN_BRANCH  (off $(git branch --show-current))"
echo "   container     : $CONTAINER   (via $DOCKER)"
echo "   model/effort  : $MODEL / $EFFORT"
echo "   max attempts  : $MAX_ATTEMPTS   timeout/attempt: $ATTEMPT_TIMEOUT"
echo "   logs          : $LOG_DIR"
echo "   issues (in order):"
for n in "${ISSUES[@]}"; do printf "     #%-5s %s\n" "$n" "${TITLE[$n]}"; done
echo "=============================================================="

if [ "$DRY_RUN" -eq 1 ]; then
  echo "ralph: --dry-run, nothing spawned."
  exit 0
fi

if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "Proceed? This runs claude --dangerously-skip-permissions over the above. [y/N] " ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "ralph: aborted."; exit 1 ;; esac
fi

# --- create the run-branch ---------------------------------------------------
git switch -c "$RUN_BRANCH" >/dev/null
echo "ralph: created $RUN_BRANCH"

# --- the loop ----------------------------------------------------------------
issue_state() { gh issue view "$1" --json state --jq .state 2>/dev/null || echo "UNKNOWN"; }

declare -A RESULT ATTEMPTS DURATION
overall_rc=0

for n in "${ISSUES[@]}"; do
  echo
  echo ">>> issue #$n — ${TITLE[$n]}"

  if [ "$(issue_state "$n")" = "CLOSED" ]; then
    echo "    already CLOSED, skipping."
    RESULT[$n]="skipped-closed"; ATTEMPTS[$n]=0; DURATION[$n]=0
    continue
  fi

  closed=0
  attempt=0
  issue_start=$(date +%s)
  while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    log="$LOG_DIR/issue-$n-attempt-$attempt.log"
    echo "    attempt $attempt/$MAX_ATTEMPTS  ->  $log"

    # Fresh client per attempt. We ignore claude's exit code for control flow —
    # the GitHub issue state is the source of truth — but we record it.
    set +e
    timeout "$ATTEMPT_TIMEOUT" \
      claude -p "/tdd Implement GitHub issue #$n end-to-end, unattended. Close it when green." \
        --model "$MODEL" \
        --effort "$EFFORT" \
        --dangerously-skip-permissions \
        --append-system-prompt "$AUTONOMY_PROMPT" \
      2>&1 | tee "$log"
    claude_rc="${PIPESTATUS[0]}"
    set -e
    [ "$claude_rc" -eq 124 ] && echo "    (claude hit the ${ATTEMPT_TIMEOUT} timeout)"

    if [ "$(issue_state "$n")" = "CLOSED" ]; then
      echo "    issue #$n CLOSED after attempt $attempt."
      closed=1
      break
    fi
    echo "    issue #$n still open after attempt $attempt (claude rc=$claude_rc)."
  done

  DURATION[$n]=$(( $(date +%s) - issue_start ))
  ATTEMPTS[$n]=$attempt

  if [ "$closed" -eq 1 ]; then
    RESULT[$n]="closed"
  else
    RESULT[$n]="FAILED"
    overall_rc=1
    echo "    !! issue #$n exhausted $MAX_ATTEMPTS attempts, still open."
    gh issue comment "$n" --body \
"Ralph (\`scripts/ralph_tdd.sh\`): exhausted $MAX_ATTEMPTS unattended /tdd attempts on branch \`$RUN_BRANCH\` without reaching a green, closeable state. Left open for a human. Per-attempt logs: \`$LOG_DIR\`." \
      || echo "    (warning: could not post failure comment to #$n)"
    echo "    fail-fast: downstream issues depend on this one — aborting the run."
    break
  fi
done

# --- summary -----------------------------------------------------------------
echo
echo "=============================================================="
echo " Ralph TDD summary  (branch $RUN_BRANCH)"
printf " %-7s %-14s %-9s %-8s\n" "ISSUE" "RESULT" "ATTEMPTS" "DUR(s)"
for n in "${ISSUES[@]}"; do
  [ -n "${RESULT[$n]:-}" ] || { printf " #%-6s %-14s\n" "$n" "not-reached"; continue; }
  printf " #%-6s %-14s %-9s %-8s\n" "$n" "${RESULT[$n]}" "${ATTEMPTS[$n]}" "${DURATION[$n]}"
done
echo " logs: $LOG_DIR"
echo " review: git log $RUN_BRANCH   (nothing was pushed)"
echo "=============================================================="

exit "$overall_rc"
