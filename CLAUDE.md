## graphify

This project has a graphify knowledge graph at backend/graphify-out/.

Rules:
- Before answering architecture or codebase questions, read backend/graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If backend/graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying backend code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('backend'))"` to keep the graph current

## Agent skills

### Issue tracker

Issues & PRDs live in GitHub Issues (`a-hrbn/ai_clipping`), via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical label vocabulary (`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`CONTEXT.md` + `docs/adr/` at the repo root). See `docs/agents/domain.md`.