# AI Clipping

Full-stack app for generating short-form clips from long-form video.

- `backend/`: FastAPI pipeline, clip generation services, and ML/training utilities
- `frontend/`: Next.js UI for submitting jobs and reviewing clips
- `graphify-out/`: generated knowledge-graph output for repo navigation

## Local Setup

1. Create local env files from the examples:
   - `cp backend/.env.example backend/.env`
   - `cp frontend/.env.example frontend/.env.local`
2. Create a backend virtual environment:
   - `python3 -m venv backend/.venv`
   - `source backend/.venv/bin/activate`
3. Install backend dependencies:
   - base runtime: `python -m pip install -e ./backend`
   - runtime + developer tools: `python -m pip install -e './backend[dev]'`
   - full local stack with ML extras: `python -m pip install -e './backend[dev,ml]'`
4. Install frontend dependencies:
   - `cd frontend && npm install`

## Running The App

Backend:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

## Dependency Layout

Backend dependencies now live in `backend/pyproject.toml`.

- base dependencies: `[project.dependencies]`
- developer tooling: `[project.optional-dependencies].dev`
- ML/audio/vision stack: `[project.optional-dependencies].ml`

## Useful Commands

```bash
python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"
```

## Services

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`

## GitHub Notes

- Local secrets and generated artifacts are ignored by the root `.gitignore`.
- The old nested frontend git metadata was moved to `frontend/.git.local-backup/` so this project can be tracked as one repository.
- Large local-only paths like temporary virtualenvs, `node_modules`, ML artifacts, transcripts, and graphify output are excluded from version control.
