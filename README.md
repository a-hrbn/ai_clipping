# AI Clipping

Full-stack app for generating short-form clips from long-form video. The repo contains:

- `backend/`: FastAPI pipeline, clip generation services, and ML/training utilities
- `frontend/`: Next.js UI for submitting jobs and reviewing clips
- `Dockerfile` + `docker-compose.yml`: Linux development container with SSH, backend, frontend, graphify, and Codex preinstalled

## Development Setup

This repo is intended to run through Docker Compose. The host machine does not
need project-local Python virtualenvs or a host-side frontend install.

1. Clone the repository on a device.
2. Create local env files from the examples:
   - `copy backend\.env.example backend\.env`
   - `copy frontend\.env.example frontend\.env.local`
3. Add your SSH public key:
   - copy `docker/ssh/authorized_keys.example` to `docker/ssh/authorized_keys`
   - replace the placeholder line with your real public key
4. Start the container:
   - `docker compose up --build`

When the container starts it will automatically:

- install backend Python dependencies into `/opt/venv`
- install frontend dependencies into a container-managed `node_modules` volume
- install `graphifyy` and run `graphify install`
- install the OpenAI Codex CLI
- start SSH on port `2222`
- start FastAPI on port `8000`
- start Next.js on port `3000`

The entire repository is bind-mounted into `/workspace`, so every file in the
project is visible inside the Linux container.

## Backend Requirements Layout

The backend dependencies are split into layered files:

- `backend/requirements-base.txt`: web app and runtime essentials
- `backend/requirements-ml.txt`: heavy ML/audio/vision dependencies
- `backend/requirements-dev.txt`: developer tooling such as `pytest` and `graphifyy`
- `backend/requirements.txt`: aggregate file that includes all three

The Docker build currently installs all three layers so the full app remains
available inside the container.

## SSH Development

Connect to the running container with:

```bash
ssh dev@localhost -p 2222
```

The project root inside the container is:

```bash
/workspace
```

Useful in-container commands:

```bash
cd /workspace
python -m pytest backend/tests -q
python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"
codex --login
```

## Services

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`

## GitHub Notes

- Local secrets and generated artifacts are ignored by the root `.gitignore`.
- The old nested frontend git metadata was moved to `frontend/.git.local-backup/` so this project can be tracked as one repository.
- Large local-only paths like temporary virtualenvs, `node_modules`, ML artifacts, transcripts, runtime temp files, SSH keys, and graphify output are excluded from version control.

## Suggested First Push

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
