# AI Clipping

Full-stack app for generating short-form clips from long-form video. The repo contains:

- `backend/`: FastAPI pipeline, clip generation services, and ML/training utilities
- `frontend/`: Next.js UI for submitting jobs and reviewing clips

## Repo Setup

1. Clone the repository on a device.
2. Create local env files from the examples:
   - `cp backend/.env.example backend/.env`
   - `cp frontend/.env.example frontend/.env.local`
3. Install dependencies:
   - Backend: create a virtualenv and install `backend/requirements.txt`
   - Frontend: `cd frontend && npm install`
4. Start the services:
   - Backend: run the FastAPI app from `backend/`
   - Frontend: run `npm run dev` from `frontend/`

## GitHub Notes

- Local secrets and generated artifacts are ignored by the root `.gitignore`.
- The old nested frontend git metadata was moved to `frontend/.git.local-backup/` so this project can be tracked as one repository.
- Large local-only paths like virtualenvs, `node_modules`, ML artifacts, transcripts, and graphify output are excluded from version control.

## Suggested First Push

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
