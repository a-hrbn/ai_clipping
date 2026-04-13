#!/usr/bin/env bash
set -e

cat <<'EOF'
This repository now uses Docker Compose for development instead of a project-
local backend virtualenv.

Start the full development stack from the repo root:

  docker compose up --build

The Linux container installs backend dependencies, frontend dependencies, and
graphify, then starts:
  - sshd on port 2222
  - FastAPI on port 8000
  - Next.js on port 3000

Inside the container, backend commands run directly with the image's global
Python environment.
EOF
