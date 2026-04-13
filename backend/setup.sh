#!/usr/bin/env bash
set -e

cat <<'EOF'
Local backend setup:

  python3 -m venv backend/.venv
  source backend/.venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e './backend[dev,ml]'

Run the API:

  cd backend
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
EOF
