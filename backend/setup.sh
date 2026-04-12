#!/usr/bin/env bash
set -e

echo "=== AI Clipping Backend Setup ==="

# Create virtualenv if not exists
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "Virtualenv erstellt."
fi

source .venv/bin/activate
pip install -r requirements.txt

echo ""
echo "=== Setup abgeschlossen ==="
echo ""
echo "Backend starten:"
echo "  source .venv/bin/activate && uvicorn main:app --reload --port 8000"
echo ""
echo "Stell sicher, dass Ollama läuft und llama3.2 geladen ist:"
echo "  ollama serve"
echo "  ollama pull llama3.2"
