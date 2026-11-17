#!/usr/bin/env bash
# Installation (Linux / macOS / Git Bash) : crée .venv et installe tout.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
command -v "$PY" >/dev/null || { echo "Python 3.10+ introuvable"; exit 1; }

[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name qaoa-ml --display-name "Python (qaoa-ml)"

echo
echo "Installation terminée. Activez l'environnement avec : source .venv/bin/activate"
