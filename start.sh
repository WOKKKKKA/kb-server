#!/usr/bin/env bash
# KB Server — one-command launcher (Linux / NAS / macOS)
# Usage: ./start.sh
set -e

PYTHON="${PYTHON:-python3}"

# --- check python ---
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "[KB] $PYTHON not found. Install Python 3.10+ first."
    exit 1
fi

# --- virtualenv ---
if [ ! -d ".venv" ]; then
    echo "[KB] creating virtualenv..."
    "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- dependencies ---
echo "[KB] installing dependencies..."
pip install -q -r requirements.txt

# --- run ---
echo "[KB] starting KB Server on http://0.0.0.0:8080"
exec python kb_server.py
