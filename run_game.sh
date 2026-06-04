#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python is required but was not found on PATH." >&2
  exit 1
fi

echo "Running backend smoke test..."
"$PYTHON_BIN" scripts/run_smoke_test.py

echo "Generating replay data..."
"$PYTHON_BIN" scripts/generate_replay.py --trace-level full

if ! command -v npm >/dev/null 2>&1; then
  echo "npm was not found on PATH."
  echo "Replay JSON is ready at visualizer/public/match_log.json."
  exit 0
fi

cd visualizer
if [ ! -d node_modules ]; then
  echo "Installing visualizer dependencies..."
  npm install
fi

echo "Starting visualizer at http://localhost:5173"
npm run dev -- --host 127.0.0.1
