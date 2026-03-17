#!/usr/bin/env bash
set -e

# Simple dev runner: starts backend and frontend together.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR/backend"

if [ -f ".env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

echo "Starting backend on http://localhost:8000 ..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

cd "$ROOT_DIR/frontend"
echo "Starting frontend on http://localhost:5173 ..."
npm run dev

echo "Stopping backend..."
kill "$BACKEND_PID" || true

