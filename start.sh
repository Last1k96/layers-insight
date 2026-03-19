#!/bin/bash
set -e
cd "$(dirname "$0")"

# --- Helper: returns 0 (true) if marker is missing or trigger is newer ---
needs_update() {
  local trigger="$1" marker="$2"
  [ ! -f "$marker" ] || [ "$trigger" -nt "$marker" ]
}

# --- Source nvm (needed for ELK layout subprocess + npm) ---
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# --- 1. Python venv ---
if [ ! -d .venv ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate

# --- 2. Python deps ---
if needs_update pyproject.toml .venv/.deps_marker; then
  echo "Installing Python dependencies..."
  pip install -e ".[dev]" --quiet
  touch .venv/.deps_marker
fi

# --- 3. Root node_modules (elkjs) ---
if needs_update package.json node_modules/.deps_marker; then
  echo "Installing root Node dependencies (elkjs)..."
  npm ci --quiet 2>/dev/null || npm install --quiet
  touch node_modules/.deps_marker
fi

# --- 4. Frontend node_modules ---
if needs_update frontend/package.json frontend/node_modules/.deps_marker; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm ci --quiet 2>/dev/null || npm install --quiet)
  touch frontend/node_modules/.deps_marker
fi

# --- 5. Frontend build ---
BUILD_MARKER="frontend/dist/.build_marker"
if [ ! -f "$BUILD_MARKER" ] || [ -n "$(find frontend/src -newer "$BUILD_MARKER" -print -quit 2>/dev/null)" ]; then
  echo "Building frontend..."
  (cd frontend && npm run build)
  touch "$BUILD_MARKER"
fi

# --- 6. Extract --ov-path for LD_LIBRARY_PATH ---
prev=""
for i in "$@"; do
  if [ "$prev" = "--ov-path" ]; then
    export LD_LIBRARY_PATH="${i}:${LD_LIBRARY_PATH:-}"
    break
  fi
  prev="$i"
done

# --- 7. Launch ---
exec python -m backend.main "$@"
