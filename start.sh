#!/bin/bash
set -e
cd "$(dirname "$0")"

# --- Helper: returns 0 (true) if marker is missing or trigger is newer ---
needs_update() {
  local trigger="$1" marker="$2"
  [ ! -f "$marker" ] || [ "$trigger" -nt "$marker" ]
}

# --- Local Node.js (like Python .venv, downloaded into .node/) ---
NODE_VERSION="20.19.0"
NODE_DIR=".node"
NODE_MARKER="$NODE_DIR/.version_marker"

ensure_local_node() {
  if [ -f "$NODE_MARKER" ] && [ "$(cat "$NODE_MARKER")" = "$NODE_VERSION" ]; then
    return
  fi
  echo "Installing local Node.js v${NODE_VERSION}..."
  local ARCH
  case "$(uname -m)" in
    x86_64)  ARCH="x64" ;;
    aarch64) ARCH="arm64" ;;
    armv7l)  ARCH="armv7l" ;;
    *)       echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
  esac
  local TARBALL="node-v${NODE_VERSION}-linux-${ARCH}.tar.xz"
  local URL="https://nodejs.org/dist/v${NODE_VERSION}/${TARBALL}"
  rm -rf "$NODE_DIR"
  mkdir -p "$NODE_DIR"
  curl -fsSL "$URL" | tar -xJ --strip-components=1 -C "$NODE_DIR"
  echo "$NODE_VERSION" > "$NODE_MARKER"
  echo "Node.js v${NODE_VERSION} installed in ${NODE_DIR}/"
}

ensure_local_node
export PATH="$(pwd)/$NODE_DIR/bin:$PATH"

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
