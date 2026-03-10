#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Setting up layers-insight ==="

# Source nvm if available (for WSL environments)
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# Python venv
if [ ! -d .venv ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "Installing Python dependencies..."
pip install -e ".[dev]" --quiet

# Root Node deps (elkjs)
echo "Installing root Node dependencies (elkjs)..."
npm ci --quiet 2>/dev/null || npm install --quiet

# Frontend
echo "Installing frontend dependencies..."
cd frontend && npm ci --quiet 2>/dev/null || npm install --quiet
echo "Building frontend..."
npm run build
cd ..

echo ""
echo "=== Setup complete ==="
echo "Run: ./run.sh [options]"
echo "Example: ./run.sh --model /path/to/model.xml --main-device GPU --ref-device CPU"
