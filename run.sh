#!/bin/bash
cd "$(dirname "$0")"

# Source nvm if available (needed for ELK layout subprocess)
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

source .venv/bin/activate

# Extract --ov-path value to set LD_LIBRARY_PATH (needed for libopenvino*.so)
prev=""
for i in "$@"; do
  if [ "$prev" = "--ov-path" ]; then
    export LD_LIBRARY_PATH="${i}:${LD_LIBRARY_PATH:-}"
    break
  fi
  prev="$i"
done

exec python -m backend.main "$@"
