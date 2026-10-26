#!/bin/bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
: "${OBSERVER_RUNTIME_ROOT:?observer parent environment required}"
: "${OBSERVER_OLD_ROOT:?observer parent environment required}"
: "${OBSERVER_REAL_PYTHON:?observer parent environment required}"
exec "$OBSERVER_REAL_PYTHON" -I -B "$TOOL_DIR/observer_run.py" \
  --root "$OBSERVER_RUNTIME_ROOT" --old-root "$OBSERVER_OLD_ROOT" --child "$@"
