#!/bin/bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
OBSERVER_ROOT="${OBSERVER_RUNTIME_ROOT:-$TOOL_DIR}"
OBSERVER_ROOT="$(cd -- "$OBSERVER_ROOT" && pwd -P)"
OBSERVER_PYTHON="${OBSERVER_PYTHON:-/Users/shiweili/code/monthly-stock-research/work/venv/bin/python}"
OBSERVER_OLD="${OBSERVER_OLD_ROOT:-/Users/shiweili/Library/CloudStorage/OneDrive-RokuInc/Documents/Codex/2026-09-18/5-grand-hyatt-fukuoka-5-14}"
if [[ $# -eq 0 || "$1" == --* ]]; then
  set -- outputs/paired_stock_monitor/run_pair.py "$@"
fi
exec /usr/bin/sandbox-exec -D "OBSERVER_ROOT=$OBSERVER_ROOT" \
  -f "$TOOL_DIR/observer.sandbox.sb" "$OBSERVER_PYTHON" -I -B \
  "$TOOL_DIR/observer_run.py" --root "$OBSERVER_ROOT" --old-root "$OBSERVER_OLD" "$@"
