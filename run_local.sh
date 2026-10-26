#!/bin/bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec /usr/bin/sandbox-exec -f "$TASK_ROOT/local.sandbox.sb" "$TASK_ROOT/work/venv/bin/python" -I -B "$TASK_ROOT/local_run.py" "$@"
