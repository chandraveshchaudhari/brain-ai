#!/bin/bash
# Brain-AI verification gate. The Supervisor runs this inside each task's git
# worktree; a task is done only when it exits 0. Run it locally the same way:
#   bash scripts/test.sh
set -u
cd "$(dirname "$0")/.."

# Git-ignored virtualenvs exist only in the main checkout, not in task
# worktrees, so fall back to the main checkout's environment. PYTHONPATH below
# makes the worktree's own source win over the environment's editable install.
find_python() {
    for candidate in "$PWD/.venv/bin/python" "/Volumes/MacSSD/01_Projects/SolarpunkWorks/projects/brain-ai/.venv/bin/python"; do
        if [ -x "$candidate" ] && "$candidate" -c 'import pytest' >/dev/null 2>&1; then
            echo "$candidate"; return 0
        fi
    done
    return 1
}
PY=$(find_python) || { echo "gate: no virtualenv with pytest; create it: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt -e ." >&2; exit 2; }
PYTHONPATH="$PWD/src" exec "$PY" -m pytest -q tests
