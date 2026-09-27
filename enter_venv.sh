#!/usr/bin/env bash
# Usage: source enter_venv.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "Activated virtual environment: $VIRTUAL_ENV"
else
    echo "Error: Virtual environment not found at $SCRIPT_DIR/venv" >&2
fi
