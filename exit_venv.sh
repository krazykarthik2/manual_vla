#!/usr/bin/env bash
# Usage: source exit_venv.sh

if type deactivate >/dev/null 2>&1; then
    deactivate
    echo "Deactivated virtual environment."
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating $VIRTUAL_ENV..."
    PATH="${PATH#$VIRTUAL_ENV/bin:}"
    unset VIRTUAL_ENV
    unset -f deactivate 2>/dev/null || true
    echo "Deactivated virtual environment."
else
    echo "No virtual environment is currently active."
fi
