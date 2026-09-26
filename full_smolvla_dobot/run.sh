#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/venv/bin/activate"
fi

echo "========================================================="
echo "      FULL SMOLVLA DOBOT CONTROLLER (SMOLVLM BACKBONE)"
echo "========================================================="
echo "Controls:"
echo "  [F]     Toggle Lightspeed Mode On/Off"
echo "  [1]     Action: Pick and Place"
echo "  [R]     Randomize Table Clutter"
echo "  [SPACE] Pause / Resume Execution"
echo "========================================================="

python3 "$SCRIPT_DIR/run_full_smolvla.py" "$@"
