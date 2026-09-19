@echo off
cd /d "%~dp0"
echo =========================================================
echo       FULL SMOLVLA DOBOT CONTROLLER (SMOLVLM BACKBONE)
echo =========================================================
echo Controls:
echo   [F]     Toggle Lightspeed Mode On/Off
echo   [1]     Action: Pick and Place
echo   [R]     Randomize Table Clutter
echo   [SPACE] Pause / Resume Execution
echo =========================================================
python run_full_smolvla.py --fast %*
