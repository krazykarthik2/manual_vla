@echo off
cd /d "%~dp0"
echo ===================================================
echo       SMOLVLA DOBOT CONTROLLER (LIGHTSPEED)
echo ===================================================
echo Controls:
echo   [F]     Toggle Lightspeed Mode On/Off
echo   [1]     Action: Pick and Place
echo   [2]     Action: Push Towards
echo   [R]     Randomize Table Clutter
echo   [SPACE] Pause / Resume Execution
echo ===================================================
python run_smolvla.py --fast %*
