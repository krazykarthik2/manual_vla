@echo off
cd /d "%~dp0"
echo =========================================================
echo   SMOLVLA PAIRED LANGUAGE TOKEN - IMAGE PATCH VIEWER
echo =========================================================
echo Controls:
echo   [UP / DOWN] Select word token in prompt
echo   [R]         Randomize table scene and objects
echo   [1]         Pick and Place task prompt
echo   [2]         Push Towards task prompt
echo =========================================================
python visualize_smolvla.py
