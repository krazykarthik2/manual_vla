@echo off
cd /d "%~dp0"
echo =========================================================
echo   SMOLVLA LIVE VISUALIZER — ATTENTION + MODEL EXECUTION
echo =========================================================
echo Controls:
echo   [UP / DOWN]  Select token in prompt (scrollable)
echo   [Scroll]     Scroll token list
echo   [F]          Toggle speed: Normal / Fast (3x) / Slow (0.5x)
echo   [R]          Randomize table scene and objects
echo   [1]          Pick and Place task prompt
echo   [2]          Push Towards task prompt
echo =========================================================
python visualize_smolvla.py
