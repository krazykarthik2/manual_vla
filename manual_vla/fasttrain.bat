@echo off
cd /d "%~dp0"
echo ===================================================
echo     FAST TRAIN PIPELINE: MANUAL VLA (DOBOT 4-DOF)
echo ===================================================

echo [1/3] Generating demonstration episodes...
python auto_generate_demos.py 40

echo [2/3] Training Flow Matching Policy (Hadamard MLP)...
python train_flow.py

echo [3/3] Cleaning up temporary demonstration dataset...
if exist "data\demos" (
    del /q "data\demos\*.npz" 2>nul
    echo Demonstration cache cleared successfully.
)

echo ===================================================
echo  Training complete! Weights saved in models/
echo ===================================================
