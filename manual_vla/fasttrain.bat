@echo off
cd /d "%~dp0"
echo ===================================================
echo     FAST TRAIN PIPELINE: MANUAL VLA (DOBOT 4-DOF)
echo ===================================================

echo [1/2] Generating demonstration episodes (80 episodes)...
python auto_generate_demos.py 80

echo [2/2] Training Flow Matching Policy (Hadamard MLP, 300 epochs)...
python train_flow.py 300

echo ===================================================
echo  Training complete! Weights saved in models/
echo ===================================================
