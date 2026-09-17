@echo off
cd /d "%~dp0"
echo =========================================================
echo   FAST TRAIN PIPELINE: SMOLVLA (ViT + CROSS-ATTENTION)
echo   [100%% PURE PICK & PLACE - GENERATE + TRAIN + RUN]
echo =========================================================

echo [1/2] Generating adaptive velocity pick-and-place demos (80 episodes)...
python auto_generate_demos.py 80

echo [2/2] Training SmolVLA Policy (ViT + Cross-Attention) (200 epochs)...
python train_smolvla.py 200

echo =========================================================
echo  SmolVLA Training Complete! Launching run_smolvla.bat...
echo =========================================================
call run_smolvla.bat
