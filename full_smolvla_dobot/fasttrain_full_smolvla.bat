@echo off
cd /d "%~dp0"
echo =========================================================
echo   FULL SMOLVLA PIPELINE (REAL SMOLVLM BACKBONE)
echo   - Pretrained Hugging Face SmolVLM-256M-Instruct
echo   - Auto-generate demos + Feature Caching + Flow Train
echo =========================================================

echo [1/2] Generating adaptive velocity pick-and-place demos (80 episodes)...
python auto_generate_demos.py 80

echo [2/2] Training Full SmolVLA Policy (200 epochs)...
python train_full_smolvla.py 200

echo =========================================================
echo  Full SmolVLA Training Complete! Launching run_full_smolvla.bat...
echo =========================================================
call run_full_smolvla.bat
