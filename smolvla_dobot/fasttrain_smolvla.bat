@echo off
cd /d "%~dp0"
echo =========================================================
echo   SMOLVLA FASTTRAIN: ViT + PRETRAINED CROSS-ATTN + RL
echo =========================================================

echo [1/3] Generating adaptive velocity demos (80 episodes)...
python auto_generate_demos.py 80

echo [2/3] Behavioral Cloning Pre-Training (50 epochs)...
python train_smolvla.py 50

echo [3/3] Demo-Anchored RL Fine-Tuning (Cross-Attention Action Head only) (150 episodes)...
python train_smolvla_rl.py 150

echo =========================================================
echo  SmolVLA Pipeline Complete! Launching run_smolvla.bat...
echo =========================================================
call run_smolvla.bat
