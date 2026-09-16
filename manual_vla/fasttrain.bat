@echo off
cd /d "%~dp0"
echo ===================================================
echo     FAST TRAIN PIPELINE: MANUAL VLA (DOBOT 4-DOF)
echo ===================================================

echo [1/3] Generating demonstration episodes (80 episodes)...
python auto_generate_demos.py 80

echo [2/3] Pre-training Flow Matching Policy with Cross-Attention Transformer (200 epochs)...
python train_flow.py 200

echo [3/3] Demo-Anchored RL Fine-Tuning in Environment (150 episodes)...
python train_flow.py rl 150

echo ===================================================
echo  Training complete! Weights saved in models/
echo ===================================================
