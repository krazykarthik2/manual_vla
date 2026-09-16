@echo off
cd /d "%~dp0"
echo ===================================================
echo   ONLY RL: LONG-HORIZON REINFORCEMENT LEARNING
echo ===================================================
echo Training Cross-Attention Transformer directly in environment...
echo Runs until achieving high target success rate (>= 90%%).
echo Press Ctrl+C anytime to stop (checkpoints saved automatically).
echo ===================================================

python train_flow.py rl 0 90.0

echo ===================================================
echo  RL Training complete! Weights updated in models/
echo ===================================================
