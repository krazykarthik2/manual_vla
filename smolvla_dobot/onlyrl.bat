@echo off
cd /d "%~dp0"
echo =========================================================
echo   SMOLVLA ONLY RL: REINFORCEMENT LEARNING DIRECTLY IN SIM
echo =========================================================
echo Freezes Vision & Language Backbone.
echo Trains ONLY the Cross-Attention Flow-Matching Action Head.
echo Uses Demo-Anchored Replay to prevent catastrophic drift.
echo Runs until target success rate (>= 90%%) is achieved.
echo Press Ctrl+C anytime to stop (checkpoints saved automatically).
echo =========================================================

python train_smolvla_rl.py 0 90.0

echo =========================================================
echo  SmolVLA RL Training complete! Weights updated in models/
echo =========================================================
