@echo off
cd /d "%~dp0"
echo =========================================================
echo   FAST TRAIN PIPELINE: SMOLVLA (ViT + MULTIMODAL ATTN)
echo =========================================================

echo [1/2] Generating adaptive velocity demos (80 episodes)...
python auto_generate_demos.py 80

echo [2/2] Training SmolVLA Policy (ViT + Cross-Attention) (200 epochs)...
python train_smolvla.py 200

echo =========================================================
echo  SmolVLA Training complete! Run run_smolvla.bat to evaluate.
echo =========================================================
