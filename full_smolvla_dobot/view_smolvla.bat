@echo off
cd /d "%~dp0"
echo =========================================================
echo   FULL SMOLVLA LIVE MULTIMODAL ACTIVATIONS VISUALIZER
echo   - Genuine SmolVLM-256M Multimodal Embeddings
echo   - Layer 1 & 2 Cross-Attention Maps
echo   - Layer-Wise Transformer Decoder Activations & Spectrogram
echo =========================================================
echo Controls:
echo   [F]     Toggle Speed: Normal (60fps) / Fast (3x) / Slow (0.5x)
echo   [TAB]   Cycle Layer Activations (L1 SA, L1 CA, L1 FFN, L2 SA, L2 CA, L2 FFN)
echo   [R]     Randomize Table Scene & Clutter
echo   [1]     Pick and Place Task Prompt
echo   [2]     Push Towards Task Prompt
echo =========================================================
python visualize_full_smolvla.py
