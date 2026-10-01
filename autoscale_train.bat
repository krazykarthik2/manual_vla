@echo off
echo =========================================================
echo        FULL SMOLVLA AUTO-SCALING TRAINING LAUNCHER
echo =========================================================

:: Auto-detect number of CPU cores
set CORES=%NUMBER_OF_PROCESSORS%
echo [INFO] Detected %CORES% CPU cores.

:: Optimize thread usage for CPU-bound tasks (physics simulation & data loading)
set OMP_NUM_THREADS=%CORES%
set MKL_NUM_THREADS=%CORES%
set OPENBLAS_NUM_THREADS=%CORES%

:: Optimize PyTorch CUDA memory allocator to reduce fragmentation on multi-GPU setups
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo [INFO] Hardware limits auto-scaled. Launching training pipeline...
echo =========================================================

:: Launch the training script (which internally scales batch sizes and GPU workers)
python full_smolvla_dobot\train_full_smolvla.py %*

echo =========================================================
echo [INFO] Training completed or terminated.
pause
