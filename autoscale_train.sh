#!/bin/bash
# autoscale_train.sh
# Automatically scales hardware limits and launches the VLA training pipeline.

echo "========================================================="
echo "       FULL SMOLVLA AUTO-SCALING TRAINING LAUNCHER       "
echo "========================================================="

# Auto-detect number of CPU cores
if command -v nproc &> /dev/null; then
    CORES=$(nproc)
else
    CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi

echo "[INFO] Detected $CORES CPU cores."

# Optimize thread usage for CPU-bound tasks
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export OPENBLAS_NUM_THREADS=$CORES

# Optimize PyTorch CUDA memory allocator to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Set file descriptor limit higher for parallel dataloading
ulimit -n 65535 2>/dev/null

echo "[INFO] Hardware limits auto-scaled. Launching training pipeline..."
echo "========================================================="

# Launch the training script (which internally scales batch sizes)
python full_smolvla_dobot/train_full_smolvla.py "$@"

echo "========================================================="
echo "[INFO] Training completed or terminated."
