#!/bin/bash
# autoscale_train.sh
# Automatically scales hardware limits (CPU + GPU) and launches the VLA training pipeline.

echo "========================================================="
echo "       FULL SMOLVLA AUTO-SCALING TRAINING LAUNCHER       "
echo "========================================================="

# --- CPU DETECTION ---
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

# --- GPU DETECTION & AUTOSCALING ---
# Try nvidia-smi first; if it fails (e.g. NVML version mismatch), fall back to PyTorch directly
GPU_CHECK=$(nvidia-smi -L 2>/dev/null)
if [ -n "$GPU_CHECK" ]; then
    NUM_GPUS=$(echo "$GPU_CHECK" | wc -l)
    GPU_NAMES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | paste -sd ", ")
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -n 1)
    echo "[INFO] Detected $NUM_GPUS NVIDIA GPU(s) via NVML: $GPU_NAMES (VRAM: $TOTAL_VRAM)"
else
    # Fallback: query via PyTorch CUDA directly (bypasses broken NVML)
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
    if [ "$NUM_GPUS" -gt "0" ]; then
        GPU_NAMES=$(python -c "import torch; print(', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))" 2>/dev/null)
        echo "[INFO] Detected $NUM_GPUS NVIDIA GPU(s) via PyTorch Driver API: $GPU_NAMES"
    else
        echo "[WARN] No CUDA GPUs detected. Proceeding with CPU hardware assumptions."
    fi
fi

if [ "$NUM_GPUS" -gt "1" ]; then
    echo "[INFO] Multi-GPU environment detected. Exposing all GPUs for scaling."
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($NUM_GPUS-1)))
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_P2P_DISABLE=0
export TF_CPP_MIN_LOG_LEVEL=3

# Set file descriptor limit higher for parallel dataset I/O streaming
ulimit -n 65535 2>/dev/null

echo "[INFO] Hardware limits auto-scaled and variables exported."
echo "[INFO] Handing off to dynamic Python distributed scaler..."
echo "========================================================="

# Launch the training script
# Note: train_full_smolvla.py natively parses torch.cuda.device_count() and auto-tunes batch_size
# and parallel worker_count dynamically based on the environment variables we just exposed.
python full_smolvla_dobot/train_full_smolvla.py "$@"

echo "========================================================="
echo "[INFO] Training completed or terminated."
