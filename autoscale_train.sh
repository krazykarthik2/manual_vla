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
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    GPU_NAMES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | paste -sd ", ")
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    echo "[INFO] Detected $NUM_GPUS NVIDIA GPU(s): $GPU_NAMES (VRAM: $TOTAL_VRAM)"
    
    if [ "$NUM_GPUS" -gt "1" ]; then
        echo "[INFO] Multi-GPU environment detected. Exposing all GPUs for scaling."
        # The python script explicitly utilizes `torch.cuda.device_count()` to spawn multiprocess extraction workers
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($NUM_GPUS-1)))
    fi
    
    # Optimize PyTorch CUDA memory allocator to eliminate VRAM fragmentation and prevent OOM
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    
    # High performance interconnect configurations (if applicable)
    export NCCL_P2P_DISABLE=0
    export TF_CPP_MIN_LOG_LEVEL=3
else
    echo "[WARN] nvidia-smi not found. Proceeding with CPU-only or non-NVIDIA GPU hardware assumptions."
fi

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
