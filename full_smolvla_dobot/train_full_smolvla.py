import os
import sys
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# Try raising file descriptor limits for high-throughput parallel IO
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_limit = min(65535, hard) if hard > 0 else 65535
    resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard))
except Exception:
    pass

# Optimize PyTorch CUDA memory allocator to eliminate fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(os.path.dirname(__file__))
from full_smolvla_model import FullSmolVLAPolicy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    NUM_CORES = os.cpu_count() or 4
    torch.set_num_threads(NUM_CORES)
    torch.set_num_interop_threads(NUM_CORES)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "demonstrations")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "full_smolvlm_arch3b_features_cache.pt")
SHARDS_DIR = os.path.join(os.path.dirname(__file__), "data", "smolvlm_cache_shards")
CACHE_VERSION = "v3_arch3b_multi_layer"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SHARDS_DIR, exist_ok=True)

def auto_detect_hardware_config(user_batch_size=None, user_num_workers=None, user_workers_per_gpu=None):
    """
    Intelligently auto-adjusts batch sizes, num_workers, and workers_per_gpu
    based on available GPUs, VRAM per GPU, and CPU core count to prevent OOM while maximizing speed.
    """
    cpu_cores = os.cpu_count() or 4
    num_gpus = torch.cuda.device_count()

    if num_gpus > 0:
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            total_vram_gb = 16.0
            gpu_name = "CUDA GPU"

        # 1 dedicated worker per GPU is cleanest & fastest without VRAM contention
        if user_workers_per_gpu is not None and user_workers_per_gpu > 0:
            workers_per_gpu = user_workers_per_gpu
        else:
            workers_per_gpu = 1

        if user_num_workers is not None and user_num_workers > 0:
            num_workers = user_num_workers
        else:
            num_workers = num_gpus * workers_per_gpu

        cache_batch_size = 16

        # Auto training batch size for Action Expert (712k parameters)
        if user_batch_size is not None and user_batch_size > 0:
            batch_size = user_batch_size
        else:
            batch_size = 64 if total_vram_gb >= 20 else 32

        hw_summary = (
            f"Hardware: {num_gpus}x {gpu_name} ({total_vram_gb:.1f} GB VRAM each) | {cpu_cores} CPU cores\n"
            f">> Auto-Adjusted: workers_per_gpu={workers_per_gpu}, total_workers={num_workers}, "
            f"cache_batch_size={cache_batch_size}, train_batch_size={batch_size} (in bfloat16/float16)"
        )
    else:
        num_workers = user_num_workers if (user_num_workers and user_num_workers > 0) else min(4, max(1, cpu_cores // 2))
        workers_per_gpu = 1
        batch_size = user_batch_size if (user_batch_size and user_batch_size > 0) else 16
        cache_batch_size = 8
        hw_summary = (
            f"Hardware: CPU environment ({cpu_cores} cores)\n"
            f">> Auto-Adjusted: CPU workers={num_workers}, cache_batch_size={cache_batch_size}, train_batch_size={batch_size}"
        )

    return {
        "num_workers": num_workers,
        "workers_per_gpu": workers_per_gpu,
        "batch_size": batch_size,
        "cache_batch_size": cache_batch_size,
        "hw_summary": hw_summary
    }

def safe_save_model(state_dict, path):
    tmp_path = path + ".tmp"
    for attempt in range(5):
        try:
            torch.save(state_dict, tmp_path)
            if os.path.exists(path):
                try:
                    os.replace(tmp_path, path)
                except OSError:
                    import time
                    time.sleep(0.05)
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    os.rename(tmp_path, path)
            else:
                os.rename(tmp_path, path)
            return
        except Exception as e:
            if attempt == 4:
                print(f"[WARN] Failed to save checkpoint: {e}", flush=True)
            import time
            time.sleep(0.1)

ACTION_MEAN = torch.tensor([0.2028, 0.0008, 0.0854, 0.5360], dtype=torch.float32)
ACTION_STD  = torch.tensor([0.0330, 0.0837, 0.0362, 0.4987], dtype=torch.float32)

def _caching_worker(worker_id, gpu_id, indexed_files, cache_batch_size, shards_dir, progress_queue, status_dict):
    """
    Dedicated worker process to extract SmolVLM multi-layer features on assigned GPU device.
    Saves results incrementally to disk after every single batch to guarantee zero lost progress.
    """
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        target_device = f"cuda:{gpu_id}"
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        target_device = "cpu"

    try:
        policy = FullSmolVLAPolicy(d_action_model=128, load_backbone=True, device=target_device)

        for i in range(0, len(indexed_files), cache_batch_size):
            chunk = indexed_files[i:i + cache_batch_size]
            batch_imgs = []
            batch_prompts = []
            orig_indices = []

            for idx, f in chunk:
                d = np.load(f, allow_pickle=True)
                img_chw = d['images'][0] # [3, 64, 64]
                img_hwc = (np.transpose(img_chw, (1, 2, 0)) * 255).astype(np.uint8)
                batch_imgs.append(Image.fromarray(img_hwc))
                batch_prompts.append(str(d['prompt'][0]))
                orig_indices.append(idx)

            with torch.no_grad():
                raw_h = policy.extract_smolvlm_context(batch_imgs, batch_prompts, return_raw_hidden=True)

            batch_results = []
            for j, orig_idx in enumerate(orig_indices):
                batch_results.append((orig_idx, raw_h[j].float().cpu()))

            # Incrementally save batch shard to disk immediately
            chunk_file = os.path.join(shards_dir, f"shard_{orig_indices[0]:06d}_{orig_indices[-1]:06d}.pt")
            tmp_chunk_file = chunk_file + ".tmp"
            torch.save(batch_results, tmp_chunk_file)
            os.replace(tmp_chunk_file, chunk_file)

            progress_queue.put(len(chunk))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status_dict[worker_id] = True

    except Exception as e:
        import traceback
        print(f"[ERROR in Worker {worker_id} on {target_device}]: {e}\n{traceback.format_exc()}", flush=True)
        status_dict[worker_id] = False

def run_parallel_caching(files, cache_file, shards_dir=SHARDS_DIR, num_workers=None, workers_per_gpu=None, cache_batch_size=None):
    """
    Parallel multi-process feature precomputation with incremental disk checkpointing & resume capabilities.
    """
    num_demos = len(files)
    num_gpus = torch.cuda.device_count()

    hw = auto_detect_hardware_config(
        user_num_workers=num_workers,
        user_workers_per_gpu=workers_per_gpu
    )
    num_workers = hw["num_workers"]
    workers_per_gpu = hw["workers_per_gpu"]
    cache_batch_size = cache_batch_size or hw["cache_batch_size"]

    os.makedirs(shards_dir, exist_ok=True)

    # 1. Scan for existing incremental shard files from previous/interrupted runs
    existing_cached_items = {}
    existing_shard_files = glob.glob(os.path.join(shards_dir, "shard_*.pt"))
    for sf in existing_shard_files:
        try:
            items = torch.load(sf, map_location="cpu")
            for orig_idx, tensor in items:
                existing_cached_items[orig_idx] = tensor
        except Exception:
            pass

    # 2. Filter demonstrations that still need extraction
    indexed_files = list(enumerate(files))
    missing_indexed_files = [item for item in indexed_files if item[0] not in existing_cached_items]

    already_done = len(existing_cached_items)
    if already_done > 0:
        print(f">> [RESUME] Found {already_done}/{num_demos} demonstrations already cached on disk! Only extracting remaining {len(missing_indexed_files)} demos...", flush=True)

    if len(missing_indexed_files) > 0:
        # Shard missing work across workers
        actual_workers = min(num_workers, len(missing_indexed_files))
        shards = [missing_indexed_files[i::actual_workers] for i in range(actual_workers)]

        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        status_dict = manager.dict()
        progress_queue = manager.Queue()

        gpu_assignments = []
        for w in range(actual_workers):
            gpu_id = (w % num_gpus) if num_gpus > 0 else None
            gpu_assignments.append(gpu_id)

        dev_desc = f"{num_gpus} GPU(s) ({workers_per_gpu} worker/GPU -> {actual_workers} parallel workers, batch_size={cache_batch_size})" if num_gpus > 0 else f"{actual_workers} CPU workers"
        print(f">> Launching parallel SmolVLM extraction across {dev_desc} for {len(missing_indexed_files)} remaining demonstrations...", flush=True)

        processes = []
        for w in range(actual_workers):
            p = ctx.Process(
                target=_caching_worker,
                args=(w, gpu_assignments[w], shards[w], cache_batch_size, shards_dir, progress_queue, status_dict)
            )
            p.start()
            processes.append(p)

        pbar = tqdm(total=num_demos, initial=already_done, desc="SmolVLM Parallel Extraction (Incremental)")
        done_count = already_done
        while done_count < num_demos:
            try:
                n = progress_queue.get(timeout=1.0)
                pbar.update(n)
                done_count += n
            except Exception:
                if all(not p.is_alive() for p in processes):
                    break
        pbar.close()

        for p in processes:
            p.join()

    # 3. Assemble all shards from disk into unified master cache file
    all_results = {}
    final_shard_files = glob.glob(os.path.join(shards_dir, "shard_*.pt"))
    for sf in final_shard_files:
        try:
            items = torch.load(sf, map_location="cpu")
            for orig_idx, tensor in items:
                all_results[orig_idx] = tensor
        except Exception as e:
            print(f"[WARN] Corrupt shard file {sf}: {e}", flush=True)

    if len(all_results) != num_demos:
        raise RuntimeError(f"Parallel caching gathered {len(all_results)}/{num_demos} items. Some workers may have failed.")

    # Sort in exact demonstration order
    cached_raw_hidden = [all_results[i] for i in range(num_demos)]

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    torch.save({
        'version': CACHE_VERSION,
        'files': files,
        'raw_hidden': cached_raw_hidden
    }, cache_file)
    print(f">> [DONE] Master SmolVLM cache assembled and saved -> {cache_file}", flush=True)

    # Clean up intermediate shard files
    for sf in final_shard_files:
        try:
            os.remove(sf)
        except OSError:
            pass

class FastFullSmolVLADataset(Dataset):
    """
    High-Performance Dataset with Auto-Tuned Multi-GPU / Multi-Worker Parallel Pre-Computed Features.
    """
    def __init__(self, data_dir, cache_file=CACHE_FILE, force_recache=False, num_workers=None, workers_per_gpu=None, cache_batch_size=None):
        valid_cache = False
        if os.path.exists(cache_file) and not force_recache:
            try:
                print(f">> Checking SmolVLM cache at {cache_file}...", flush=True)
                cache = torch.load(cache_file, map_location="cpu")
                if isinstance(cache, dict) and cache.get("version") == CACHE_VERSION and "raw_hidden" in cache:
                    valid_cache = True
                else:
                    print(">> Cache version mismatch or legacy format. Regenerating cache...", flush=True)
            except Exception as e:
                print(f">> Cache file corrupted ({e}). Regenerating cache...", flush=True)

        if not valid_cache:
            if os.path.exists(cache_file) and force_recache:
                try:
                    os.remove(cache_file)
                except OSError:
                    pass
                # Also clean shards if forced recache
                for sf in glob.glob(os.path.join(SHARDS_DIR, "shard_*.pt")):
                    try:
                        os.remove(sf)
                    except OSError:
                        pass

            files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
            if not files:
                print(">> No demonstrations found. Auto-generating 100 clean demonstrations...", flush=True)
                from auto_generate_demos import run_auto_demonstrator
                run_auto_demonstrator(num_demos=100)
                files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

            run_parallel_caching(
                files=files,
                cache_file=cache_file,
                shards_dir=SHARDS_DIR,
                num_workers=num_workers,
                workers_per_gpu=workers_per_gpu,
                cache_batch_size=cache_batch_size
            )

        print(f">> Loading Precomputed SmolVLM multi-layer features into memory...", flush=True)
        cache = torch.load(cache_file, map_location="cpu")
        files = cache['files']
        cached_raw = cache['raw_hidden']

        self.samples = []
        for idx, f in enumerate(files):
            d = np.load(f, allow_pickle=True)
            proprio = d['proprioception'].astype(np.float32)
            acts = d['actions'].astype(np.float32)

            raw_traj = np.concatenate([proprio[:, :3], acts[:, 4:5]], axis=-1)
            norm_traj = (torch.tensor(raw_traj, dtype=torch.float32) - ACTION_MEAN) / (ACTION_STD + 1e-6)
            proprio0 = torch.tensor(proprio[0], dtype=torch.float32)

            self.samples.append((
                cached_raw[idx],
                proprio0,
                norm_traj
            ))

        print(f">> Ready! Loaded {len(self.samples)} demonstrations into memory.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def pad_collate_fn(batch):
    raw_list, proprio_list, traj_list = zip(*batch)
    max_len = max(t.size(0) for t in raw_list)
    hidden_dim = raw_list[0].size(1)

    padded_raw = torch.zeros(len(raw_list), max_len, hidden_dim, dtype=torch.float32)
    for i, t in enumerate(raw_list):
        padded_raw[i, :t.size(0)] = t

    proprios = torch.stack(proprio_list)
    trajs = torch.stack(traj_list)
    return padded_raw, proprios, trajs

def train(epochs=200, batch_size=None, lr=1.5e-3, force_recache=False, num_workers=None, workers_per_gpu=None):
    hw = auto_detect_hardware_config(
        user_batch_size=batch_size,
        user_num_workers=num_workers,
        user_workers_per_gpu=workers_per_gpu
    )
    batch_size = hw["batch_size"]
    num_workers = hw["num_workers"]
    workers_per_gpu = hw["workers_per_gpu"]

    print("=" * 70, flush=True)
    print("   Full SmolVLA Policy with Real Hugging Face SmolVLM Backbone", flush=True)
    print("   - Backbone: HuggingFaceTB/SmolVLM-256M-Instruct (bfloat16 / float16)", flush=True)
    print("   - Architecture: 3B (Multi-Layer Intermediate Feature Fusion: Layers 10, 20, 30)", flush=True)
    print("   - Parallelism: Auto-Tuned Multi-GPU / Multi-Worker Feature Precomputation", flush=True)
    print("   - Checkpointing: Incremental Disk Sharding (Zero Progress Lost on Interrupt)", flush=True)
    print(f"   - {hw['hw_summary']}", flush=True)
    print("=" * 70, flush=True)

    dataset = FastFullSmolVLADataset(
        DATA_DIR,
        force_recache=force_recache,
        num_workers=num_workers,
        workers_per_gpu=workers_per_gpu,
        cache_batch_size=hw["cache_batch_size"]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    policy = FullSmolVLAPolicy(d_action_model=128, load_backbone=False, device=DEVICE).to(DEVICE)
    policy.train()

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    print(f">> Trainable Policy Parameters: {sum(p.numel() for p in trainable_params):,}", flush=True)

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss(reduction='none')

    model_path = os.path.join(MODEL_DIR, "dobot_full_smolvla_policy.pth")
    best_loss = float("inf")

    print(f"\n>> Training Full SmolVLA Policy across {len(dataset)} demonstrations ({epochs} epochs, batch_size={batch_size})...", flush=True)
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training Full SmolVLA")

    try:
        for epoch in epoch_pbar:
            total_loss = 0.0
            for raw_hidden, proprio, x_1 in dataloader:
                raw_hidden = raw_hidden.to(DEVICE)
                proprio = proprio.to(DEVICE)
                x_1 = x_1.to(DEVICE)

                B = raw_hidden.size(0)
                optimizer.zero_grad(set_to_none=True)

                vlm_tokens = policy.vlm_proj(raw_hidden)

                x_0 = torch.randn_like(x_1)
                t = torch.rand(B, device=DEVICE)
                t_expanded = t.view(B, 1, 1)

                x_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1
                u_t = x_1 - x_0

                v_pred = policy.forward_from_embeddings(x_t, t, vlm_tokens, proprio=proprio)

                loss_raw = loss_fn(v_pred, u_t)
                dim_weights = torch.tensor([1.2, 1.2, 1.5, 2.5], device=DEVICE).view(1, 1, 4)
                flow_loss = (loss_raw * dim_weights).mean()

                flow_loss.backward()
                optimizer.step()
                total_loss += flow_loss.item() * B

            scheduler.step()
            avg_loss = total_loss / len(dataset)

            if avg_loss < best_loss or epoch % 10 == 0:
                best_loss = min(best_loss, avg_loss)
                filtered = {k: v for k, v in policy.state_dict().items() if not k.startswith("smolvlm.")}
                safe_save_model(filtered, model_path)

            current_lr = scheduler.get_last_lr()[0]
            epoch_pbar.set_postfix({
                "OT_Loss": f"{avg_loss:.5f}",
                "Best": f"{best_loss:.5f}",
                "LR": f"{current_lr:.6f}"
            })

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Saving checkpoint...", flush=True)
        filtered = {k: v for k, v in policy.state_dict().items() if not k.startswith("smolvlm.")}
        safe_save_model(filtered, model_path)
        return

    filtered = {k: v for k, v in policy.state_dict().items() if not k.startswith("smolvlm.")}
    safe_save_model(filtered, model_path)
    print(f"\n[SUCCESS] Full SmolVLA Checkpoint saved -> {model_path}", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import argparse
    parser = argparse.ArgumentParser(description="Auto-Tuned Multi-GPU / Multi-Worker Parallel Training for Full SmolVLA")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: auto-detected based on GPU VRAM)")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate (default: 1.5e-3)")
    parser.add_argument("--recache", action="store_true", help="Force re-compute the SmolVLM multi-layer features")
    parser.add_argument("--num-workers", type=int, default=None, help="Total caching workers (default: auto-detected)")
    parser.add_argument("--workers-per-gpu", type=int, default=None, help="Workers per GPU (default: auto-detected)")
    args, unknown = parser.parse_known_args()

    if len(unknown) > 0 and unknown[0].isdigit():
        args.epochs = int(unknown[0])

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        force_recache=args.recache,
        num_workers=args.num_workers,
        workers_per_gpu=args.workers_per_gpu
    )
