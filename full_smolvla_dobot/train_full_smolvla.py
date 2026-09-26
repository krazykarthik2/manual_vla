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
CACHE_VERSION = "v3_arch3b_multi_layer"
os.makedirs(MODEL_DIR, exist_ok=True)

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

def _caching_worker(worker_id, gpu_id, indexed_files, cache_batch_size, out_dict, progress_queue):
    """
    Dedicated worker process to extract SmolVLM multi-layer features in parallel on assigned GPU/device.
    """
    target_device = f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu"
    try:
        policy = FullSmolVLAPolicy(d_action_model=128, load_backbone=True, device=target_device)
        results = []

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

            for j, orig_idx in enumerate(orig_indices):
                results.append((orig_idx, raw_h[j].float().cpu()))

            progress_queue.put(len(chunk))

        out_dict[worker_id] = results
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        print(f"[ERROR in Worker {worker_id} on {target_device}]: {e}\n{traceback.format_exc()}", flush=True)
        out_dict[worker_id] = []

def run_parallel_caching(files, cache_file, num_workers=None, workers_per_gpu=2, cache_batch_size=32):
    """
    Parallel multi-process feature precomputation across multiple GPUs and multiple workers per GPU.
    """
    num_demos = len(files)
    num_gpus = torch.cuda.device_count()

    if num_workers is None:
        if num_gpus > 0:
            num_workers = num_gpus * workers_per_gpu
        else:
            num_workers = min(4, os.cpu_count() or 2)

    # Shard demonstrations across workers
    indexed_files = list(enumerate(files))
    shards = [indexed_files[i::num_workers] for i in range(num_workers)]
    
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    out_dict = manager.dict()
    progress_queue = manager.Queue()

    gpu_assignments = []
    for w in range(num_workers):
        gpu_id = (w % num_gpus) if num_gpus > 0 else None
        gpu_assignments.append(gpu_id)

    dev_desc = f"{num_gpus} GPU(s) ({workers_per_gpu} workers/GPU -> {num_workers} parallel workers)" if num_gpus > 0 else f"{num_workers} CPU workers"
    print(f">> Launching parallel SmolVLM feature precomputation across {dev_desc} for {num_demos} demonstrations...", flush=True)

    processes = []
    for w in range(num_workers):
        p = ctx.Process(
            target=_caching_worker,
            args=(w, gpu_assignments[w], shards[w], cache_batch_size, out_dict, progress_queue)
        )
        p.start()
        processes.append(p)

    # Main thread tracks unified live progress bar
    pbar = tqdm(total=num_demos, desc="Parallel SmolVLM Extraction")
    done_count = 0
    while done_count < num_demos:
        try:
            n = progress_queue.get(timeout=1.0)
            pbar.update(n)
            done_count += n
        except Exception:
            # Check if all processes are still alive
            if all(not p.is_alive() for p in processes):
                break
    pbar.close()

    for p in processes:
        p.join()

    # Reassemble results in exact demonstration file order
    all_results = []
    for w in range(num_workers):
        if w in out_dict:
            all_results.extend(out_dict[w])

    all_results.sort(key=lambda x: x[0])
    cached_raw_hidden = [item[1] for item in all_results]

    if len(cached_raw_hidden) != num_demos:
        raise RuntimeError(f"Parallel caching gathered {len(cached_raw_hidden)}/{num_demos} items. Some workers may have failed.")

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    torch.save({
        'version': CACHE_VERSION,
        'files': files,
        'raw_hidden': cached_raw_hidden
    }, cache_file)
    print(f">> [DONE] Parallel caching complete. Saved -> {cache_file}", flush=True)

class FastFullSmolVLADataset(Dataset):
    """
    High-Performance Dataset with Multi-GPU / Multi-Worker Parallel Pre-Computed Features.
    """
    def __init__(self, data_dir, cache_file=CACHE_FILE, force_recache=False, num_workers=None, workers_per_gpu=2, cache_batch_size=32):
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
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
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

def train(epochs=200, batch_size=32, lr=1.5e-3, force_recache=False, num_workers=None, workers_per_gpu=2):
    print("=" * 70, flush=True)
    print("   Full SmolVLA Policy with Real Hugging Face SmolVLM Backbone", flush=True)
    print("   - Backbone: HuggingFaceTB/SmolVLM-256M-Instruct", flush=True)
    print("   - Architecture: 3B (Multi-Layer Intermediate Feature Fusion: Layers 10, 20, 30)", flush=True)
    print("   - Parallelism: Multi-GPU / Multi-Worker Parallel Feature Precomputation", flush=True)
    print("   - Multimodal Action Expert (Cross-Attention Action Decoder Head)", flush=True)
    print(f"   - Hardware Compute Engine: {DEVICE}", flush=True)
    print("=" * 70, flush=True)

    dataset = FastFullSmolVLADataset(
        DATA_DIR,
        force_recache=force_recache,
        num_workers=num_workers,
        workers_per_gpu=workers_per_gpu,
        cache_batch_size=32 if DEVICE.type == "cuda" else 8
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
    parser = argparse.ArgumentParser(description="Multi-GPU / Multi-Worker Parallel Pre-computation and Training for Full SmolVLA")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for Action Expert training (default: 32)")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate (default: 1.5e-3)")
    parser.add_argument("--recache", action="store_true", help="Force re-compute the SmolVLM multi-layer features")
    parser.add_argument("--num-workers", type=int, default=None, help="Total number of parallel caching workers")
    parser.add_argument("--workers-per-gpu", type=int, default=2, help="Number of parallel workers per GPU (default: 2)")
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
