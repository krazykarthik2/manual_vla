import os
import sys
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "full_smolvlm_features_cache.pt")
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

CACHE_VERSION = "v2_arch3b_multi_layer"

class FastFullSmolVLADataset(Dataset):
    def __init__(self, data_dir, cache_file=CACHE_FILE, force_recache=False):
        valid_cache = False
        if os.path.exists(cache_file) and not force_recache:
            try:
                cache = torch.load(cache_file, map_location="cpu")
                if isinstance(cache, dict) and cache.get("version") == CACHE_VERSION and "raw_hidden" in cache:
                    valid_cache = True
                else:
                    print(f">> Cache version mismatch or legacy cache found at {cache_file}. Regenerating...", flush=True)
            except Exception as e:
                print(f">> Cache file corrupted ({e}). Regenerating...", flush=True)

        if not valid_cache:
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except OSError:
                    pass

            print(f">> Generating Architecture 3B multi-layer cache with SmolVLM backbone on {DEVICE}...", flush=True)
            files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
            if not files:
                print(">> No demonstrations found. Auto-generating 100 clean demonstrations...", flush=True)
                from auto_generate_demos import run_auto_demonstrator
                run_auto_demonstrator(num_demos=100)
                files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

            policy = FullSmolVLAPolicy(d_action_model=128, load_backbone=True, device=DEVICE)
            cached_raw_hidden = []
            print(f">> Caching multi-layer intermediate hidden states for {len(files)} demonstrations...", flush=True)

            for f in tqdm(files, desc="SmolVLM Multi-Layer Caching"):
                d = np.load(f, allow_pickle=True)
                img_chw = d['images'][0] # [3, 64, 64] float in [0, 1]
                img_hwc = (np.transpose(img_chw, (1, 2, 0)) * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_hwc)
                prompt_str = str(d['prompt'][0])

                with torch.no_grad():
                    # Architecture 3B: extract raw multi-layer intermediate hidden states (layers 10, 20, 30)
                    raw_hidden = policy.extract_smolvlm_context([pil_img], [prompt_str], return_raw_hidden=True)
                cached_raw_hidden.append(raw_hidden[0].float().cpu())

            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save({
                'version': CACHE_VERSION,
                'files': files,
                'raw_hidden': cached_raw_hidden
            }, cache_file)
            print(f">> Saved Architecture 3B multi-layer cache -> {cache_file}", flush=True)

        print(f">> Loading Pretrained SmolVLM multi-layer features from {cache_file}...", flush=True)
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
            proprio0 = proprio[0]

            self.samples.append((
                cached_raw[idx],
                torch.tensor(proprio0, dtype=torch.float32),
                norm_traj
            ))

        print(f">> Loaded {len(self.samples)} cached demonstration trajectories with SmolVLM backbone.", flush=True)

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

def train(epochs=150, batch_size=16, lr=1.5e-3, force_recache=False):
    print("=" * 70, flush=True)
    print("   Full SmolVLA Policy with Real Hugging Face SmolVLM Backbone", flush=True)
    print("   - Backbone: HuggingFaceTB/SmolVLM-256M-Instruct", flush=True)
    print("   - Architecture: 3B (Multi-Layer Intermediate Feature Fusion: Layers 10, 20, 30)", flush=True)
    print("   - Multimodal Action Expert (Cross-Attention Action Decoder Head)", flush=True)
    print(f"   - Hardware Compute Engine: {DEVICE}", flush=True)
    print("=" * 70, flush=True)

    dataset = FastFullSmolVLADataset(DATA_DIR, force_recache=force_recache)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    # Instantiate FullSmolVLAPolicy without loading heavy LLM into memory during training since features are cached
    policy = FullSmolVLAPolicy(d_action_model=128, load_backbone=False, device=DEVICE).to(DEVICE)
    policy.train()

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    print(f">> Trainable Policy Parameters: {sum(p.numel() for p in trainable_params):,}", flush=True)

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss(reduction='none')

    model_path = os.path.join(MODEL_DIR, "dobot_full_smolvla_policy.pth")
    best_loss = float("inf")

    print(f"\n>> Training Full SmolVLA Policy across {len(dataset)} demonstrations ({epochs} epochs)...", flush=True)
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

                # Project multi-layer features into action expert dimension with trainable vlm_proj
                vlm_tokens = policy.vlm_proj(raw_hidden)

                x_0 = torch.randn_like(x_1)
                t = torch.rand(B, device=DEVICE)
                t_expanded = t.view(B, 1, 1)

                x_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1
                u_t = x_1 - x_0

                v_pred = policy.forward_from_embeddings(x_t, t, vlm_tokens, proprio=proprio)

                # Flow-matching loss with weights on precision dims (z and grip)
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
                # Only save trainable action expert parameters, not frozen SmolVLM backbone
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
    import argparse
    parser = argparse.ArgumentParser(description="Train Full SmolVLA Policy with Architecture 3B Multi-Layer Features")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default: 150)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate (default: 1.5e-3)")
    parser.add_argument("--recache", action="store_true", help="Force regenerate the SmolVLM multi-layer cache")
    args, unknown = parser.parse_known_args()

    # Support passing epoch as first positional arg for backwards compatibility with .bat scripts
    if len(unknown) > 0 and unknown[0].isdigit():
        args.epochs = int(unknown[0])

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, force_recache=args.recache)
