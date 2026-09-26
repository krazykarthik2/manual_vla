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

class FastFullSmolVLADataset(Dataset):
    def __init__(self, data_dir, cache_file=CACHE_FILE):
        if not os.path.exists(cache_file):
            print(f">> Cache file {cache_file} not found. Generating cache with SmolVLM backbone on {DEVICE}...", flush=True)
            files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
            if not files:
                print(">> No demonstrations found. Auto-generating 100 clean demonstrations...", flush=True)
                from auto_generate_demos import run_auto_demonstrator
                run_auto_demonstrator(num_demos=100)
                files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

            policy = FullSmolVLAPolicy(d_action_model=128, device=DEVICE).to(DEVICE)
            cached_vlm_tokens = []
            print(f">> Caching SmolVLM multimodal tokens for {len(files)} demonstrations...", flush=True)

            for f in tqdm(files, desc="SmolVLM Caching"):
                d = np.load(f, allow_pickle=True)
                img_chw = d['images'][0] # [3, 64, 64] float in [0, 1]
                img_hwc = (np.transpose(img_chw, (1, 2, 0)) * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_hwc)
                prompt_str = str(d['prompt'][0])

                with torch.no_grad():
                    vlm_tokens = policy.extract_smolvlm_context([pil_img], [prompt_str]) # [1, seq_len, 128]
                cached_vlm_tokens.append(vlm_tokens[0].float().cpu())

            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save({
                'files': files,
                'vlm_tokens': cached_vlm_tokens
            }, cache_file)
            print(f">> Saved SmolVLM cache -> {cache_file}", flush=True)

        print(f">> Loading Pretrained SmolVLM features from {cache_file}...", flush=True)
        cache = torch.load(cache_file, map_location="cpu")
        files = cache['files']
        cached_tokens = cache['vlm_tokens']

        self.samples = []
        for idx, f in enumerate(files):
            d = np.load(f, allow_pickle=True)
            proprio = d['proprioception'].astype(np.float32)
            acts = d['actions'].astype(np.float32)

            raw_traj = np.concatenate([proprio[:, :3], acts[:, 4:5]], axis=-1)
            norm_traj = (torch.tensor(raw_traj, dtype=torch.float32) - ACTION_MEAN) / (ACTION_STD + 1e-6)
            proprio0 = proprio[0]

            self.samples.append((
                cached_tokens[idx],
                torch.tensor(proprio0, dtype=torch.float32),
                norm_traj
            ))

        print(f">> Loaded {len(self.samples)} cached demonstration trajectories with SmolVLM backbone.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def pad_collate_fn(batch):
    tokens_list, proprio_list, traj_list = zip(*batch)
    max_len = max(t.size(0) for t in tokens_list)
    d_model = tokens_list[0].size(1)

    padded_tokens = torch.zeros(len(tokens_list), max_len, d_model, dtype=torch.float32)
    for i, t in enumerate(tokens_list):
        padded_tokens[i, :t.size(0)] = t

    proprios = torch.stack(proprio_list)
    trajs = torch.stack(traj_list)
    return padded_tokens, proprios, trajs

def train(epochs=150, batch_size=16, lr=1.5e-3):
    print("=" * 70, flush=True)
    print("   Full SmolVLA Policy with Real Hugging Face SmolVLM Backbone", flush=True)
    print("   - Backbone: HuggingFaceTB/SmolVLM-256M-Instruct", flush=True)
    print("   - Multimodal Action Expert (Cross-Attention Action Decoder Head)", flush=True)
    print(f"   - Hardware Compute Engine: {DEVICE}", flush=True)
    print("=" * 70, flush=True)

    dataset = FastFullSmolVLADataset(DATA_DIR)
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
            for vlm_tokens, proprio, x_1 in dataloader:
                vlm_tokens = vlm_tokens.to(DEVICE)
                proprio = proprio.to(DEVICE)
                x_1 = x_1.to(DEVICE)

                B = vlm_tokens.size(0)
                optimizer.zero_grad(set_to_none=True)

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
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    train(epochs=epochs)
