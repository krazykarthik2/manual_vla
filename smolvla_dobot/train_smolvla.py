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

from smolvla_embedding import SmolVLMTokenizer
from smolvla_model import SmolVLABackbone

# Device Configuration (Auto CUDA / Multi-core CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    NUM_CORES = os.cpu_count() or 4
    torch.set_num_threads(NUM_CORES)
    torch.set_num_interop_threads(NUM_CORES)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "demonstrations")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
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

# Normalization constants for 4D actions (X, Y, Z, Gripper)
ACTION_MEAN = torch.tensor([0.2066, 0.0024, 0.0738, 0.2617], dtype=torch.float32)
ACTION_STD  = torch.tensor([0.0326, 0.0816, 0.0405, 0.4396], dtype=torch.float32)

class ContinuousSinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# -----------------------------------------------------------------------------
# 1. SmolVLA Policy Architecture with True ViT & Multimodal Cross-Attention
# -----------------------------------------------------------------------------
class SmolVLAPolicy(nn.Module):
    """
    True SmolVLA Policy:
    - Backbone: SmolVLABackbone (ViT Patch Encoder + Multimodal Self/Cross Attention)
    - Inputs: Subword Language Token IDs + 64x64 Overhead Image + 5D Proprioception
    - Action Head: Flow Matching Vector Field Regressor over Horizon H=128
    """
    def __init__(self, vocab_size=32, horizon=128, action_dim=4, d_model=128, num_layers=2):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.d_model = d_model

        # 1. Multimodal Backbone (Language + ViT 256 Patches)
        self.backbone = SmolVLABackbone(vocab_size=vocab_size, d_model=d_model, text_len=16, nhead=4, num_layers=num_layers)

        # 2. Proprioception & Time Conditioning
        self.proprio_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.time_embed = nn.Sequential(
            ContinuousSinusoidalTimeEmbedding(64),
            nn.Linear(64, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 3. Action Decoder Queries & Cross-Attention Head
        self.action_in_proj = nn.Linear(action_dim, d_model)
        self.pos_queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        self.action_cross_attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.norm_act1 = nn.LayerNorm(d_model)
        self.norm_act2 = nn.LayerNorm(d_model)
        self.act_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

        # Output Flow Vector Field
        self.out_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim)
        )

    def forward_flow(self, x_t, t, img, token_ids, proprio=None):
        B = img.size(0)

        # 1. Multimodal feature extraction through ViT + Self/Cross Attention
        text_feats, vis_feats, _ = self.backbone(token_ids, img) # text: [B, 16, d], vis: [B, 256, d]

        # 2. Proprioception & Time conditioning
        if proprio is None:
            proprio = torch.zeros(B, 5, device=img.device)
        p_token = self.proprio_proj(proprio).unsqueeze(1) # [B, 1, d]
        t_token = self.time_embed(t).unsqueeze(1)         # [B, 1, d]

        # Joint Multimodal Context: 16 (text) + 256 (vit) + 1 (proprio) + 1 (time) = 274 tokens
        context = torch.cat([p_token, t_token, text_feats, vis_feats], dim=1) # [B, 274, d]

        # 3. Action Decoder Stream
        act_queries = self.action_in_proj(x_t) + self.pos_queries # [B, horizon, d]
        act_norm = self.norm_act1(act_queries)
        attn_out, _ = self.action_cross_attn(act_norm, context, context)
        act_queries = act_queries + attn_out
        act_queries = act_queries + self.act_ffn(self.norm_act2(act_queries))

        # 4. Predict velocity field
        v_pred = self.out_head(act_queries) # [B, horizon, 4]
        return v_pred

    @torch.no_grad()
    def sample(self, img, token_ids, proprio=None, num_steps=20):
        B = img.size(0)
        x = torch.randn(B, self.horizon, self.action_dim, device=img.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=img.device)
            v = self.forward_flow(x, t, img, token_ids, proprio=proprio)
            x = x - v * dt

        raw_x = x * ACTION_STD.to(img.device) + ACTION_MEAN.to(img.device)
        kernel = torch.ones(1, 1, 5, device=img.device) / 5.0
        raw_perm = raw_x.permute(0, 2, 1).reshape(B * self.action_dim, 1, self.horizon)
        smoothed = F.conv1d(raw_perm, kernel, padding=2)
        smoothed = smoothed.view(B, self.action_dim, self.horizon).permute(0, 2, 1)
        return smoothed


# -----------------------------------------------------------------------------
# 2. Dataset with Language Tokenizer
# -----------------------------------------------------------------------------
class SmolVLADataset(Dataset):
    def __init__(self, data_dir):
        files = sorted(glob.glob(os.path.join(data_dir, "demo_*.npz")))
        if not files:
            raise ValueError(f"No demonstration files found in {data_dir}. Run auto_generate_demos.py first!")

        self.tokenizer = SmolVLMTokenizer()
        self.samples = []
        print(f">> Indexing {len(files)} demonstrations for SmolVLA Training...", flush=True)

        for f in files:
            d = np.load(f, allow_pickle=True)
            img0 = d['images'][0].astype(np.float32) # [3, 64, 64]
            proprio = d['proprioception'].astype(np.float32) # [128, 5]
            acts = d['actions'].astype(np.float32) # [128, 6]
            
            prompt_str = str(d['prompt'][0]) if 'prompt' in d else "pick up the red cube and place it on the green platform"
            token_ids = self.tokenizer.encode(prompt_str, max_len=16)

            raw_traj = np.concatenate([proprio[:, :3], acts[:, 4:5]], axis=-1) # [128, 4]
            norm_traj = (torch.tensor(raw_traj, dtype=torch.float32) - ACTION_MEAN) / (ACTION_STD + 1e-6)
            proprio0 = proprio[0] # [5]
            self.samples.append((img0, token_ids, proprio0, norm_traj))

        print(f">> Successfully indexed {len(self.samples)} trajectory demonstrations with subword language tokens.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, token_ids, proprio0, norm_traj = self.samples[idx]
        return torch.tensor(img, dtype=torch.float32), token_ids, torch.tensor(proprio0, dtype=torch.float32), norm_traj


# -----------------------------------------------------------------------------
# 3. Flow Matching Training
# -----------------------------------------------------------------------------
def train(epochs=200, batch_size=16, lr=1.8e-3):
    print("=" * 68, flush=True)
    print("   SmolVLA Multimodal Flow-Matching Policy Training", flush=True)
    print("   - ViT Patch Encoder (256 Patches) & Subword Tokenizer", flush=True)
    print("   - Full Multi-Head Self & Cross-Attention Fusion Layers", flush=True)
    print(f"   - Hardware Compute Engine: {DEVICE} ({'CUDA GPU Acceleration' if DEVICE.type == 'cuda' else 'Optimized Multi-core CPU'})", flush=True)
    print("=" * 68, flush=True)

    dataset = SmolVLADataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmolVLAPolicy(vocab_size=len(dataset.tokenizer.vocab), d_model=128, num_layers=2).to(DEVICE)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss(reduction='none')

    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")
    best_loss = float("inf")

    print(f"\n>> Training SmolVLA Policy across {len(dataset)} demonstrations ({epochs} epochs)...", flush=True)
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training SmolVLA")

    try:
        for epoch in epoch_pbar:
            total_loss = 0.0
            for img, token_ids, proprio, x_1 in dataloader:
                img = img.to(DEVICE)
                token_ids = token_ids.to(DEVICE)
                proprio = proprio.to(DEVICE)
                x_1 = x_1.to(DEVICE)

                B = img.size(0)
                optimizer.zero_grad(set_to_none=True)

                x_0 = torch.randn_like(x_1)
                t = torch.rand(B, device=DEVICE)
                t_expanded = t.view(B, 1, 1)

                x_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1
                u_t = x_1 - x_0

                v_pred = model.forward_flow(x_t, t, img, token_ids, proprio=proprio)

                loss_raw = loss_fn(v_pred, u_t)
                dim_weights = torch.tensor([1.2, 1.2, 1.5, 2.5], device=DEVICE).view(1, 1, 4)
                weighted_loss = (loss_raw * dim_weights).mean()

                weighted_loss.backward()
                optimizer.step()
                total_loss += weighted_loss.item() * B

            scheduler.step()
            avg_loss = total_loss / len(dataset)

            if avg_loss < best_loss or epoch % 20 == 0:
                best_loss = min(best_loss, avg_loss)
                safe_save_model(model.state_dict(), model_path)

            current_lr = scheduler.get_last_lr()[0]
            epoch_pbar.set_postfix({
                "OT_Loss": f"{avg_loss:.5f}",
                "Best": f"{best_loss:.5f}",
                "LR": f"{current_lr:.6f}"
            })

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Saving checkpoint...", flush=True)
        safe_save_model(model.state_dict(), model_path)
        return

    safe_save_model(model.state_dict(), model_path)
    print(f"\n[SUCCESS] SmolVLA Model Checkpoint saved -> {model_path}", flush=True)

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    train(epochs=epochs)
