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
import clip

sys.path.append(os.path.dirname(__file__))
from smolvla_embedding import SmolVLMTokenizer
from smolvla_model import SmolVLABackbone, PretrainedVLMEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    NUM_CORES = os.cpu_count() or 4
    torch.set_num_threads(NUM_CORES)
    torch.set_num_interop_threads(NUM_CORES)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "demonstrations")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "vlm_features_cache.pt")
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
# 1. SmolVLA Policy with Real Pretrained VLM Backbone
# -----------------------------------------------------------------------------
class SmolVLAPolicy(nn.Module):
    """
    Genuine SmolVLA Policy:
    - Backbone: Pretrained OpenAI CLIP ViT-B/32 (49 Visual Patches + 77 BPE Language Tokens)
    - Projections into d_model=128 for real-time inference & training
    - Multimodal Cross-Attention Action Decoder over Horizon H=128
    """
    def __init__(self, horizon=128, action_dim=4, d_model=128, num_layers=2):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.d_model = d_model

        # 1. Real Pretrained VLM Backbone
        self.backbone = SmolVLABackbone(d_model=d_model, nhead=4, num_layers=num_layers)

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

        # 3. Action Decoder Queries & 2-Layer Transformer Cross-Attention Head
        self.action_in_proj = nn.Linear(action_dim, d_model)
        self.pos_queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.grounded_proj = nn.Linear(2, d_model)

        self.dec_sa1 = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.dec_ca1 = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.dec_n1 = nn.LayerNorm(d_model)
        self.dec_n2 = nn.LayerNorm(d_model)
        self.dec_n3 = nn.LayerNorm(d_model)
        self.dec_ffn1 = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

        self.dec_sa2 = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.dec_ca2 = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.dec_n4 = nn.LayerNorm(d_model)
        self.dec_n5 = nn.LayerNorm(d_model)
        self.dec_n6 = nn.LayerNorm(d_model)
        self.dec_ffn2 = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

        # Output Flow Vector Field
        self.out_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim)
        )

    def forward_from_embeddings(self, x_t, t, vis_patches, text_feats, proprio=None):
        B = vis_patches.size(0)

        # Project 512 -> 128
        cur_txt = self.backbone.txt_proj(text_feats[:, :16])
        cur_vis = self.backbone.vis_proj(vis_patches)

        # Cross attention layers
        for layer in self.backbone.layers:
            cur_txt, cur_vis, _ = layer(cur_txt, cur_vis)

        # Soft-argmax 2D coordinate extraction
        txt_q = self.backbone.grounding_head(cur_txt.mean(dim=1))
        attn_logits = torch.bmm(txt_q.unsqueeze(1), cur_vis.transpose(1, 2)) / (self.d_model ** 0.5)
        spatial_probs = F.softmax(attn_logits * 4.0, dim=-1)
        grid_x = self.backbone.vlm.grid_x.to(vis_patches.device)
        grid_y = self.backbone.vlm.grid_y.to(vis_patches.device)
        gx = (spatial_probs * grid_x).sum(dim=-1)
        gy = (spatial_probs * grid_y).sum(dim=-1)
        grounded_2d = torch.cat([gx, gy], dim=-1)

        # Conditioning
        if proprio is None:
            proprio = torch.zeros(B, 5, device=vis_patches.device)
        p_token = self.proprio_proj(proprio).unsqueeze(1)
        t_token = self.time_embed(t).unsqueeze(1)
        g_token = self.grounded_proj(grounded_2d).unsqueeze(1)

        # Joint Multimodal Context: 1 + 1 + 1 + 16 + 49 = 68 tokens
        context = torch.cat([g_token, p_token, t_token, cur_txt, cur_vis], dim=1)

        # Action Decoder Stream
        act_queries = self.action_in_proj(x_t) + self.pos_queries

        # Layer 1
        sa_out, _ = self.dec_sa1(act_queries, act_queries, act_queries)
        act_queries = self.dec_n1(act_queries + sa_out)
        ca_out, _ = self.dec_ca1(act_queries, context, context)
        act_queries = self.dec_n2(act_queries + ca_out)
        act_queries = self.dec_n3(act_queries + self.dec_ffn1(act_queries))

        # Layer 2
        sa_out2, _ = self.dec_sa2(act_queries, act_queries, act_queries)
        act_queries = self.dec_n4(act_queries + sa_out2)
        ca_out2, _ = self.dec_ca2(act_queries, context, context)
        act_queries = self.dec_n5(act_queries + ca_out2)
        act_queries = self.dec_n6(act_queries + self.dec_ffn2(act_queries))

        v_pred = self.out_head(act_queries)
        return v_pred, grounded_2d

    def forward_flow(self, x_t, t, img, token_ids, proprio=None):
        with torch.no_grad():
            vis_patches, _ = self.backbone.vlm.encode_vision(img)
            text_feats = self.backbone.vlm.encode_text(token_ids)
        return self.forward_from_embeddings(x_t, t, vis_patches, text_feats, proprio=proprio)

    @torch.no_grad()
    def sample(self, img, token_ids, proprio=None, num_steps=15):
        B = img.size(0)
        x = torch.randn(B, self.horizon, self.action_dim, device=img.device)
        dt = 1.0 / num_steps

        vis_patches, _ = self.backbone.vlm.encode_vision(img)
        text_feats = self.backbone.vlm.encode_text(token_ids)

        for i in range(num_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=img.device)
            v, _ = self.forward_from_embeddings(x, t, vis_patches, text_feats, proprio=proprio)
            x = x + v * dt

        raw_x = x * ACTION_STD.to(img.device) + ACTION_MEAN.to(img.device)
        kernel = torch.ones(1, 1, 5, device=img.device) / 5.0
        raw_perm = raw_x.permute(0, 2, 1).reshape(B * self.action_dim, 1, self.horizon)
        smoothed = F.conv1d(raw_perm, kernel, padding=2)
        smoothed = smoothed.view(B, self.action_dim, self.horizon).permute(0, 2, 1)
        return smoothed


# -----------------------------------------------------------------------------
# 2. Fast Cached Dataset
# -----------------------------------------------------------------------------
class FastSmolVLADataset(Dataset):
    def __init__(self, data_dir, cache_file=CACHE_FILE):
        if not os.path.exists(cache_file):
            raise ValueError(f"Cache file {cache_file} missing!")
            
        print(f">> Loading Pretrained VLM features from {cache_file}...", flush=True)
        cache = torch.load(cache_file, map_location="cpu")
        files = cache['files']
        cached_patches = cache['patches']
        cached_txt = cache['txt_feats']

        self.samples = []
        for idx, f in enumerate(files):
            d = np.load(f, allow_pickle=True)
            proprio = d['proprioception'].astype(np.float32)
            acts = d['actions'].astype(np.float32)

            raw_traj = np.concatenate([proprio[:, :3], acts[:, 4:5]], axis=-1)
            norm_traj = (torch.tensor(raw_traj, dtype=torch.float32) - ACTION_MEAN) / (ACTION_STD + 1e-6)
            proprio0 = proprio[0]

            if 'observations' in d and len(d['observations']) > 0:
                cx_phys, cy_phys = d['observations'][0][5:7]
                px = 32.0 + (cy_phys / 0.28) * 28.0
                py = 58.0 - ((cx_phys - 0.10) / 0.25) * 52.0
                target_2d = np.array([(px - 31.5) / 31.5, (py - 31.5) / 31.5], dtype=np.float32)
            else:
                target_2d = np.zeros(2, dtype=np.float32)

            self.samples.append((
                cached_patches[idx],
                cached_txt[idx],
                torch.tensor(proprio0, dtype=torch.float32),
                norm_traj,
                torch.tensor(target_2d, dtype=torch.float32)
            ))

        print(f">> Loaded {len(self.samples)} cached demonstration trajectories.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# -----------------------------------------------------------------------------
# 3. Flow Matching Training
# -----------------------------------------------------------------------------
def train(epochs=150, batch_size=16, lr=1.5e-3):
    print("=" * 68, flush=True)
    print("   SmolVLA Policy with Real Pretrained VLM (OpenAI CLIP ViT-B/32)", flush=True)
    print("   - Genuine 49 ViT Patch Embeddings + 77 Pretrained Language Tokens", flush=True)
    print("   - Multimodal Cross-Attention Action Decoder + Spatial Grounding", flush=True)
    print(f"   - Hardware Compute Engine: {DEVICE}", flush=True)
    print("=" * 68, flush=True)

    dataset = FastSmolVLADataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmolVLAPolicy(d_model=128, num_layers=2).to(DEVICE)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f">> Trainable Policy Parameters: {sum(p.numel() for p in trainable_params):,}", flush=True)

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss(reduction='none')

    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")
    best_loss = float("inf")

    print(f"\n>> Training Pretrained SmolVLA Policy across {len(dataset)} demonstrations ({epochs} epochs)...", flush=True)
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training SmolVLA")

    try:
        for epoch in epoch_pbar:
            total_loss = 0.0
            for vis_patches, txt_feats, proprio, x_1, target_2d in dataloader:
                vis_patches = vis_patches.to(DEVICE)
                txt_feats = txt_feats.to(DEVICE)
                proprio = proprio.to(DEVICE)
                x_1 = x_1.to(DEVICE)
                target_2d = target_2d.to(DEVICE)

                B = vis_patches.size(0)
                optimizer.zero_grad(set_to_none=True)

                x_0 = torch.randn_like(x_1)
                t = torch.rand(B, device=DEVICE)
                t_expanded = t.view(B, 1, 1)

                x_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1
                u_t = x_1 - x_0

                v_pred, pred_grounded_2d = model.forward_from_embeddings(x_t, t, vis_patches, txt_feats, proprio=proprio)

                # 1. Flow-matching velocity vector loss
                loss_raw = loss_fn(v_pred, u_t)
                dim_weights = torch.tensor([1.2, 1.2, 1.5, 2.5], device=DEVICE).view(1, 1, 4)
                flow_loss = (loss_raw * dim_weights).mean()

                # 2. Auxiliary Spatial Grounding Loss
                grounding_loss = F.mse_loss(pred_grounded_2d, target_2d)

                total_batch_loss = flow_loss + 0.30 * grounding_loss
                total_batch_loss.backward()
                optimizer.step()
                total_loss += total_batch_loss.item() * B

            scheduler.step()
            avg_loss = total_loss / len(dataset)

            if avg_loss < best_loss or epoch % 10 == 0:
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
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    train(epochs=epochs)
