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

# Multi-core CPU Configuration
DEVICE = torch.device("cpu")
NUM_CORES = os.cpu_count() or 4
torch.set_num_threads(NUM_CORES)
torch.set_num_interop_threads(NUM_CORES)
os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)
os.environ["MKL_NUM_THREADS"] = str(NUM_CORES)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "demos")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Parameterized Embeddings (Without heavy LLM overhead: intent, source, dest)
# -----------------------------------------------------------------------------
COLOR_MAP = {
    "red": 0, "blue": 1, "yellow": 2, "green": 3,
    "purple": 4, "orange": 5, "cyan": 6
}
ACTION_MAP = {"pick_place": 0, "push": 1}

def get_intent_embedding_vector(act_str="pick_place", src_str="red", dst_str="green", dim=128):
    """
    Constructs a 128-dim structured continuous parameter embedding:
    - Action type (Pick & Place vs Push)
    - Source object color/target
    - Destination platform color/target
    """
    act_idx = ACTION_MAP.get(act_str, 0)
    src_idx = COLOR_MAP.get(src_str, 0)
    dst_idx = COLOR_MAP.get(dst_str, 3)

    act_onehot = np.zeros(4, dtype=np.float32)
    act_onehot[act_idx] = 1.0

    src_onehot = np.zeros(8, dtype=np.float32)
    src_onehot[src_idx] = 1.0

    dst_onehot = np.zeros(8, dtype=np.float32)
    dst_onehot[dst_idx] = 1.0

    raw_vec = np.concatenate([act_onehot, src_onehot, dst_onehot]) # 20 dims
    return raw_vec

# Normalization constants for 4D actions (X, Y, Z, Gripper) matching empirical dataset
ACTION_MEAN = torch.tensor([0.2066, 0.0024, 0.0738, 0.2617], dtype=torch.float32)
ACTION_STD  = torch.tensor([0.0326, 0.0816, 0.0405, 0.4396], dtype=torch.float32)

# -----------------------------------------------------------------------------
# 1. Optimal Transport Dataset
# -----------------------------------------------------------------------------
class OTFlowMatchingDataset(Dataset):
    def __init__(self, data_dir):
        files = sorted(glob.glob(os.path.join(data_dir, "demo_*.npz")))
        if not files:
            raise ValueError(f"No demonstration files found in {data_dir}. Run auto_generate_demos.py first!")

        self.samples = []
        print(f">> Indexing {len(files)} demonstrations for Flow-Matching...", flush=True)

        for f in files:
            d = np.load(f, allow_pickle=True)
            img0 = d['images'][0].astype(np.float32) # [3, 64, 64]
            proprio = d['proprioception'].astype(np.float32) # [128, 5]
            acts = d['actions'].astype(np.float32) # [128, 6]
            
            act_type = str(d['action_type'][0]) if 'action_type' in d else "pick_place"
            tgt_color = str(d['target_color'][0]) if 'target_color' in d else "red"
            tgt_plat = str(d['target_plat_color'][0]) if 'target_plat_color' in d else "green"

            intent_vec = get_intent_embedding_vector(act_type, tgt_color, tgt_plat)

            raw_traj = np.concatenate([proprio[:, :3], acts[:, 4:5]], axis=-1) # [128, 4]
            norm_traj = (torch.tensor(raw_traj, dtype=torch.float32) - ACTION_MEAN) / (ACTION_STD + 1e-6)
            self.samples.append((img0, intent_vec, norm_traj))

        print(f">> Successfully indexed {len(self.samples)} trajectory demonstrations.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, intent_vec, norm_traj = self.samples[idx]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(intent_vec, dtype=torch.float32), norm_traj

# -----------------------------------------------------------------------------
# 2. Vision Patch & Flow Matching Policy Architecture
# -----------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x):
        return x + self.conv(x)

class CoordConv(nn.Module):
    """Appends normalized (x, y) spatial coordinate channels [-1, 1] to input feature maps."""
    def forward(self, x):
        B, _, H, W = x.size()
        xx_channel = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        yy_channel = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        return torch.cat([x, xx_channel, yy_channel], dim=1)

class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax layer that converts 2D feature maps directly into continuous
    sub-pixel (x, y) coordinate keypoints, preserving exact object positions.
    """
    def __init__(self, temperature=None):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        B, C, H, W = features.shape
        if self.temperature is not None:
            features = features / self.temperature

        # Compute spatial softmax across (H, W)
        features_flat = features.view(B, C, H * W)
        softmax_attention = F.softmax(features_flat, dim=-1).view(B, C, H, W)

        # Coordinate grid [-1, 1]
        pos_x = torch.linspace(-1, 1, W, device=features.device).view(1, 1, 1, W)
        pos_y = torch.linspace(-1, 1, H, device=features.device).view(1, 1, H, 1)

        expected_x = torch.sum(softmax_attention * pos_x, dim=(-2, -1)) # [B, C]
        expected_y = torch.sum(softmax_attention * pos_y, dim=(-2, -1)) # [B, C]

        keypoints = torch.cat([expected_x, expected_y], dim=-1) # [B, 2*C]
        return keypoints

class ManualVLAPolicy(nn.Module):
    """
    Lightweight Vision-Action Policy with Spatial Softmax + Hadamard Conditioning:
    - High-Resolution CoordConv ResNet Visual Extractor
    - Spatial Softmax Keypoint Bottleneck (preserves sub-pixel continuous coordinates)
    - Intent Embedder (No heavy LLM needed; takes action, src, dst embeddings)
    - Hadamard Product Feature Modulation (Image Patches * Intent Tokens)
    - Continuous Optimal Transport Flow Matching Vector Field
    """
    def __init__(self, raw_intent_dim=20, horizon=128, action_dim=4):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.total_act_dim = horizon * action_dim

        # 1. Vision Patch / ResBlock Encoder with CoordConv & Spatial Softmax
        self.coord_conv = CoordConv()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3 + 2, 32, kernel_size=3, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 16x16
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.spatial_softmax = SpatialSoftmax() # 64 channels * 2 (x, y) = 128 keypoints
        self.v_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )

        # 2. Intent Parameter MLP
        self.intent_mlp = nn.Sequential(
            nn.Linear(raw_intent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256)
        )

        # 3. Continuous Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )

        # 4. Flow Matching Vector Field Network
        # Condition size: 256 (Hadamard vision*intent) + 256 (intent) = 512 + 128 (time) = 640
        self.flow_net = nn.Sequential(
            nn.Linear(self.total_act_dim + 512 + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, self.total_act_dim)
        )

    def extract_visual_features(self, img):
        img_coord = self.coord_conv(img)
        feat_map = self.conv_stem(img_coord)
        keypoints = self.spatial_softmax(feat_map)
        return self.v_proj(keypoints)

    def forward_flow(self, x_t, t, img, intent_vec):
        B = img.size(0)
        v_feat = self.extract_visual_features(img)            # [B, 256]
        i_feat = self.intent_mlp(intent_vec)                 # [B, 256]

        # Hadamard modulation on visual features
        modulated_v = v_feat * i_feat                        # [B, 256]
        cond = torch.cat([modulated_v, i_feat], dim=-1)      # [B, 512]

        t_feat = self.time_embed(t)                          # [B, 128]
        x_flat = x_t.reshape(B, -1)                          # [B, 512]

        inp = torch.cat([x_flat, cond, t_feat], dim=-1)
        v_pred = self.flow_net(inp)
        return v_pred.reshape(B, self.horizon, self.action_dim)

    @torch.no_grad()
    def sample(self, img, intent_vec, num_steps=20):
        """Continuous Euler ODE integration from noise to predicted robot trajectory."""
        B = img.size(0)
        x = torch.randn(B, self.horizon, self.action_dim, device=img.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=img.device)
            v = self.forward_flow(x, t, img, intent_vec)
            x = x - v * dt

        raw_x = x * ACTION_STD.to(img.device) + ACTION_MEAN.to(img.device)
        
        # Temporal smoothing filter across horizon to eliminate high-frequency micro-jitter
        # Box filter smoothing over window of 5 timesteps
        kernel = torch.ones(1, 1, 5, device=img.device) / 5.0
        # raw_x shape: [B, horizon, 4] -> permute to [B*4, 1, horizon]
        raw_perm = raw_x.permute(0, 2, 1).reshape(B * self.action_dim, 1, self.horizon)
        smoothed = F.conv1d(raw_perm, kernel, padding=2)
        smoothed = smoothed.view(B, self.action_dim, self.horizon).permute(0, 2, 1)

        return smoothed

# Aliases
TrueSmolVLAPolicy = ManualVLAPolicy

# -----------------------------------------------------------------------------
# 3. Flow Matching Training
# -----------------------------------------------------------------------------
def train(epochs=120, batch_size=16, lr=1.8e-3):
    print("=" * 68, flush=True)
    print("   Manual VLA Flow-Matching Policy Training (Hadamard MLP)", flush=True)
    print("   - Action Conditioning: Pick & Place vs Push", flush=True)
    print("   - Patch Visual Extractor & Parameterized Embeddings", flush=True)
    print("   - Optimal Transport Vector Field Regression | 100% CPU", flush=True)
    print("=" * 68, flush=True)

    dataset = OTFlowMatchingDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ManualVLAPolicy()
    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.SmoothL1Loss(reduction='none') # Huber loss
    # Weighting: X, Y, Z spatial accuracy + 2.5x boosting on gripper grasp/release transitions
    dim_weights = torch.tensor([1.0, 1.0, 1.2, 2.5], device=DEVICE).view(1, 1, 4)

    print(f"\n>> Training Flow-Matching across {len(dataset)} trajectories ({epochs} epochs)...", flush=True)
    best_loss = float('inf')

    try:
        epoch_pbar = tqdm(range(1, epochs + 1), desc="Training VLA", unit="epoch", dynamic_ncols=True)
        for epoch in epoch_pbar:
            model.train()
            total_loss = 0.0

            for img_b, intent_b, traj_x0 in dataloader:
                B = img_b.size(0)
                optimizer.zero_grad(set_to_none=True)

                x_1 = torch.randn_like(traj_x0)
                t = torch.rand(B, device=img_b.device)
                t_expand = t.view(B, 1, 1)

                # Optimal Transport path interpolation
                x_t = (1.0 - t_expand) * traj_x0 + t_expand * x_1
                target_v = x_1 - traj_x0

                pred_v = model.forward_flow(x_t, t, img_b, intent_vec=intent_b)
                # Huber loss with gripper boosting
                raw_loss = loss_fn(pred_v, target_v) # [B, horizon, 4]
                weighted_loss = (raw_loss * dim_weights).mean()
                
                weighted_loss.backward()
                optimizer.step()
                total_loss += weighted_loss.item() * B

            scheduler.step()
            avg_loss = total_loss / len(dataset)

            if avg_loss < best_loss or epoch % 20 == 0:
                best_loss = min(best_loss, avg_loss)
                torch.save(model.state_dict(), model_path)

            current_lr = scheduler.get_last_lr()[0]
            epoch_pbar.set_postfix({
                "OT_Loss": f"{avg_loss:.5f}",
                "Best": f"{best_loss:.5f}",
                "LR": f"{current_lr:.6f}"
            })

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Saving checkpoint...", flush=True)
        torch.save(model.state_dict(), model_path)
        return

    torch.save(model.state_dict(), model_path)
    print(f"\n[SUCCESS] Manual VLA checkpoint saved -> {model_path}", flush=True)

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    train(epochs=epochs)
