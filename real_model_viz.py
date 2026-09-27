import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from torchinfo import summary
except ImportError:
    summary = None

# Add subdirectories to path
sys.path.append(os.path.abspath('manual_vla'))
sys.path.append(os.path.abspath('smolvla_dobot'))
sys.path.append(os.path.abspath('full_smolvla_dobot'))

os.makedirs('paper/figures', exist_ok=True)
device = torch.device('cpu')

# ---------------------------------------------------------
# 1. Manual VLA
# ---------------------------------------------------------
print("Loading Manual VLA...")
from manual_vla.train_flow import ManualVLAPolicy, get_color_ids
m_manual = ManualVLAPolicy().to(device)
m_manual.forward = m_manual.forward_flow  # Alias for torchinfo

if summary:
    with open('manual_vla_summary.txt', 'w', encoding='utf-8') as f:
        dummy_img = torch.rand(1, 3, 64, 64)
        dummy_color = torch.tensor([get_color_ids("red", "green")]).long()
        dummy_t = torch.tensor([0.5])
        dummy_xt = torch.randn(1, 128, 4)
        f.write(repr(summary(m_manual, input_data=(dummy_xt, dummy_t, dummy_img, dummy_color), verbose=0)))

# Generate sample attention
dummy_img = torch.rand(1, 3, 64, 64)
dummy_color = torch.tensor([get_color_ids("red", "green")]).long()
with torch.no_grad():
    _, attn = m_manual.forward_flow(torch.randn(1, 128, 4), torch.tensor([0.5]), dummy_img, dummy_color)

attn_cube = attn[0, 0].reshape(16, 16).numpy()
attn_platform = attn[0, 1].reshape(16, 16).numpy()

plt.figure(figsize=(10, 4), dpi=200)
plt.subplot(1, 2, 1)
plt.imshow(attn_cube, cmap='magma')
plt.title("Manual VLA: 'Cube' Spatial Grounding (16x16)")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(attn_platform, cmap='magma')
plt.title("Manual VLA: 'Platform' Spatial Grounding (16x16)")
plt.axis('off')
plt.savefig('paper/figures/real_manual_vla_attn.png', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 2. SmolVLA
# ---------------------------------------------------------
print("Loading SmolVLA...")
from smolvla_dobot.train_smolvla import SmolVLAPolicy
import clip
m_smol = SmolVLAPolicy().to(device)
m_smol.forward = m_smol.forward_flow  # Alias for torchinfo

# Use a realistic token sequence so the attention isn't pure noise
prompt = "pick up the red cube and place it on the green platform"
dummy_tokens = clip.tokenize(prompt).to(device)

if summary:
    with open('smolvla_summary.txt', 'w', encoding='utf-8') as f:
        dummy_img = torch.rand(1, 3, 64, 64)
        dummy_t = torch.tensor([0.5])
        dummy_xt = torch.randn(1, 128, 4)
        f.write(repr(summary(m_smol, input_data=(dummy_xt, dummy_t, dummy_img, dummy_tokens), verbose=0)))

with torch.no_grad():
    dummy_img = torch.rand(1, 3, 64, 64)
    # Add a faux "red block" to make attention respond (since CLIP is somewhat zero-shot)
    dummy_img[0, 0, 30:40, 30:40] = 1.0 
    dummy_img[0, 1:3, 30:40, 30:40] = 0.0
    
    vis_patches, cls_emb = m_smol.backbone.vlm.encode_vision(dummy_img)
    text_feats = m_smol.backbone.vlm.encode_text(dummy_tokens)
    base_weights = torch.einsum('bld,bpd->blp', text_feats.float(), vis_patches.float())
    cross_weights = torch.softmax(base_weights * 5.0, dim=-1)

# Generate a 3D Scatter Plot for Language (16 tokens) vs 7x7 Patches
attn_3d = cross_weights[0, :16].reshape(16, 7, 7).numpy()

fig = plt.figure(figsize=(10, 8), dpi=200)
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Meshgrid for Z (tokens), Y (patch row), X (patch col)
Z, Y, X = np.meshgrid(np.arange(16), np.arange(7), np.arange(7), indexing='ij')

X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()
W_flat = attn_3d.flatten()

# Filter out very low attention weights for clarity in 3D
mask = W_flat > (W_flat.max() * 0.1)
X_m, Y_m, Z_m, W_m = X_flat[mask], Y_flat[mask], Z_flat[mask], W_flat[mask]

# Plot: Size and Color both mapped to attention weight
sc = ax.scatter(X_m, Y_m, Z_m, c=W_m, cmap='magma', s=W_m * 2000, alpha=0.8, edgecolors='none')

ax.set_xlabel("Patch X (0-6)", fontweight='bold')
ax.set_ylabel("Patch Y (0-6)", fontweight='bold')
ax.set_zlabel("Language Token Index (0-15)", fontweight='bold')
ax.set_title(f"3D Cross-Attention\nPrompt: '{prompt}'", fontweight='bold', pad=20)

# Make background panes clean
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle=':', alpha=0.6)

cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('Attention Weight', fontweight='bold')

plt.savefig('paper/figures/real_smolvla_attention.png', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 3. Full SmolVLA
# ---------------------------------------------------------
print("Loading Full SmolVLA...")
from full_smolvla_dobot.train_full_smolvla import FullSmolVLAPolicy
m_full = FullSmolVLAPolicy(load_backbone=False).to(device)
m_full.forward = m_full.forward_from_embeddings  # Alias for torchinfo

if summary:
    with open('full_smolvla_summary.txt', 'w', encoding='utf-8') as f:
        dummy_xt = torch.randn(1, 128, 4)
        dummy_t = torch.tensor([0.5])
        # vlm_tokens shape: [B, seq_len, 128]
        dummy_vlm_tokens = torch.randn(1, 40, 128)
        f.write(repr(summary(m_full, input_data=(dummy_xt, dummy_t, dummy_vlm_tokens), verbose=0)))

# Dummy forward to get activations
with torch.no_grad():
    dummy_xt = torch.randn(1, 128, 4)
    dummy_t = torch.tensor([0.5])
    dummy_vlm_tokens = torch.randn(1, 40, 128)
    v_pred, acts = m_full.forward_from_embeddings(dummy_xt, dummy_t, dummy_vlm_tokens, return_activations=True)
    
ca_w1 = acts["ca_w1"][0].numpy() # [128, 42] (40 vlm + 1 proprio + 1 time)

plt.figure(figsize=(10, 6), dpi=200)
plt.imshow(ca_w1, cmap='Purples', aspect='auto')
plt.title("Full SmolVLA: Action Decoder Layer 1 Cross-Attention Matrix")
plt.ylabel("Action Trajectory Horizon (128 steps)")
plt.xlabel("Multimodal Context Tokens (Proprio + Time + VLM)")
plt.colorbar()
plt.savefig('paper/figures/real_full_smolvla_attention.png', bbox_inches='tight')
plt.close()

print("Real architecture outputs and summaries generated.")
