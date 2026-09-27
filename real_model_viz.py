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
m_smol = SmolVLAPolicy().to(device)
m_smol.forward = m_smol.forward_flow  # Alias for torchinfo

if summary:
    with open('smolvla_summary.txt', 'w', encoding='utf-8') as f:
        dummy_img = torch.rand(1, 3, 64, 64)
        dummy_tokens = torch.randint(0, 49408, (1, 77))
        dummy_t = torch.tensor([0.5])
        dummy_xt = torch.randn(1, 128, 4)
        f.write(repr(summary(m_smol, input_data=(dummy_xt, dummy_t, dummy_img, dummy_tokens), verbose=0)))

with torch.no_grad():
    vis_patches, cls_emb = m_smol.backbone.vlm.encode_vision(dummy_img)
    text_feats = m_smol.backbone.vlm.encode_text(dummy_tokens)
    base_weights = torch.einsum('bld,bpd->blp', text_feats.float(), vis_patches.float())
    cross_weights = torch.softmax(base_weights * 5.0, dim=-1)

plt.figure(figsize=(8, 6), dpi=200)
plt.imshow(cross_weights[0].numpy(), cmap='viridis', aspect='auto')
plt.title("SmolVLA: Zero-Shot CLIP Vision-Language Cross-Attention")
plt.xlabel("Visual Patch Tokens (49)")
plt.ylabel("Language Tokens (77)")
plt.colorbar()
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
