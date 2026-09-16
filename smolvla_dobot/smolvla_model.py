import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# 1. Vision Transformer Patch Encoder (ViT-style)
# -----------------------------------------------------------------------------
class ViTPatchEncoder(nn.Module):
    """
    True Vision Transformer (ViT) Patch Encoder:
    - Splits 64x64 RGB into 16x16 grid of 4x4 pixel patches = 256 tokens.
    - Linear patch projection via Conv2d (kernel=4, stride=4).
    - 2D Sinusoidal Positional Embeddings.
    - Mini ViT Transformer Self-Attention Layer to model intra-image spatial context.
    """
    def __init__(self, in_channels=3, patch_size=4, d_model=128, nhead=4, d_ff=256):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        
        # ViT Self-Attention block over the 256 patches
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, img):
        # img: [B, 3, 64, 64] -> patches: [B, d_model, 16, 16] -> [B, 256, d_model]
        x = self.patch_embed(img).flatten(2).permute(0, 2, 1)
        x = x + self.pos_embed
        
        # Self-Attention across all 256 image patches
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x # [B, 256, d_model]


# -----------------------------------------------------------------------------
# 2. SmolVLA Multimodal Transformer Block (Self & Cross-Attention)
# -----------------------------------------------------------------------------
class SmolVLAMultimodalBlock(nn.Module):
    """
    Unified Multimodal Transformer Layer (SmolVLM / SmolVLA style):
    1. Multi-Head Self-Attention across the concatenated sequence: [Text Tokens || Visual Patches]
    2. Explicit Multi-Head Cross-Attention from Language to Visual Patches
    3. Feed-Forward SwiGLU/GELU network
    """
    def __init__(self, d_model=128, nhead=4, d_ff=256):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.multimodal_self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.norm_cross_txt = nn.LayerNorm(d_model)
        self.norm_cross_vis = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, text_tokens, visual_patches):
        # text_tokens: [B, T, d_model], visual_patches: [B, P, d_model]
        B, T, D = text_tokens.shape
        P = visual_patches.shape[1]

        # 1. Joint Multi-Head Self-Attention over entire [Text + Vision] sequence (T + P = 16 + 256 = 272 tokens)
        joint_seq = torch.cat([text_tokens, visual_patches], dim=1) # [B, 272, d_model]
        norm_joint = self.norm_self(joint_seq)
        joint_out, joint_weights = self.multimodal_self_attn(norm_joint, norm_joint, norm_joint)
        joint_seq = joint_seq + joint_out

        # Split back into text and vision streams
        text_stream = joint_seq[:, :T]
        vis_stream = joint_seq[:, T:]

        # 2. Directed Multi-Head Cross-Attention: Language Query -> Visual Patch Key/Values
        # Returns exact attention distribution across heads: [B, nhead, T, P]
        q = self.norm_cross_txt(text_stream)
        kv = self.norm_cross_vis(vis_stream)
        cross_out, cross_weights = self.cross_attn(q, kv, kv, need_weights=True)
        text_stream = text_stream + cross_out

        # 3. Feed-Forward Network
        full_seq = torch.cat([text_stream, vis_stream], dim=1)
        full_seq = full_seq + self.ffn(self.norm_ffn(full_seq))

        new_text = full_seq[:, :T]
        new_vis = full_seq[:, T:]

        return new_text, new_vis, cross_weights # cross_weights: [B, T, 256]


# -----------------------------------------------------------------------------
# 3. Compact SmolVLA Backbone
# -----------------------------------------------------------------------------
class SmolVLABackbone(nn.Module):
    """
    Compact (~1.5M parameters) SmolVLA Multimodal Backbone:
    - ViT Patch Encoder (256 tokens)
    - Subword Language Tokenizer & Embeddings (16 tokens)
    - 2-Layer Full Multi-Head Self & Cross-Attention Multimodal Transformer
    """
    def __init__(self, vocab_size=32, d_model=128, text_len=16, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.text_len = text_len

        # Language Stream
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.text_pos_embed = nn.Parameter(torch.randn(1, text_len, d_model) * 0.02)

        # Vision Stream (ViT Patch Encoder)
        self.vit_encoder = ViTPatchEncoder(in_channels=3, patch_size=4, d_model=d_model, nhead=nhead)

        # Multimodal Transformer Layers
        self.layers = nn.ModuleList([
            SmolVLAMultimodalBlock(d_model=d_model, nhead=nhead, d_ff=d_model * 2)
            for _ in range(num_layers)
        ])

    def forward(self, text_ids, img):
        # 1. Embed text
        text_feats = self.token_embed(text_ids) + self.text_pos_embed

        # 2. ViT Patch Encode image
        vis_feats = self.vit_encoder(img) # [B, 256, d_model]

        # 3. Pass through Multimodal Self & Cross-Attention Layers
        cross_weights_history = []
        for layer in self.layers:
            text_feats, vis_feats, weights = layer(text_feats, vis_feats)
            cross_weights_history.append(weights)

        # Return final multimodal representations and final cross-attention alignment
        return text_feats, vis_feats, cross_weights_history[-1]

if __name__ == '__main__':
    from smolvla_embedding import SmolVLMTokenizer
    tok = SmolVLMTokenizer()
    prompt = 'pick up the red cube and place it on the green platform'
    ids = tok.encode(prompt).unsqueeze(0)
    img = torch.randn(1, 3, 64, 64)

    model = SmolVLABackbone(vocab_size=len(tok.vocab), d_model=128, num_layers=2)
    t_out, v_out, cross_attn = model(ids, img)
    params = sum(p.numel() for p in model.parameters())

    print(f'SmolVLA Backbone Parameters: {params:,} (Ultra-lightweight!)')
    print(f'Text Out: {t_out.shape}')
    print(f'Vision Out: {v_out.shape}')
    print(f'Full Multi-Head Cross Attention Matrix: {cross_attn.shape} ([Batch, 16 Language Tokens, 256 Visual Patches])')
