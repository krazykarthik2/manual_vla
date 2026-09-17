import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. Spatial Coordinate Convolution (CoordConv) & Feature Pyramidal Backbone
# -----------------------------------------------------------------------------
class SpatialGroundedVisualEncoder(nn.Module):
    """
    Spatially Grounded Visual Backbone (CoordConv + ConvNeXt-style Spatial Pyramid):
    - Injects explicit 2D coordinate meshgrids [-1, 1] as channels into the 64x64 RGB image.
    - Preserves 2D metric spatial geometry across spatial layers.
    - Yields 256 tokens corresponding to a 16x16 feature grid with exact spatial calibration.
    """
    def __init__(self, in_channels=3, d_model=128):
        super().__init__()
        self.d_model = d_model
        
        # Stage 1: [B, 5, 64, 64] -> [B, 64, 32, 32]
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels + 2, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        
        # Stage 2: [B, 64, 32, 32] -> [B, 128, 16, 16]
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU()
        )
        
        # Stage 3: Projection to d_model tokens
        self.proj = nn.Sequential(
            nn.Conv2d(128, d_model, kernel_size=1),
            nn.GroupNorm(16, d_model),
            nn.GELU()
        )

        # 16x16 grid coordinates buffer
        y_g, x_g = torch.meshgrid(torch.linspace(-1, 1, 16), torch.linspace(-1, 1, 16), indexing='ij')
        self.register_buffer('grid_x', x_g.reshape(1, 1, 256))
        self.register_buffer('grid_y', y_g.reshape(1, 1, 256))

    def forward(self, img):
        # img: [B, 3, 64, 64]
        B = img.size(0)
        y_c, x_c = torch.meshgrid(
            torch.linspace(-1, 1, 64, device=img.device),
            torch.linspace(-1, 1, 64, device=img.device),
            indexing='ij'
        )
        coords = torch.stack([x_c, y_c], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        x_in = torch.cat([img, coords], dim=1) # [B, 5, 64, 64]

        feat_map = self.proj(self.stage2(self.stage1(x_in))) # [B, d_model, 16, 16]
        vis_tokens = feat_map.flatten(2).permute(0, 2, 1)    # [B, 256, d_model]
        return vis_tokens, feat_map


# -----------------------------------------------------------------------------
# 2. SmolVLA Multimodal Cross-Attention Block
# -----------------------------------------------------------------------------
class SmolVLAMultimodalBlock(nn.Module):
    """
    Multimodal Transformer Layer:
    - Language Query -> Spatial Visual Patches Cross-Attention
    - Self-Attention across fused multimodal representations
    """
    def __init__(self, d_model=128, nhead=4, d_ff=256):
        super().__init__()
        self.norm_cross_txt = nn.LayerNorm(d_model)
        self.norm_cross_vis = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, text_tokens, visual_patches):
        # 1. Cross-Attention: Language Query attends to Visual Patches
        q = self.norm_cross_txt(text_tokens)
        kv = self.norm_cross_vis(visual_patches)
        cross_out, cross_weights = self.cross_attn(q, kv, kv, need_weights=True)
        text_tokens = text_tokens + cross_out

        # 2. Joint Self-Attention over sequence
        joint_seq = torch.cat([text_tokens, visual_patches], dim=1)
        norm_joint = self.norm_self(joint_seq)
        joint_out, _ = self.self_attn(norm_joint, norm_joint, norm_joint)
        joint_seq = joint_seq + joint_out

        # 3. Feed Forward
        joint_seq = joint_seq + self.ffn(self.norm_ffn(joint_seq))

        T = text_tokens.size(1)
        new_text = joint_seq[:, :T]
        new_vis = joint_seq[:, T:]
        return new_text, new_vis, cross_weights


# -----------------------------------------------------------------------------
# 3. Complete Spatially Grounded SmolVLA Backbone
# -----------------------------------------------------------------------------
class SmolVLABackbone(nn.Module):
    """
    Spatially Grounded SmolVLA Multimodal Backbone:
    - CoordConv Spatial Feature Pyramidal Encoder (256 tokens)
    - Subword Language Tokenizer & Positional Embeddings (16 tokens)
    - Multi-Head Language-Visual Cross-Attention Fusion
    - Differentiable Spatial Soft-Argmax for exact 2D Target Grounding
    """
    def __init__(self, vocab_size=32, d_model=128, text_len=16, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.text_len = text_len

        # Language Stream
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.text_pos_embed = nn.Parameter(torch.randn(1, text_len, d_model) * 0.02)

        # Spatial Grounded Vision Stream
        self.visual_encoder = SpatialGroundedVisualEncoder(in_channels=3, d_model=d_model)

        # Multimodal Transformer Fusion Layers
        self.layers = nn.ModuleList([
            SmolVLAMultimodalBlock(d_model=d_model, nhead=nhead, d_ff=d_model * 2)
            for _ in range(num_layers)
        ])

        # Color token queries for spatial object localization
        self.obj_query_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, text_ids, img):
        # 1. Embed text
        text_feats = self.token_embed(text_ids) + self.text_pos_embed

        # 2. Spatially grounded visual encoding
        vis_feats, feat_map = self.visual_encoder(img) # vis: [B, 256, d], feat_map: [B, d, 16, 16]

        # 3. Multimodal cross-attention layers
        cross_weights_history = []
        for layer in self.layers:
            text_feats, vis_feats, weights = layer(text_feats, vis_feats)
            cross_weights_history.append(weights)

        # 4. Extract grounded 2D target estimates from multimodal attention
        # Spatial grounding logits: pooling text representation across visual feature tokens
        txt_query = self.obj_query_head(text_feats.mean(dim=1)) # [B, d_model]
        vis_t = vis_feats.transpose(1, 2) # [B, d_model, 256]
        attn_logits = torch.bmm(txt_query.unsqueeze(1), vis_t) / (self.d_model ** 0.5) # [B, 1, 256]
        spatial_probs = F.softmax(attn_logits * 4.0, dim=-1) # [B, 1, 256]

        # Soft-Argmax (differentiable target location in [-1, 1] coordinates)
        grid_x = self.visual_encoder.grid_x.to(img.device)
        grid_y = self.visual_encoder.grid_y.to(img.device)
        gx = (spatial_probs * grid_x).sum(dim=-1) # [B, 1]
        gy = (spatial_probs * grid_y).sum(dim=-1) # [B, 1]
        grounded_2d = torch.cat([gx, gy], dim=-1) # [B, 2]

        return text_feats, vis_feats, cross_weights_history[-1], grounded_2d
