import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

os.environ['USE_TF'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.models.smolvlm import SmolVLMConfig, SmolVLMForConditionalGeneration

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

class ContinuousSinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class SABlock(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        q = self.norm1(x)
        L = q.size(1)
        mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)
        attn_out, _ = self.attn(q, q, q, attn_mask=mask, is_causal=True)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class CABlock(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, context, need_weights=False):
        q = self.norm1(x)
        attn_out, weights = self.attn(q, context, context, need_weights=need_weights)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, weights

class FullSmolVLAPolicy(nn.Module):
    """
    Official SmolVLA Architecture implementation:
    - Backbone: HuggingFace SmolVLM-256M-Instruct
    - Layer Skipping: Extracts features at N = L/2 (Layer 15)
    - Expert Width: 0.75 * d_vlm (432 dims)
    - Architecture: Interleaved Causal Self-Attention and Cross-Attention blocks
    """
    def __init__(self, horizon=128, action_dim=4, freeze_backbone=True, load_backbone=True, smolvlm_hidden_dim=576, device='cpu'):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = device
        self.processor = None
        self.smolvlm = None

        # Official Paper: "we find setting N to half the total layers (N = L/2) offers a good tradeoff"
        self.selected_layer = 15  # SmolVLM-256M has 30 layers
        d_vlm = smolvlm_hidden_dim 

        if load_backbone:
            print(f"[FullSmolVLA] Initializing genuine SmolVLM foundation backbone ({MODEL_NAME})...", flush=True)
            self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
            
            is_cuda = "cuda" in str(device)
            vlm_dtype = torch.bfloat16 if (is_cuda and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if is_cuda else torch.float32)
            
            self.smolvlm = SmolVLMForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                dtype=vlm_dtype,
                low_cpu_mem_usage=True
            ).to(device)

            if freeze_backbone:
                print("[FullSmolVLA] Freezing pretrained SmolVLM weights...", flush=True)
                for p in self.smolvlm.parameters():
                    p.requires_grad = False
                self.smolvlm.eval()

            d_vlm = getattr(self.smolvlm.config.text_config, "hidden_size", d_vlm)

        # Official Paper: "we use a reduced hidden size of 0.75 * d for v_theta"
        self.d_model = int(0.75 * d_vlm) # 432
        
        self.vlm_proj = nn.Sequential(
            nn.LayerNorm(d_vlm),
            nn.Linear(d_vlm, self.d_model)
        )

        self.proprio_proj = nn.Sequential(
            nn.Linear(5, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )
        self.time_embed = nn.Sequential(
            ContinuousSinusoidalTimeEmbedding(64),
            nn.Linear(64, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )

        self.action_in_proj = nn.Linear(action_dim, self.d_model)
        self.pos_queries = nn.Parameter(torch.randn(1, horizon, self.d_model) * 0.02)

        # Official Paper: "we employ an interleaved approach, where each block contains either a CA or a SA layer."
        # We use 4 blocks total to match the 4-layer specification
        self.blocks = nn.ModuleList([
            SABlock(self.d_model, nhead=8),
            CABlock(self.d_model, nhead=8),
            SABlock(self.d_model, nhead=8),
            CABlock(self.d_model, nhead=8)
        ])

        self.out_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, action_dim)
        )

    def extract_smolvlm_context(self, images_pil, prompt_texts, return_raw_hidden=False):
        formatted_prompts = []
        for prompt in prompt_texts:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt.strip()}]}]
            formatted_prompts.append(self.processor.apply_chat_template(messages, add_generation_prompt=False))

        inputs = self.processor(text=formatted_prompts, images=images_pil, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.smolvlm(**inputs, output_hidden_states=True)
            # Official Paper: extract features at N-th layer
            layer_hidden = outputs.hidden_states[self.selected_layer].float()

        if return_raw_hidden:
            return layer_hidden

        return self.vlm_proj(layer_hidden)

    def forward(self, raw_hidden, proprio, x_t, t):
        vlm_tokens = self.vlm_proj(raw_hidden)
        return self.forward_from_embeddings(x_t, t, vlm_tokens, proprio=proprio)

    def forward_from_embeddings(self, x_t, t, vlm_tokens, proprio=None, return_activations=False):
        B = vlm_tokens.size(0)

        if proprio is None:
            proprio = torch.zeros(B, 5, device=vlm_tokens.device)
        p_token = self.proprio_proj(proprio).unsqueeze(1)
        t_token = self.time_embed(t).unsqueeze(1)

        # Official Paper: "States as Prefix"
        context = torch.cat([p_token, t_token, vlm_tokens], dim=1)

        x = self.action_in_proj(x_t) + self.pos_queries
        
        ca_weights = []
        block_outputs = []
        for block in self.blocks:
            if isinstance(block, CABlock):
                x, w = block(x, context, need_weights=return_activations)
                if return_activations:
                    ca_weights.append(w)
            else:
                x = block(x)
            block_outputs.append(x)

        v_pred = self.out_head(x)

        if return_activations:
            activations = {
                "context": context,
                "l1_sa": block_outputs[0],
                "l1_ca": block_outputs[1],
                "l1_out": block_outputs[1],
                "l2_sa": block_outputs[2] if len(block_outputs) > 2 else block_outputs[0],
                "l2_ca": block_outputs[3] if len(block_outputs) > 3 else block_outputs[1],
                "l2_out": block_outputs[3] if len(block_outputs) > 3 else block_outputs[1],
                "ca_w1": ca_weights[0],
                "ca_w2": ca_weights[1] if len(ca_weights) > 1 else ca_weights[0],
                "v_pred": v_pred
            }
            return v_pred, activations
        return v_pred

    @torch.no_grad()
    def sample(self, images_pil, prompt_texts, proprio=None, num_steps=15, return_activations=False):
        B = len(images_pil)
        vlm_tokens = self.extract_smolvlm_context(images_pil, prompt_texts)

        stats_path = os.path.join(os.path.dirname(__file__), "models", "dobot_full_smolvla_policy_stats.pt")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location=self.device)
            ACTION_MEAN = stats["mean"].to(self.device)
            ACTION_STD = stats["std"].to(self.device)
        else:
            ACTION_MEAN = torch.tensor([0.2028, 0.0008, 0.0854, 0.5360], dtype=torch.float32, device=self.device)
            ACTION_STD  = torch.tensor([0.0330, 0.0837, 0.0362, 0.4987], dtype=torch.float32, device=self.device)

        x = torch.randn(B, self.horizon, self.action_dim, device=self.device)
        dt = 1.0 / num_steps
        last_acts = None

        for i in range(num_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=self.device)
            need_act = return_activations and (i == num_steps - 1)
            if need_act:
                v, last_acts = self.forward_from_embeddings(x, t, vlm_tokens, proprio=proprio, return_activations=True)
            else:
                v = self.forward_from_embeddings(x, t, vlm_tokens, proprio=proprio, return_activations=False)
            x = x + v * dt

        raw_x = x * ACTION_STD + ACTION_MEAN
        kernel = torch.ones(1, 1, 5, device=self.device) / 5.0
        raw_perm = raw_x.permute(0, 2, 1).reshape(B * self.action_dim, 1, self.horizon)
        smoothed = F.conv1d(raw_perm, kernel, padding=2)
        smoothed = smoothed.view(B, self.action_dim, self.horizon).permute(0, 2, 1)

        if return_activations:
            return smoothed, last_acts, vlm_tokens

        return smoothed
