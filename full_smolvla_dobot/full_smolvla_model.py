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

class FullSmolVLAPolicy(nn.Module):
    """
    Full Real SmolVLA Architecture:
    - Entire SmolVLM Vision-Language Foundation Backbone (Hugging Face SmolVLM-256M-Instruct)
    - Multimodal context projection from SmolVLM hidden dimension to d_action_model
    - Continuous Sinusoidal Diffusion Time + Proprioception Conditioning
    - Multimodal Cross-Attention Action Decoder Head over Horizon H=128 for 4D actions (x, y, z, grip)
    """
    def __init__(self, d_action_model=128, horizon=128, action_dim=4, freeze_backbone=True, load_backbone=True, smolvlm_hidden_dim=576, device='cpu'):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.d_model = d_action_model
        self.device = device
        self.processor = None
        self.smolvlm = None

        # Architecture 3B: Multi-layer intermediate feature tapping
        # Layer 10 (early spatial/edges), Layer 20 (mid relational), Layer 30 (high-level goal semantics)
        self.selected_layers = (10, 20, 30)
        num_layers = len(self.selected_layers)

        if load_backbone:
            print(f"[FullSmolVLA] Initializing genuine SmolVLM foundation backbone ({MODEL_NAME})...", flush=True)
            self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
            
            # Use bfloat16/float16 on CUDA to reduce VRAM from 1.2GB -> 300MB and cut activation memory by 50%
            is_cuda = "cuda" in str(device)
            vlm_dtype = torch.bfloat16 if (is_cuda and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if is_cuda else torch.float32)
            
            self.smolvlm = SmolVLMForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                dtype=vlm_dtype,
                low_cpu_mem_usage=True
            ).to(device)

            if freeze_backbone:
                print("[FullSmolVLA] Freezing pretrained SmolVLM weights (training Action Expert Cross-Attention Head)...", flush=True)
                for p in self.smolvlm.parameters():
                    p.requires_grad = False
                self.smolvlm.eval()

            smolvlm_hidden_dim = getattr(self.smolvlm.config.text_config, "hidden_size", smolvlm_hidden_dim)

        # Multi-layer intermediate projection (Architecture 3B)
        self.vlm_proj = nn.Sequential(
            nn.Linear(smolvlm_hidden_dim * num_layers, d_action_model),
            nn.LayerNorm(d_action_model),
            nn.GELU(),
            nn.Linear(d_action_model, d_action_model),
            nn.LayerNorm(d_action_model)
        )

        # Proprioception & Continuous Diffusion Time
        self.proprio_proj = nn.Sequential(
            nn.Linear(5, d_action_model),
            nn.GELU(),
            nn.Linear(d_action_model, d_action_model)
        )
        self.time_embed = nn.Sequential(
            ContinuousSinusoidalTimeEmbedding(64),
            nn.Linear(64, d_action_model),
            nn.GELU(),
            nn.Linear(d_action_model, d_action_model)
        )

        # Action Decoder Head (Cross-Attention over SmolVLM multimodal sequence)
        self.action_in_proj = nn.Linear(action_dim, d_action_model)
        self.pos_queries = nn.Parameter(torch.randn(1, horizon, d_action_model) * 0.02)

        self.dec_sa1 = nn.MultiheadAttention(d_action_model, 4, batch_first=True)
        self.dec_ca1 = nn.MultiheadAttention(d_action_model, 4, batch_first=True)
        self.dec_n1 = nn.LayerNorm(d_action_model)
        self.dec_n2 = nn.LayerNorm(d_action_model)
        self.dec_n3 = nn.LayerNorm(d_action_model)
        self.dec_ffn1 = nn.Sequential(
            nn.Linear(d_action_model, d_action_model * 2),
            nn.GELU(),
            nn.Linear(d_action_model * 2, d_action_model)
        )

        self.dec_sa2 = nn.MultiheadAttention(d_action_model, 4, batch_first=True)
        self.dec_ca2 = nn.MultiheadAttention(d_action_model, 4, batch_first=True)
        self.dec_n4 = nn.LayerNorm(d_action_model)
        self.dec_n5 = nn.LayerNorm(d_action_model)
        self.dec_n6 = nn.LayerNorm(d_action_model)
        self.dec_ffn2 = nn.Sequential(
            nn.Linear(d_action_model, d_action_model * 2),
            nn.GELU(),
            nn.Linear(d_action_model * 2, d_action_model)
        )

        self.out_head = nn.Sequential(
            nn.LayerNorm(d_action_model),
            nn.Linear(d_action_model, d_action_model),
            nn.GELU(),
            nn.Linear(d_action_model, action_dim)
        )

    def extract_smolvlm_context(self, images_pil, prompt_texts, return_raw_hidden=False):
        """
        Passes images and simplified text prompts through the full SmolVLM backbone.
        Architecture 3B: Taps intermediate layers (10, 20, 30) for multi-scale spatial + semantic grounding.
        """
        # Simplified prompt format without unnecessary wrappers
        formatted_prompts = []
        for prompt in prompt_texts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt.strip()}
                    ]
                }
            ]
            formatted_prompts.append(self.processor.apply_chat_template(messages, add_generation_prompt=False))

        inputs = self.processor(text=formatted_prompts, images=images_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.smolvlm(**inputs, output_hidden_states=True)
            # Architecture 3B: Concatenate hidden states from early (10), mid (20), and late (30) layers
            multi_layer_hidden = torch.cat([outputs.hidden_states[idx].float() for idx in self.selected_layers], dim=-1)

        if return_raw_hidden:
            return multi_layer_hidden

        vlm_tokens = self.vlm_proj(multi_layer_hidden) # [B, seq_len, d_action_model]
        return vlm_tokens

    def forward_from_embeddings(self, x_t, t, vlm_tokens, proprio=None, return_activations=False):
        """
        Action generation conditioned on cached or extracted SmolVLM tokens.
        If return_activations=True, also returns dict of layer outputs and cross-attention maps.
        """
        B = vlm_tokens.size(0)

        if proprio is None:
            proprio = torch.zeros(B, 5, device=vlm_tokens.device)
        p_token = self.proprio_proj(proprio).unsqueeze(1) # [B, 1, d_model]
        t_token = self.time_embed(t).unsqueeze(1)         # [B, 1, d_model]

        # Full context: [proprio, time, SmolVLM multimodal tokens]
        context = torch.cat([p_token, t_token, vlm_tokens], dim=1)

        # Action Decoder
        act_queries = self.action_in_proj(x_t) + self.pos_queries

        # Layer 1
        sa_out, sa_w1 = self.dec_sa1(act_queries, act_queries, act_queries, need_weights=return_activations)
        act_queries = self.dec_n1(act_queries + sa_out)
        l1_sa = act_queries

        ca_out, ca_w1 = self.dec_ca1(act_queries, context, context, need_weights=True)
        act_queries = self.dec_n2(act_queries + ca_out)
        l1_ca = act_queries

        act_queries = self.dec_n3(act_queries + self.dec_ffn1(act_queries))
        l1_out = act_queries

        # Layer 2
        sa_out2, sa_w2 = self.dec_sa2(act_queries, act_queries, act_queries, need_weights=return_activations)
        act_queries = self.dec_n4(act_queries + sa_out2)
        l2_sa = act_queries

        ca_out2, ca_w2 = self.dec_ca2(act_queries, context, context, need_weights=True)
        act_queries = self.dec_n5(act_queries + ca_out2)
        l2_ca = act_queries

        act_queries = self.dec_n6(act_queries + self.dec_ffn2(act_queries))
        l2_out = act_queries

        v_pred = self.out_head(act_queries)

        if return_activations:
            activations = {
                "context": context,              # [B, 2+seq_len, d_model]
                "l1_sa": l1_sa,                  # [B, horizon, d_model]
                "l1_ca": l1_ca,                  # [B, horizon, d_model]
                "l1_out": l1_out,                # [B, horizon, d_model]
                "ca_w1": ca_w1,                  # [B, horizon, context_len]
                "l2_sa": l2_sa,                  # [B, horizon, d_model]
                "l2_ca": l2_ca,                  # [B, horizon, d_model]
                "l2_out": l2_out,                # [B, horizon, d_model]
                "ca_w2": ca_w2,                  # [B, horizon, context_len]
                "v_pred": v_pred                 # [B, horizon, action_dim]
            }
            return v_pred, activations

        return v_pred

    @torch.no_grad()
    def sample(self, images_pil, prompt_texts, proprio=None, num_steps=15, return_activations=False):
        B = len(images_pil)
        vlm_tokens = self.extract_smolvlm_context(images_pil, prompt_texts)

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
