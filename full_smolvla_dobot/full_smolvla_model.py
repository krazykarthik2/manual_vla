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

        if load_backbone:
            print(f"[FullSmolVLA] Initializing genuine SmolVLM foundation backbone ({MODEL_NAME})...", flush=True)
            self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
            self.smolvlm = SmolVLMForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(device)

            if freeze_backbone:
                print("[FullSmolVLA] Freezing pretrained SmolVLM weights (training Action Expert Cross-Attention Head)...", flush=True)
                for p in self.smolvlm.parameters():
                    p.requires_grad = False
                self.smolvlm.eval()

            # SmolVLM text config hidden dimension
            smolvlm_hidden_dim = getattr(self.smolvlm.config.text_config, "hidden_size", smolvlm_hidden_dim)

        self.vlm_proj = nn.Sequential(
            nn.Linear(smolvlm_hidden_dim, d_action_model),
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

    def extract_smolvlm_context(self, images_pil, prompt_texts):
        """
        Passes images and text prompts through the full SmolVLM backbone
        and extracts the final layer multimodal representations.
        Returns: [B, seq_len, d_action_model]
        """
        # Formulate conversation prompts for SmolVLM Instruct format
        formatted_prompts = []
        for prompt in prompt_texts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"Instruction: {prompt}. Predict robotic motion."}
                    ]
                }
            ]
            formatted_prompts.append(self.processor.apply_chat_template(messages, add_generation_prompt=True))

        inputs = self.processor(text=formatted_prompts, images=images_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.smolvlm(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1] # [B, seq_len, smolvlm_hidden_dim]

        vlm_tokens = self.vlm_proj(last_hidden.float()) # [B, seq_len, d_action_model]
        return vlm_tokens

    def forward_from_embeddings(self, x_t, t, vlm_tokens, proprio=None):
        """
        Action generation conditioned on cached or extracted SmolVLM tokens.
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
        return v_pred

    @torch.no_grad()
    def sample(self, images_pil, prompt_texts, proprio=None, num_steps=15):
        B = len(images_pil)
        vlm_tokens = self.extract_smolvlm_context(images_pil, prompt_texts)

        ACTION_MEAN = torch.tensor([0.2028, 0.0008, 0.0854, 0.5360], dtype=torch.float32, device=self.device)
        ACTION_STD  = torch.tensor([0.0330, 0.0837, 0.0362, 0.4987], dtype=torch.float32, device=self.device)

        x = torch.randn(B, self.horizon, self.action_dim, device=self.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=self.device)
            v = self.forward_from_embeddings(x, t, vlm_tokens, proprio=proprio)
            x = x + v * dt

        raw_x = x * ACTION_STD + ACTION_MEAN
        kernel = torch.ones(1, 1, 5, device=self.device) / 5.0
        raw_perm = raw_x.permute(0, 2, 1).reshape(B * self.action_dim, 1, self.horizon)
        smoothed = F.conv1d(raw_perm, kernel, padding=2)
        smoothed = smoothed.view(B, self.action_dim, self.horizon).permute(0, 2, 1)
        return smoothed
