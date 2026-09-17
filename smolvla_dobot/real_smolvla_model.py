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

MODEL_NAME = 'HuggingFaceTB/SmolVLM-256M-Instruct'

class RealPretrainedSmolVLA(nn.Module):
    """
    Real Official Hugging Face SmolVLM Vision-Language-Action Policy:
    1. Pretrained SmolVLM Backbone (Frozen weights from Hugging Face).
    2. Real Image-Text Multimodal Projection & Context Extraction.
    3. Trainable Cross-Attention Flow-Matching Action Head (Horizon H=128, Action Dim=4).
    4. Proprioception & Diffusion Continuous Time Injection.
    """
    def __init__(self, d_action_model=128, horizon=128, action_dim=4, freeze_backbone=True):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.d_model = d_action_model

        # Load official SmolVLM Config & Model
        print(f'[SmolVLA] Initializing real SmolVLM architecture ({MODEL_NAME})...', flush=True)
        self.processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-Instruct')
        self.smolvlm = SmolVLMForConditionalGeneration.from_pretrained(
            'HuggingFaceTB/SmolVLM-Instruct',
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        if freeze_backbone:
            print('[SmolVLA] Freezing pretrained SmolVLM weights (only training Cross-Attn Action Head)...', flush=True)
            for p in self.smolvlm.parameters():
                p.requires_grad = False

        # SmolVLM hidden dimension projection -> action head dimension (d_model=128)
        smolvlm_hidden_dim = self.smolvlm.config.text_config.hidden_size # 576 or 2048
        self.vlm_proj = nn.Linear(smolvlm_hidden_dim, d_action_model)

        # Proprioception & Continuous Diffusion Time
        self.proprio_proj = nn.Sequential(
            nn.Linear(5, d_action_model),
            nn.GELU(),
            nn.Linear(d_action_model, d_action_model)
        )
        self.time_embed = nn.Sequential(
            nn.Linear(64, d_action_model),
            nn.GELU(),
            nn.Linear(d_action_model, d_action_model)
        )

        # Trainable Cross-Attention Flow Matching Action Head (4 layers)
        self.action_in_proj = nn.Linear(action_dim, d_action_model)
        self.pos_queries = nn.Parameter(torch.randn(1, horizon, d_action_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_action_model,
            nhead=4,
            dim_feedforward=d_action_model * 2,
            activation='gelu',
            batch_first=True
        )
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.out_head = nn.Sequential(
            nn.LayerNorm(d_action_model),
            nn.Linear(d_action_model, d_action_model),
            nn.GELU(),
            nn.Linear(d_action_model, action_dim)
        )

    def extract_smolvlm_embeddings(self, images_pil, prompt_texts):
        # Official Hugging Face multimodal forward pass
        inputs = self.processor(text=prompt_texts, images=images_pil, return_tensors='pt')
        with torch.no_grad():
            outputs = self.smolvlm(**inputs, output_hidden_states=True)
            # Take last hidden layer of SmolVLM multimodal representations
            last_hidden = outputs.hidden_states[-1] # [B, seq_len, hidden_dim]
        return self.vlm_proj(last_hidden) # [B, seq_len, 128]
