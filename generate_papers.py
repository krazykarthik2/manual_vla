"""
generate_papers.py  -  Generates three PDF papers using fpdf2 (pure Python, no LaTeX engine needed).
Papers focus on METHODOLOGY and AI ARCHITECTURE only.
  1. papers/manual_vla.pdf
  2. papers/smolvla_dobot.pdf
  3. papers/full_smolvla_dobot.pdf
"""
import os, sys
from fpdf import FPDF

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO, "papers")
FIG_DIR = os.path.join(REPO, "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
class Paper(FPDF):
    def __init__(self, title, subtitle=""):
        super().__init__()
        self._title = title
        self._subtitle = subtitle
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, self._title, align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_title_page(self):
        self.add_page()
        self.ln(30)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 12, self._title, align="C")
        if self._subtitle:
            self.ln(4)
            self.set_font("Helvetica", "", 12)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 8, self._subtitle, align="C")
        self.ln(8)
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, "Autonomous Robotics & VLA Research Lab", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "https://github.com/krazykarthik2/manual_vla", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 8, "September 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    def section(self, title):
        self.ln(6)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 120)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(4)

    def subsection(self, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, text, fill=True)
        self.ln(3)

    def add_fig(self, path, caption=""):
        if not os.path.exists(path):
            self.body(f"[Figure not found: {path}]")
            return
        avail = self.h - self.get_y() - self.b_margin - 20
        w = 180
        if avail < 60:
            self.add_page()
        self.image(path, x=15, w=w)
        if caption:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 5, caption, align="C")
        self.ln(4)


# ============================================================
# PAPER 1: manual_vla.pdf
# ============================================================
def make_manual_vla():
    p = Paper("Manual VLA: Methodology and AI Architecture",
              "A Lightweight Vision-Language-Action Transformer for Robotic Manipulation")
    p.add_title_page()

    p.add_page()
    p.section("1. Abstract")
    p.body(
        "This paper describes the Manual VLA system, a lightweight Vision-Language-Action transformer "
        "designed for real-time robotic manipulation on a 4-DOF Dobot Magician. The architecture uses a "
        "custom patch-based vision encoder, Hadamard MLP intent conditioning, and a compact transformer "
        "encoder to directly predict end-effector delta actions from a single overhead RGB image and a "
        "structured intent vector."
    )

    p.section("2. System Overview")
    fig1 = os.path.join(FIG_DIR, "fig1_manual_vla_arch.png")
    p.add_fig(fig1, "Figure 1: Manual VLA Architecture - Patch Embedding, Hadamard Conditioning, Transformer Encoder.")
    p.body(
        "Manual VLA is the foundational architecture in the repository. Unlike SmolVLA or Full SmolVLA, "
        "it does not use any pretrained vision-language backbone. Instead, it employs a fully trainable, "
        "from-scratch vision encoder combined with a structured intent vector to condition actions.\n\n"
        "The system pipeline is:\n"
        "  1. Capture a 64x64 RGB overhead image of the workspace.\n"
        "  2. Encode the image into patch tokens using a convolutional patch embedding.\n"
        "  3. Encode a 16-dimensional intent vector specifying action type, source color, and target color.\n"
        "  4. Apply Hadamard (element-wise) conditioning of visual tokens with intent features.\n"
        "  5. Concatenate a proprioceptive state token and process through a transformer encoder.\n"
        "  6. Predict a 5-dimensional delta action: [dx, dy, dz, dyaw, gripper]."
    )

    p.section("3. Architecture Details")

    p.subsection("3.1 Patch Embedding (PatchEmbedding)")
    p.body(
        "Input images of shape (B, 3, 64, 64) are processed by a Conv2d layer with kernel_size=16 "
        "and stride=16, producing (B, embed_dim, 4, 4) feature maps. These are flattened into "
        "16 patch tokens of dimension embed_dim=128."
    )
    p.code_block(
        "Conv2d(3, 128, kernel_size=16, stride=16)\n"
        "-> flatten(2).transpose(1,2)\n"
        "-> Output: (B, 16, 128)"
    )

    p.subsection("3.2 Intent Encoding")
    p.body(
        "The intent vector is 16-dimensional, composed of:\n"
        "  - Action type (2D one-hot): pick_place or push\n"
        "  - Source object color (7D one-hot): red, blue, yellow, green, purple, orange, cyan\n"
        "  - Target destination color (7D one-hot): same 7 colors\n\n"
        "This is projected through a 2-layer MLP with GELU activation:\n"
        "  Linear(16 -> 128) -> GELU -> Linear(128 -> 128)"
    )

    p.subsection("3.3 Proprioception Encoding")
    p.body(
        "The 5-dimensional proprioceptive state (current end-effector position + gripper state) "
        "is projected through the same MLP structure:\n"
        "  Linear(5 -> 128) -> GELU -> Linear(128 -> 128)\n\n"
        "This produces a single proprioceptive token of dimension 128."
    )

    p.subsection("3.4 Hadamard MLP Conditioning")
    p.body(
        "Instead of cross-attention, Manual VLA uses element-wise (Hadamard) multiplication to "
        "condition the visual patch tokens with the intent embedding:\n\n"
        "  conditioned_patches = patches * intent_embed(intent).unsqueeze(1)\n\n"
        "This modulates each patch token's activation pattern based on the task intent, "
        "encouraging the vision encoder to focus on task-relevant spatial regions."
    )

    p.subsection("3.5 Transformer Encoder")
    p.body(
        "The conditioned visual patches (16 tokens) and the proprioceptive token (1 token) "
        "are concatenated into a sequence of 17 tokens, each of dimension 128.\n\n"
        "This sequence is processed by a 2-layer TransformerEncoder:\n"
        "  - d_model = 128\n"
        "  - nhead = 4\n"
        "  - dim_feedforward = 256\n"
        "  - batch_first = True"
    )

    p.subsection("3.6 Action Head")
    p.body(
        "The transformer output is flattened across all 17 tokens (17 * 128 = 2176 dimensions) "
        "and passed through a 2-layer MLP:\n"
        "  Linear(2176 -> 128) -> GELU -> Linear(128 -> 5)\n\n"
        "Output: 5D delta action [dx, dy, dz, dyaw, gripper]."
    )

    p.section("4. Training Methodology")
    p.body(
        "Manual VLA is trained with Optimal Transport Conditional Flow Matching (OT-CFM). "
        "Demonstrations are auto-generated by a rule-based oracle that produces pick-and-place "
        "trajectories with random object placements and visual distractors.\n\n"
        "The training pipeline:\n"
        "  1. Generate demonstrations using auto_generate_demos.py\n"
        "  2. Train the model with train_flow.py using AdamW optimizer and cosine annealing\n"
        "  3. Evaluate with run.py in the simulated Dobot environment"
    )

    p.section("5. Flow Matching Formulation")
    p.body(
        "Given a clean demonstration trajectory x_1 and noise x_0 ~ N(0, I), the interpolated "
        "sample at time t is:\n\n"
        "  x_t = (1 - t) * x_0 + t * x_1,    t ~ U(0, 1)\n\n"
        "The target velocity field is:\n\n"
        "  u_t = x_1 - x_0\n\n"
        "The model learns to predict this velocity field conditioned on the current image and intent. "
        "At inference, trajectories are generated by Euler integration from pure noise over K steps."
    )

    p.output(os.path.join(OUT_DIR, "manual_vla.pdf"))
    print("[OK] papers/manual_vla.pdf")


# ============================================================
# PAPER 2: smolvla_dobot.pdf
# ============================================================
def make_smolvla_dobot():
    p = Paper("SmolVLA: Dual-Stream Frozen CLIP Policy",
              "Pretrained Vision-Language Backbone for Robotic Manipulation")
    p.add_title_page()

    p.add_page()
    p.section("1. Abstract")
    p.body(
        "SmolVLA is a compact Vision-Language-Action policy that leverages a frozen pretrained "
        "OpenAI CLIP ViT-B/32 backbone for multimodal perception, combined with lightweight "
        "cross-attention fusion and an Optimal Transport Conditional Flow Matching action decoder "
        "for trajectory generation on a 4-DOF Dobot Magician manipulator."
    )

    p.section("2. Architecture Overview")
    fig1 = os.path.join(FIG_DIR, "fig2_smolvla_arch.png")
    p.add_fig(fig1, "Figure 1: SmolVLA dual-stream architecture with frozen CLIP ViT-B/32 backbone.")

    p.body(
        "SmolVLA follows a dual-stream design:\n"
        "  - Vision Stream: frozen CLIP ViT-B/32 visual encoder\n"
        "  - Language Stream: frozen CLIP text transformer\n"
        "  - Multimodal Fusion: trainable cross-attention + self-attention blocks\n"
        "  - Action Decoder: transformer-based flow matching decoder"
    )

    p.section("3. Pretrained VLM Encoder (PretrainedVLMEncoder)")

    p.subsection("3.1 Vision Stream")
    p.body(
        "Input: 64x64 RGB images upsampled to 224x224 with bilinear interpolation.\n"
        "The image is normalized with CLIP's ImageNet statistics:\n"
        "  mean = [0.48145466, 0.4578275, 0.40821073]\n"
        "  std  = [0.26862954, 0.26130258, 0.27577711]\n\n"
        "Processing through CLIP's visual transformer:\n"
        "  1. Conv1 (patch projection): produces 7x7 = 49 spatial patch features\n"
        "  2. Prepend CLS token + add positional embeddings (50 tokens)\n"
        "  3. LayerNorm pre-processing\n"
        "  4. 12-layer ViT transformer\n"
        "  5. LayerNorm post-processing\n"
        "  6. Project through CLIP's visual projection matrix\n"
        "  7. L2-normalize\n\n"
        "Output: 49 patch tokens in R^{49 x 512} + 1 CLS embedding in R^512.\n"
        "All CLIP weights are frozen (requires_grad=False)."
    )

    p.subsection("3.2 Language Stream")
    p.body(
        "Input text instructions are tokenized into 77 BPE tokens.\n"
        "Processing:\n"
        "  1. Token embedding lookup\n"
        "  2. Add positional embeddings\n"
        "  3. Cast to CLIP dtype (float16 on CUDA)\n"
        "  4. 12-layer text transformer\n"
        "  5. LayerNorm final\n"
        "  6. Project through text_projection matrix\n"
        "  7. L2-normalize\n\n"
        "Output: 77 token embeddings in R^{77 x 512}. All weights frozen."
    )

    p.subsection("3.3 Spatial Grid Buffer")
    p.body(
        "A 7x7 spatial coordinate grid in [-1, 1] range is registered as a buffer, "
        "mapping each of the 49 visual patch tokens to a 2D spatial position for "
        "grounding and attention visualization."
    )

    p.section("4. Multimodal Fusion (MultimodalCrossAttentionBlock)")
    p.body(
        "After projection from 512-dim to d_model=128, the fusion operates in two stages "
        "per block. There are 2 stacked fusion blocks."
    )

    p.subsection("4.1 Cross-Attention: Language attends to Vision")
    p.body(
        "Query: LayerNorm(text_tokens)  [B, 16, 128]\n"
        "Key/Value: LayerNorm(visual_patches)  [B, 49, 128]\n"
        "Output: cross_out + residual text_tokens\n\n"
        "This grounds language tokens in the visual scene, allowing the model to "
        "associate words like 'blue cube' with specific spatial patch regions."
    )

    p.subsection("4.2 Joint Self-Attention")
    p.body(
        "The grounded text tokens and visual patches are concatenated:\n"
        "  joint = [text_tokens || visual_patches]  ->  [B, 65, 128]\n\n"
        "Full self-attention is applied over the joint sequence, allowing "
        "bidirectional information flow between language and vision modalities."
    )

    p.subsection("4.3 Feed-Forward Network")
    p.body(
        "A 2-layer MLP with GELU activation:\n"
        "  Linear(128 -> 256) -> GELU -> Linear(256 -> 128)\n"
        "Applied with residual connection after LayerNorm."
    )

    p.section("5. SmolVLA Backbone (SmolVLABackbone)")
    p.body(
        "The backbone class orchestrates the full forward pass:\n\n"
        "  1. Encode vision (frozen): 49 patches at 512-dim + CLS token\n"
        "  2. Encode text (frozen): 77 tokens at 512-dim, truncated to first 16\n"
        "  3. Compute zero-shot similarity heatmap:\n"
        "     base_weights = einsum('bld,bpd->blp', text_feats, vis_patches)\n"
        "     base_probs = softmax(base_weights * 5.0, dim=-1)\n"
        "  4. Project both streams: 512 -> 128 via Linear + LayerNorm\n"
        "  5. Pass through 2 MultimodalCrossAttentionBlocks\n\n"
        "Output: grounded text tokens [B, 16, 128], fused visual patches [B, 49, 128], "
        "and the pretrained attention heatmap [B, 77, 49]."
    )

    p.section("6. Flow Matching Action Decoder")
    fig2 = os.path.join(FIG_DIR, "fig4_flow_matching.png")
    p.add_fig(fig2, "Figure 2: OT-CFM linear interpolation and velocity field.")
    p.body(
        "The action decoder predicts 128-step continuous trajectories for 4 action dimensions "
        "(dx, dy, dz, grip) using Optimal Transport Conditional Flow Matching.\n\n"
        "Formulation:\n"
        "  x_t = (1-t)*x_0 + t*x_1,  where x_0 ~ N(0,I) and x_1 is the demo trajectory\n"
        "  Target: u_t = x_1 - x_0 (constant velocity field)\n"
        "  Loss weights: w = [1.2, 1.2, 1.5, 2.5] for [x, y, z, grip]\n\n"
        "Inference: Euler integration from noise over 15 steps."
    )

    p.section("7. Visual Grounding Analysis")
    fig3 = os.path.join(FIG_DIR, "fig5_cross_attention.png")
    p.add_fig(fig3, "Figure 3: Cross-attention heatmaps and spatial grounding.")

    p.output(os.path.join(OUT_DIR, "smolvla_dobot.pdf"))
    print("[OK] papers/smolvla_dobot.pdf")


# ============================================================
# PAPER 3: full_smolvla_dobot.pdf
# ============================================================
def make_full_smolvla_dobot():
    p = Paper("Full SmolVLA: Autoregressive Foundation VLM Policy",
              "SmolVLM-256M-Instruct Backbone for Robotic Manipulation")
    p.add_title_page()

    p.add_page()
    p.section("1. Abstract")
    p.body(
        "Full SmolVLA integrates the complete HuggingFace SmolVLM-256M-Instruct foundation model "
        "as a native perception backbone. Unlike SmolVLA's dual-stream frozen CLIP approach, "
        "Full SmolVLA feeds images and text into a unified autoregressive vision-language model "
        "and taps multi-layer intermediate representations for rich multimodal conditioning of the "
        "flow matching action decoder."
    )

    p.section("2. Architecture Overview")
    fig1 = os.path.join(FIG_DIR, "fig3_full_smolvla_arch.png")
    p.add_fig(fig1, "Figure 1: Full SmolVLA architecture using SmolVLM-256M-Instruct backbone.")
    p.body(
        "Full SmolVLA consists of three major components:\n"
        "  1. SmolVLM-256M-Instruct: frozen autoregressive vision-language backbone\n"
        "  2. Multi-layer intermediate feature projection\n"
        "  3. Cross-attention action decoder with flow matching"
    )

    p.section("3. SmolVLM Foundation Backbone")

    p.subsection("3.1 Model Details")
    p.body(
        "Model: HuggingFaceTB/SmolVLM-256M-Instruct\n"
        "Type: SmolVLMForConditionalGeneration (autoregressive VLM)\n"
        "Hidden dimension: 576\n\n"
        "The backbone is loaded in reduced precision:\n"
        "  - bfloat16 on CUDA if BF16 is supported\n"
        "  - float16 on CUDA otherwise\n"
        "  - float32 on CPU\n\n"
        "All backbone weights are frozen (requires_grad=False)."
    )

    p.subsection("3.2 Input Formatting")
    p.body(
        "Images and text are formatted using HuggingFace's chat template:\n\n"
        '  messages = [{"role": "user", "content": [\n'
        '      {"type": "image"},\n'
        '      {"type": "text", "text": prompt}\n'
        "  ]}]\n\n"
        "The processor tokenizes text and encodes images into the model's expected format "
        "with special <image> tokens embedded in the token sequence."
    )

    p.subsection("3.3 Multi-Layer Intermediate Feature Tapping (Architecture 3B)")
    p.body(
        "Instead of using only the final hidden state, Full SmolVLA taps three intermediate "
        "layers of the SmolVLM transformer:\n\n"
        "  - Layer 10: Early spatial and edge features\n"
        "  - Layer 20: Mid-level relational features\n"
        "  - Layer 30: High-level goal and semantic features\n\n"
        "The hidden states from these three layers are concatenated along the feature dimension:\n"
        "  multi_layer_hidden = cat([h_10, h_20, h_30], dim=-1)\n"
        "  Shape: [B, seq_len, 576 * 3] = [B, seq_len, 1728]\n\n"
        "This multi-scale representation captures both fine spatial detail and abstract "
        "semantic understanding."
    )

    p.section("4. Multimodal Context Projection")
    p.body(
        "The concatenated multi-layer features are projected to the action model dimension:\n\n"
        "  vlm_proj = Sequential(\n"
        "    Linear(1728 -> 128),\n"
        "    LayerNorm(128),\n"
        "    GELU(),\n"
        "    Linear(128 -> 128),\n"
        "    LayerNorm(128)\n"
        "  )\n\n"
        "Output: [B, seq_len, 128] multimodal context tokens."
    )

    p.section("5. Conditioning Signals")

    p.subsection("5.1 Proprioception Encoding")
    p.body(
        "The 5D proprioceptive state (end-effector position + gripper) is projected:\n"
        "  Linear(5 -> 128) -> GELU -> Linear(128 -> 128)\n"
        "Produces a single proprioceptive token [B, 1, 128]."
    )

    p.subsection("5.2 Continuous Sinusoidal Time Embedding")
    p.body(
        "The diffusion timestep t in [0, 1] is encoded via sinusoidal frequencies:\n\n"
        "  freqs = exp(-log(10000) * arange(32) / 31)\n"
        "  time_emb = cat([sin(t * freqs), cos(t * freqs)], dim=-1)  -> R^64\n\n"
        "Then projected:\n"
        "  Linear(64 -> 128) -> GELU -> Linear(128 -> 128)\n"
        "Produces a single time token [B, 1, 128]."
    )

    p.subsection("5.3 Full Context Assembly")
    p.body(
        "The conditioning context is assembled by concatenation:\n\n"
        "  context = [proprio_token, time_token, vlm_tokens]\n"
        "  Shape: [B, 2 + seq_len, 128]\n\n"
        "This provides the action decoder with proprioceptive grounding, temporal awareness, "
        "and rich multimodal scene understanding."
    )

    p.section("6. Cross-Attention Action Decoder")
    p.body(
        "The action decoder generates 128-step trajectories for 4 action dimensions "
        "(dx, dy, dz, grip). It consists of 2 decoder layers, each with:"
    )

    p.subsection("6.1 Action Query Initialization")
    p.body(
        "Noisy action inputs x_t are projected to d_model=128 and combined with "
        "learnable positional queries:\n\n"
        "  act_queries = Linear(4 -> 128)(x_t) + pos_queries\n"
        "  pos_queries: Parameter(1, 128, 128) initialized with N(0, 0.02)"
    )

    p.subsection("6.2 Decoder Layer Structure (x2)")
    p.body(
        "Each layer performs three operations:\n\n"
        "  1. Self-Attention: act_queries attend to themselves\n"
        "     MultiheadAttention(128, 4 heads) + LayerNorm + residual\n\n"
        "  2. Cross-Attention: act_queries attend to the full context\n"
        "     MultiheadAttention(128, 4 heads) + LayerNorm + residual\n"
        "     This is where the action trajectory is grounded in visual/language understanding.\n\n"
        "  3. Feed-Forward: 2-layer MLP\n"
        "     Linear(128 -> 256) -> GELU -> Linear(256 -> 128) + LayerNorm + residual"
    )

    p.subsection("6.3 Output Head")
    p.body(
        "The final action predictions are produced by:\n"
        "  LayerNorm(128) -> Linear(128 -> 128) -> GELU -> Linear(128 -> 4)\n\n"
        "Output: predicted velocity field v_pred of shape [B, 128, 4]."
    )

    p.section("7. Flow Matching Trajectory Generation")
    fig2 = os.path.join(FIG_DIR, "fig4_flow_matching.png")
    p.add_fig(fig2, "Figure 2: Optimal Transport Conditional Flow Matching dynamics.")
    p.body(
        "Training:\n"
        "  x_t = (1-t)*x_0 + t*x_1,  x_0 ~ N(0,I),  x_1 = demo trajectory\n"
        "  Target velocity: u_t = x_1 - x_0\n"
        "  Loss weights: w = [1.2, 1.2, 1.5, 2.5] for [x, y, z, grip]\n\n"
        "Inference (sampling):\n"
        "  1. Start from noise: x_0 ~ N(0, I), shape [B, 128, 4]\n"
        "  2. Euler integration over K=15 steps: x_{t+dt} = x_t + v(x_t, t, context) * dt\n"
        "  3. Denormalize: raw = x * ACTION_STD + ACTION_MEAN\n"
        "  4. Temporal smoothing: 1D convolution with kernel size 5"
    )

    p.section("8. Activation Returns for Visualization")
    p.body(
        "When return_activations=True, the decoder returns a dictionary of intermediate "
        "layer outputs for visualization and debugging:\n\n"
        "  - context: full conditioning sequence [B, 2+seq_len, 128]\n"
        "  - l1_sa, l1_ca, l1_out: Layer 1 self-attn, cross-attn, FFN outputs\n"
        "  - l2_sa, l2_ca, l2_out: Layer 2 outputs\n"
        "  - ca_w1, ca_w2: Cross-attention weight matrices [B, 128, context_len]\n"
        "  - v_pred: Final velocity prediction [B, 128, 4]\n\n"
        "These activations enable spectrograms of layer-wise processing and "
        "cross-attention grounding maps."
    )

    p.output(os.path.join(OUT_DIR, "full_smolvla_dobot.pdf"))
    print("[OK] papers/full_smolvla_dobot.pdf")


# ---------- main ----------
if __name__ == "__main__":
    print("=" * 50)
    print("Generating papers with fpdf2 (pure Python)...")
    print("=" * 50)
    make_manual_vla()
    make_smolvla_dobot()
    make_full_smolvla_dobot()
    print("=" * 50)
    print("Done. All PDFs in papers/ folder.")
