"""
generate_papers.py
Generates highly formal, academically-formatted PDFs showcasing real architectures,
methodologies, mathematical formulations, and flow-matching generation.
Uses FPDF2.
"""
import os
from fpdf import FPDF

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figures")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

class AcademicPaper(FPDF):
    def __init__(self, title):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.paper_title = title
        
        # Title
        self.set_font("helvetica", "B", 18)
        self.multi_cell(0, 10, title, align="C")
        self.ln(5)
        
        # Authors
        self.set_font("helvetica", "B", 11)
        self.cell(0, 6, "Karthik Goparaju", ln=True, align="C")
        self.set_font("helvetica", "", 10)
        self.cell(0, 5, "R&D Intern at Innovarsity and Founder LABKRAZY", ln=True, align="C")
        self.cell(0, 5, "goparajukarthik2@gmail.com", ln=True, align="C")
        self.ln(3)
        
        self.set_font("helvetica", "B", 11)
        self.cell(0, 6, "Gopi Krishna Ratnakaram", ln=True, align="C")
        self.set_font("helvetica", "", 10)
        self.cell(0, 5, "Founder of Innovarsity", ln=True, align="C")
        self.cell(0, 5, "info@innovarsity.org", ln=True, align="C")
        self.ln(10)

    def heading(self, text):
        self.ln(8)
        self.set_font("helvetica", "B", 13)
        self.cell(0, 8, text, ln=True)
        self.ln(2)

    def sub_heading(self, text):
        self.ln(5)
        self.set_font("helvetica", "B", 11)
        self.cell(0, 6, text, ln=True)
        self.ln(1)

    def paragraph(self, text):
        self.set_font("helvetica", "", 10)
        text = text.replace("\u2014", " - ") # Fix unicode issues
        self.multi_cell(0, 5, text)
        self.ln(3)

    def math_block(self, text):
        self.ln(2)
        self.set_font("courier", "I", 10)
        self.multi_cell(0, 5, text, align="C")
        self.ln(2)

    def code_block(self, text):
        # Sanitize torchinfo unicode drawing characters for FPDF
        replacements = {
            '\u2500': '-', '\u2502': '|', '\u250c': '+', '\u2510': '+',
            '\u2514': '+', '\u2518': '+', '\u251c': '+', '\u2524': '+',
            '\u252c': '+', '\u2534': '+', '\u253c': '+', '\u2550': '=',
            '\u2551': '|', '\u2552': '+', '\u2553': '+', '\u2554': '+',
            '\u2555': '+', '\u2556': '+', '\u2557': '+', '\u2558': '+',
            '\u2559': '+', '\u255a': '+', '\u255b': '+', '\u255c': '+',
            '\u255d': '+', '\u255e': '+', '\u255f': '+', '\u2560': '+',
            '\u2561': '+', '\u2562': '+', '\u2563': '+', '\u2564': '+',
            '\u2565': '+', '\u2566': '+', '\u2567': '+', '\u2568': '+',
            '\u2569': '+', '\u256a': '+', '\u256b': '+', '\u256c': '+'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        # catch any remaining non-ascii
        text = ''.join(c if ord(c) < 128 else '?' for c in text)
        
        self.ln(3)
        self.set_font("courier", "", 8)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 4, text, fill=True)
        self.ln(3)

    def figure(self, path, caption):
        if not os.path.exists(path):
            print(f"Warning: Figure {path} not found.")
            return
        self.ln(5)
        # Check current Y position to avoid awkward page breaks across figure/caption
        if self.get_y() > 200:
            self.add_page()
            
        w = 160
        x = (self.w - w) / 2
        try:
            self.image(path, x=x, w=w)
            self.ln(3)
            self.set_font("helvetica", "I", 9)
            self.multi_cell(0, 5, caption, align="C")
            self.ln(5)
        except Exception as e:
            print(f"Error adding image {path}: {e}")

def load_summary(filename):
    path = os.path.join(ROOT_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 40:
                # truncate middle to fit page nicely
                return "".join(lines[:20]) + "\n    [ ... intermediate layers omitted ... ]\n" + "".join(lines[-20:])
            return "".join(lines)
    return "[Network summary not available]"

# ==============================================================================
# 1. Manual VLA Paper
# ==============================================================================
def make_manual_vla():
    p = AcademicPaper("Manual VLA: A Ground-Up Transformer Architecture for Flow-Matching Trajectory Generation")
    
    p.heading("Abstract")
    p.paragraph(
        "Vision-Language-Action (VLA) models have revolutionized robotic manipulation by tying language instructions "
        "to visual grounding and continuous action spaces. We present Manual VLA, a robust baseline architecture constructed "
        "entirely from scratch. Instead of relying on massive pretrained foundation models, Manual VLA employs a highly structured "
        "convolutional patch-embedding pipeline combined with Hadamard intent conditioning and Optimal Transport Conditional "
        "Flow Matching (OT-CFM). This paper dissects the architectural formulation, spatial grounding mechanics, and ODE-based "
        "generation pipeline of Manual VLA."
    )

    p.heading("1. Problem Formulation")
    p.paragraph(
        "The objective of the robotic agent is to generate a continuous 4D action trajectory over a horizon H=128 "
        "given an RGB observation and a discrete intent prompt. The action space encompasses the 3D translation of the "
        "end-effector (dx, dy, dz) and a continuous scalar for gripper state. The state space is normalized using offline "
        "dataset statistics to ensure stable diffusion dynamics."
    )

    p.heading("2. Architecture Design")
    p.paragraph(
        "The core of Manual VLA is a fully trainable Transformer encoder network. The vision pipeline begins with a "
        "custom convolutional patch embedder (Conv2d, kernel_size=4, stride=4) which tokenizes a 64x64 RGB observation "
        "into a 16x16 grid of 256 visual patches. A language prompt is mapped to an intent embedding, representing the "
        "target object and destination context."
    )
    p.figure(os.path.join(FIG_DIR, "fig1_manual_vla_arch.png"), "Figure 1: 3D tensor block visualization of the Manual VLA Architecture showing data geometric flow from the input 2D image plane, fracturing into 3D patch pillars, merging with intent strips, and routing through translucent Transformer decoder layers.")

    p.sub_heading("2.1 Offline Network Activation Extraction")
    p.paragraph(
        "To provide rigorous insight into the network topology, we execute a forward pass offline, capturing the precise "
        "dimensionality of every sub-layer. The summary table below illustrates the hierarchical transformation from the "
        "input tensor geometries to the final 128x4 trajectory output."
    )
    p.code_block(load_summary("manual_vla_summary.txt"))

    p.heading("3. Multimodal Fusion & Visual Grounding")
    p.paragraph(
        "Manual VLA does not inherently understand language; it must learn to ground intent vectors into raw pixels. "
        "We inject spatial priors into the network via an auxiliary VLM Cross-Attention module. The intent embeddings "
        "act as queries attending to the 256 visual patch keys/values. Below is a real attention matrix extracted from "
        "a live inference pass of the Manual VLA, demonstrating emergent spatial localization."
    )
    p.figure(os.path.join(FIG_DIR, "real_manual_vla_attn.png"), "Figure 2: Real inference extraction of Manual VLA spatial grounding heatmaps over the 16x16 visual grid.")

    p.heading("4. Optimal Transport Conditional Flow Matching (OT-CFM)")
    p.paragraph(
        "Trajectory generation is formulated as a continuous normalizing flow. Let x_0 be pure Gaussian noise and x_1 "
        "be the normalized target trajectory. The OT-CFM linear interpolation path is defined by:"
    )
    p.math_block("x_t = (1 - t) * x_0 + t * x_1")
    p.paragraph(
        "The neural network parameterizes a velocity vector field v_pred which regresses the optimal transport vector u_t:"
    )
    p.math_block("u_t = x_1 - x_0")
    p.paragraph(
        "At inference, the agent samples x_1 by integrating the learned vector field through 20 steps of Euler ODE integration. "
        "This approach bypasses the mode-collapse and instability often seen in standard autoregressive behavior cloning."
    )
    p.figure(os.path.join(FIG_DIR, "fig4_flow_matching.png"), "Figure 3: OT-CFM formulation depicting training interpolation and ODE integration at inference.")

    p.output(os.path.join(OUT_DIR, "manual_vla.pdf"))

# ==============================================================================
# 2. SmolVLA Paper
# ==============================================================================
def make_smolvla():
    p = AcademicPaper("SmolVLA: Dual-Stream Flow Matching with Frozen Vision-Language Prio")
    
    p.heading("Abstract")
    p.paragraph(
        "Building upon the foundational constraints of purely from-scratch models, SmolVLA introduces pre-trained knowledge "
        "distillation by leveraging a frozen OpenAI CLIP ViT-B/32 backbone. This dual-stream architecture processes raw language "
        "instructions and camera observations into a unified multimodal context via hierarchical cross-attention layers. "
        "The result is a highly parameter-efficient policy network that achieves robust zero-shot generalization while "
        "delegating continuous trajectory regression to a lightweight OT-CFM action expert."
    )

    p.heading("1. Architectural Overview")
    p.paragraph(
        "SmolVLA is composed of two primary modules: the pretrained Multimodal Backbone and the Flow-Matching Action Decoder. "
        "The backbone utilizes frozen CLIP weights, ensuring that the vast web-scale pre-training is completely preserved without "
        "catastrophic forgetting. Images are tokenized into 49 patches, and instructions are parsed into 77 byte-pair encoded tokens. "
        "These independent embeddings are projected to a 128-dimensional latent space before fusion."
    )
    p.figure(os.path.join(FIG_DIR, "fig2_smolvla_arch.png"), "Figure 1: 3D tensor block diagram of the SmolVLA Dual-Stream Architecture. Notice the cross-attention web binding the sequence of language strips to the spatial visual pillars before routing into the unified decoder block.")

    p.sub_heading("1.1 Extracted Network Dimensions")
    p.paragraph(
        "To explicitly define the model complexity and tensor routing, we provide a live network summary extracted "
        "from the initialized PyTorch module. Note the parameter efficiency obtained by freezing the CLIP components."
    )
    p.code_block(load_summary("smolvla_summary.txt"))

    p.heading("2. Cross-Attention Multimodal Fusion")
    p.paragraph(
        "The fundamental challenge of dual-stream architectures is semantic bridging. We implement a custom "
        "MultimodalCrossAttentionBlock wherein language tokens query the visual patches. While the raw OpenAI CLIP text encoder "
        "inherently pads all sequences to a fixed context length of 77 tokens, SmolVLA aggressively slices this down to just the "
        "first 16 tokens. This optimization drastically reduces computational overhead since robotic instructions are typically concise. "
        "Thus, the visual grounding explicitly maps 16 language tokens against the 7x7 patch grid."
    )
    p.figure(os.path.join(FIG_DIR, "real_smolvla_attention.png"), "Figure 2: Real CLIP Zero-Shot Attention decomposed. The left shows the actual 64x64 RGB inference image containing a red cube and green platform. The 4x4 grid on the right reveals the 2D spatial attention intensity for each of the 16 active language tokens, demonstrating how the model successfully grounds specific words to physical spatial coordinates.")
    
    p.paragraph(
        "As seen in the extraction above, the frozen CLIP weights innately cluster high-relevance patches corresponding "
        "to the object nouns in the prompt, allowing the downstream action decoder to isolate grasping coordinates instantaneously."
    )

    p.heading("3. Flow Matching Action Decoder")
    p.paragraph(
        "Once the multimodal tokens are fused, they are concatenated with proprioceptive and temporal state tokens to form "
        "a 67-token context block. A 2-layer Transformer decoder generates the optimal trajectory by casting 128 positional "
        "action queries against this context."
    )
    p.math_block("Loss = MSE( v_pred(x_t, t, Context), u_t )")
    p.paragraph(
        "We optimize the network utilizing an asymmetric weighting matrix [1.2, 1.2, 1.5, 2.5] to strictly penalize variance "
        "in Z-axis manipulation and gripper transitions, ensuring high-fidelity pick-and-place success rates."
    )

    p.output(os.path.join(OUT_DIR, "smolvla_dobot.pdf"))

# ==============================================================================
# 3. Full SmolVLA Paper
# ==============================================================================
def make_full_smolvla():
    p = AcademicPaper("Full SmolVLA: Foundation Action Experts on the SmolVLM-256M Backbone")
    
    p.heading("Abstract")
    p.paragraph(
        "Full SmolVLA represents the final architectural paradigm in our repository: the direct integration of an end-to-end "
        "Vision-Language Model (VLM) foundation backbone (SmolVLM-256M-Instruct). Instead of segregated dual-stream fusion, "
        "multimodal reasoning occurs innately within the 30-layer autoregressive transformer backbone. We introduce a novel "
        "multi-layer tapping architecture that extracts hierarchical spatial and semantic contexts, routing them into an OT-CFM "
        "Action Expert. This framework yields unparalleled generalization capabilities while demanding less than 1 million "
        "trainable parameters."
    )

    p.heading("1. Deep Architecture Formulation")
    p.paragraph(
        "The foundation backbone is completely frozen and loaded in hardware-efficient bfloat16 precision. Rather than "
        "extracting exclusively from the terminal layer, Full SmolVLA taps intermediate hidden states across layers 10, 20, and 30. "
        "Layer 10 provides low-level spatial geometry, Layer 20 provides relational semantic maps, and Layer 30 resolves the "
        "global task abstraction."
    )
    p.figure(os.path.join(FIG_DIR, "fig3_full_smolvla_arch.png"), "Figure 1: 3D tensor volume of the Full SmolVLA Architecture featuring the stacked 30-layer SmolVLM-256M backbone. Wires demonstrate intermediate layer extraction routing to the downstream Action Expert.")

    p.sub_heading("1.1 Network Graph & Parameter Allocation")
    p.paragraph(
        "The following architectural readout maps the flow from the concatenated 1728-dimensional multi-layer VLM features "
        "through the downstream projection network and onto the multi-head action decoder."
    )
    p.code_block(load_summary("full_smolvla_summary.txt"))

    p.heading("2. Trajectory Generation and Cross-Attention Mechanics")
    p.paragraph(
        "The Flow Matching Action Expert constructs its trajectory over a horizon of 128 continuous steps. The Action Decoder "
        "queries the foundation multi-layer tokens via 128 self-attending temporal query vectors. By inspecting the extracted "
        "attention matrices, we can observe precisely which multimodal tokens the action trajectory references during different "
        "temporal phases of the robotic motion."
    )
    p.figure(os.path.join(FIG_DIR, "real_full_smolvla_attention.png"), "Figure 2: Real Action Decoder Cross-Attention Weights (128 Trajectory Queries vs. VLM Context).")
    
    p.paragraph(
        "Figure 2 illustrates a live forward-pass extraction of the cross-attention layer. The network dynamically distributes "
        "attention: early trajectory states heavily attend to object grounding tokens, while terminal states strongly correlate "
        "with destination context and proprioceptive limits."
    )

    p.heading("3. Intermediate Foundation Layer Semantics")
    p.paragraph(
        "By tapping the SmolVLM backbone at different depths, the Action Expert gains access to a hierarchy of multimodal reasoning. "
        "To explicitly visualize this relationship, we extracted the token-to-token self-attention matrices at layers 10, 20, and 30 "
        "during a forward pass. The tokens are broadly partitioned into Vision Tokens (0-99) and Language Tokens (100-123). "
        "In Layers 10 and 20, the identity mapping (self-attention) and immediate neighbor attention weights were explicitly masked out (set to zero) "
        "to prevent them from saturating the heatmap. This allows us to observe the true underlying structural relationships between broader token clusters."
    )
    p.figure(os.path.join(FIG_DIR, "real_full_smolvla_intermediate.png"), "Figure 3: Evolution of Self-Attention across SmolVLM layers. The far-left shows the multimodal input state (RGB image + language instruction). Layer 10 exhibits local spatial geometry. Layer 20 demonstrates dense cross-modal semantic routing (Language tokens heavily attending to Vision tokens). In both layers 10 and 20, the main diagonal and immediate neighbors are explicitly masked out to isolate the structural routing pathways. Layer 30 resolves global task abstraction, where the final action token aggregates all relevant context.")

    p.paragraph(
        "To further dissect the dense semantic routing occurring in Layer 20, we isolate the text-to-vision attention block. "
        "Since the visual sequence inherently corresponds to a fine-grained 10x10 spatial patch grid (100 tokens), we can dynamically unfold the vision tokens back into their 2D spatial arrangement. "
        "Below is the spatial grounding intensity for the sequence of language tokens against the 10x10 visual patch grid. The explicitly annotated numerical values are raw and normalized across the entire global token space, preserving their relative magnitudes."
    )
    p.figure(os.path.join(FIG_DIR, "real_full_smolvla_layer20_spatial.png"), "Figure 4: Fine-grained spatial unfold of Layer 20 cross-modal attention. Each grid displays a text token's attention distributed across the 10x10 visual patch space. Values are explicitly annotated and retain their global normalization magnitudes.")

    p.heading("4. Conditional Flow Optimization")
    p.paragraph(
        "The network synthesizes the temporal ODE vector field using Euler integration. Crucially, by offloading the "
        "heavy multimodal inference to a pre-cached offline worker script, the training of the Action Expert completes in a fraction "
        "of the time required for traditional end-to-end architectures, rendering Full SmolVLA both deeply comprehensive and "
        "computationally lightweight."
    )

    p.output(os.path.join(OUT_DIR, "full_smolvla_dobot.pdf"))

if __name__ == '__main__':
    print("Compiling Formal Academic Papers...")
    make_manual_vla()
    print("  [OK] papers/manual_vla.pdf")
    make_smolvla()
    print("  [OK] papers/smolvla_dobot.pdf")
    make_full_smolvla()
    print("  [OK] papers/full_smolvla_dobot.pdf")
    print("All formal academic papers successfully compiled.")
