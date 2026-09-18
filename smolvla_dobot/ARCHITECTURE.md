# SmolVLA Architecture Overview

A compact, real-time **Vision-Language-Action (VLA)** model designed for robotic manipulation (Dobot Magician 4-DOF).

---

## High-Level Pipeline

```
[ Camera RGB (64x64) ]   [ Text Prompt ]        [ Proprioception (5D) ]   [ Time Step t ]
          │                     │                         │                      │
          ▼                     ▼                         │                      │
  Frozen CLIP ViT-B/32   Frozen CLIP Text                 │                      │
     (49 Patches)          (77 Tokens)                    │                      │
          │                     │                         │                      │
          ▼                     ▼                         │                      │
      vis_proj               txt_proj               proprio_proj            time_embed
     (512 -> 128)          (512 -> 128)              (5 -> 128)             (64 -> 128)
          │                     │                         │                      │
          └──────────┬──────────┘                         │                      │
                     ▼                                    │                      │
       Multimodal Cross & Self-Attention                  │                      │
                     │                                    │                      │
                     ▼                                    │                      │
         [ Text (16) + Vision (49) ] ◄────────────────────┴──────────────────────┘
                                │
                                ▼  Joint Context (67 tokens, dim=128)
             ┌─────────────────────────────────────────┐
             │   Flow-Matching Action Decoder Head     │
             │   (Transformer Cross-Attention, H=128)  │
             └─────────────────────────────────────────┘
                                │
                                ▼
                 Predicted Action Trajectory:
               128 steps × 4D [dx, dy, dz, grip]
```

---

## Key Components

### 1. Vision-Language Backbone (Frozen Pretrained CLIP ViT-B/32)
* **Vision Encoder**: Takes a $64 \times 64$ RGB image (bilinear upscaled to $224 \times 224$), processes it through ViT-B/32, and outputs **49 spatial visual patch tokens** ($7 \times 7$ grid) of dimension 512.
* **Text Encoder**: Processes tokenized task instructions (up to 77 BPE tokens) into 512-dim token representations.
* **Benefit**: Zero-shot open-vocabulary spatial semantic understanding without needing millions of robotic training demonstrations.

### 2. Linear Feature Projectors
* `vis_proj`: Projects visual patch embeddings from $512 \to 128$.
* `txt_proj`: Projects text tokens from $512 \to 128$ (first 16 tokens used).
* `proprio_proj`: Maps current robot state `[x, y, z, yaw, gripper]` ($5\text{D} \to 128$).
* `time_embed`: Continuous sinusoidal diffusion / flow time embedding ($64 \to 128$).

### 3. Multimodal Fusion Layers
* Stack of 2 blocks consisting of:
  1. **Cross-Attention**: Language queries attend to visual patch keys/values (text grounds into scene objects).
  2. **Joint Self-Attention**: Text and visual tokens interact globally.
  3. **Feed-Forward Network (FFN)** with GELU activations.

### 4. Flow-Matching Action Decoder (Action Expert)
* **Input**: Noisy trajectory queries $x_t$ over horizon $H = 128$, conditioned on time $t$ and proprioception.
* **Context**: Concatenated 67 tokens:
  `[proprio_token (1), time_token (1), text_tokens (16), visual_patches (49)]`
* **Architecture**: 2-layer Transformer decoder with self-attention and cross-attention over the multimodal context tokens.
* **Output**: Velocity vector field $v_\theta(x_t, t)$ predicting how to denoise from random Gaussian noise $x_0 \sim \mathcal{N}(0, I)$ to valid trajectory $x_1$ via Optimal Transport Flow Matching (OT-CFM).
* **Action Dimensions (4D)**:
  * End-effector position: `[x, y, z]`
  * Gripper command: `[grip]` (thresholded at 0.5 for open/close)
