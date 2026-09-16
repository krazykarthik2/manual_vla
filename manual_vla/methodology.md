# Manual VLA: Autonomous Robotic Manipulation Methodology

This document details the architectural principles, mathematical formulation, visual tokenization, proprioception conditioning, and training methodologies developed for the **Manual VLA** 4-DOF Dobot manipulation system.

---

## 1. System Overview & Problem Formulation

The objective is to achieve reliable, multi-task, high-precision manipulation on a simulated 4-DOF Dobot arm across variable table scenes with clutter and distractors. The agent must successfully:
1. **Pick and Place**: Locate the specified target colored cube, grasp it, elevate it, translate to the specified target colored platform, and place it gently without dropping.
2. **Push Towards**: Navigate behind the specified colored cube and push it across the workspace onto the specified target platform.

### Constraints & Design Principles
- **Overhead Camera Only**: Single overhead camera rendering at 64 x 64 RGB pixels.
- **Robot Proprioception**: End-effector state [x, y, z, yaw, gripper] in R^5.
- **No Natural Language / No LLMs**: Low-latency, deterministic intent embeddings R^20 encoding discrete action types and target colors.
- **Physical Verification**: Trajectories must complete in full before evaluating placement success (no premature termination heuristics).

---

## 2. Mathematical Formulation: Optimal Transport Flow-Matching

Instead of autoregressive generation or standard diffusion ODEs requiring scores/denoising steps, the policy is parameterized as an **Optimal Transport Flow Matching (OT-CFM)** vector field regressor.

### 2.1 Trajectory Probability Path
Let x_1 in R^{H x D} be an expert trajectory sampled from demonstration distribution q(x_1) (with horizon H=128, action dimension D=4), and let x_0 ~ N(0, I) be Gaussian noise.

The linear Optimal Transport probability path between noise and data is:
x_t = (1 - t)x_0 + t * x_1,  t in [0, 1]

### 2.2 Target Vector Field
The conditional vector field driving noise x_0 to target trajectory x_1 is constant along the straight line:
u_t(x_t | x_0, x_1) = x_1 - x_0

### 2.3 Training Objective
The neural network v_theta(x_t, t, c) is trained via mean squared error regression to match the true vector field conditioned on observation context c:
L_CFM(theta) = E_{t, x_0, x_1} [ || v_theta(x_t, t, c) - (x_1 - x_0) ||_2^2 ]

### 2.4 Continuous Euler ODE Sampling
During inference, a trajectory is sampled by starting from x(0) ~ N(0, I) and integrating forward with N=20 Euler steps:
x(t + dt) = x(t) - v_theta(x(t), t, c) * dt,  dt = 1 / N
Followed by standard deviation un-normalization and temporal 1D smoothing across the action horizon.

---

## 3. Architecture & Tokenization

The policy employs a **Cross-Attention Transformer Decoder** (inspired by Action Chunking Transformers / ACT) processing visual, goal, and proprioceptive tokens simultaneously.

`
                  +--------------------------------------------------+
                  |               Overhead Camera (64x64x3)          |
                  +--------------------------------------------------+
                                           |
                                [Conv2d 4x4, Stride 4]
                                           |
                              256 Visual Tokens [B, 256, 128]
                                           +
                            2D Sinusoidal Positional Embeddings
                                           |
                                           v
[Intent Embedding (20)] --> [MLP] --> [Intent Token (1)]  \
[Proprioception (5)]    --> [MLP] --> [Proprio Token (1)]  --> [Cross-Attention Memory: 259 Tokens]
[Continuous Time (t)]   --> [Sin/Cos]>[Time Token (1)]    /
                                                                          |
                                                                          | (Cross-Attention)
                                                                          v
    Learned Horizon Pos Queries [1, 128, 128]  --> [4x Cross-Attention Decoder Blocks]
    + Action In Projection x_t [B, 128, 128]                               |
                                                                           v
                                                            [Output Linear Head]
                                                                           |
                                                    Predicted Vector Field v_theta [B, 128, 4]
`

### 3.1 Visual Patch Tokenizer
- **Input**: Overhead image tensor I in R^{3 x 64 x 64}.
- **Patch Extraction**: 2D Convolution with kernel size 4 and stride 4 to produce a 16 x 16 grid of feature vectors with d_model = 128.
- **Token Count**: 16 x 16 = 256 spatial tokens.
- **Positional Encoding**: Fixed 2D sinusoidal embeddings computed across row and column axes independently.

### 3.2 Context & Memory Tokens
1. **Discrete Intent Embedding**: Encodes task category (one-hot 2D for pick/push) concatenated with one-hot vectors for target cube color (9D) and target platform color (9D) -> R^20. Projected via 2-layer MLP to [B, 1, 128].
2. **Proprioception Embedding**: Real-time arm state [x, y, z, yaw, gripper] in R^5 projected via 2-layer MLP to [B, 1, 128].
3. **Continuous Diffusion Time**: Scalar t in [0, 1] expanded with 64-dim sinusoidal embeddings and projected to [B, 1, 128].
4. **Memory Sequence**: Formulates a joint context sequence of length 256 + 3 = 259 tokens.

### 3.3 Transformer Decoder Specifications
- **Number of Layers**: 4 Decoder Layers.
- **Attention Heads**: 4 heads per layer.
- **Hidden Dimension (d_model)**: 128.
- **Feedforward Dimension (d_ff)**: 256 (GELU activations).
- **Query Length**: Fixed action horizon queries H = 128.

---

## 4. Training Pipelines & Loss Dynamics

### 4.1 Fasttrain Pipeline (asttrain.bat)
The complete training pipeline comprises three stages:
1. **Demonstration Collection**: 80 automated synthetic demonstrations (40 Pick & Place, 40 Push) with randomized scene layouts and distractors.
2. **Behavioral Cloning Pre-training (200 Epochs)**:
   - Batch size: 16
   - Optimizer: AdamW, learning rate 1.8e-3
   - Loss: OT-CFM vector field regression over all 80 episodes.
3. **Demo-Anchored RL Fine-Tuning (150 Episodes)**:
   - Policy updates combine environment advantage guidance with a **70% Demonstration Replay Anchor**:
     L_total = 0.70 * L_demo_anchor + 0.30 * L_rl_advantage
   - Demo anchor samples mini-batches from recorded expert trajectories, keeping the vector field aligned with the kinematically smooth demonstration manifold.

### 4.2 Pure RL Pipeline (onlyrl.bat)
- Online policy optimization executing solely against environment rollouts without demonstration regularization.
- In long horizons without imitation anchoring, sparse spatial rewards lead to noisy gradient estimates and arm instability.

---

## 5. Physical Verification & Task Success Evaluation

To prevent false positives, task success is evaluated strictly after full trajectory completion:
- Cube center-to-platform center 2D Euclidean distance:
  || p_cube^{xy} - p_plat^{xy} ||_2 < 0.040 m
- Cube resting elevation (ensuring cube has been lowered and resting on table/platform surface):
  z_cube <= 0.025 m
- Gripper opened:
  gripper_state == False
- If and only if all physical conditions hold at the end of the trajectory, the episode is marked as **SUCCESS**.

---

## 6. Comparative Benchmark: Pure RL vs. Fasttrain Pipeline

Evaluated across **50 identical randomized test scenarios** (fixed pseudo-random seeds, dynamic cube/platform placements, and multiple distractors):

| Benchmark Metric | Pure RL Baseline (onlyrl.bat) | Full Fasttrain Pipeline (asttrain.bat) |
| :--- | :---: | :---: |
| **Task Placement Success Rate** | **0.0%** (0 / 50) | **26.0%** (13 / 50) |
| **Grasp Initiation Rate** | 4.0% | **42.0%** |
| **Average Min Distance to Platform** | 0.1517 m | **0.0926 m** |
| **Policy Kinematic Stability** | Unstable / Chaotic drift | Smooth kinematic execution |

### Findings & Insights:
1. **The Role of the Demo Anchor**: Without pre-training demonstrations or imitation anchoring, pure RL experiences reward sparsity over the 128-step continuous 4-DOF horizon, yielding 0% successful task completions.
2. **Fasttrain Stability**: The BC + 70/30 Demo-Anchored RL maintains smooth reaching and grasping arcs learned from expert demos while fine-tuning terminal placement accuracy.
