# Full SmolVLA with Real SmolVLM Backbone

This folder (`full_smolvla_dobot`) implements the genuine **SmolVLA** architecture utilizing the **Hugging Face SmolVLM** (`HuggingFaceTB/SmolVLM-256M-Instruct`) foundation vision-language model.

---

## Architecture Flow

```
[ Camera RGB (64x64) -> PIL ]   [ Natural Language Task Instruction ]
                   │                     │
                   └──────────┬──────────┘
                              ▼
        ┌──────────────────────────────────────────────┐
        │   Pretrained SmolVLM Foundation Backbone     │
        │   (HuggingFaceTB/SmolVLM-256M-Instruct)      │
        │   - Vision Transformer Encoder               │
        │   - Multimodal Connector                     │
        │   - Autoregressive Language Transformer      │
        └──────────────────────────────────────────────┘
                              │  Multimodal Hidden States (last layer)
                              ▼
            vlm_proj (Linear + LayerNorm: 576 -> 128)
                              │
                              ▼  Multimodal Context Tokens
           [ + Proprioception (5D -> 128) + Time (64 -> 128) ]
                              │
                              ▼
        ┌──────────────────────────────────────────────┐
        │   Flow-Matching Action Decoder Head          │
        │   (Transformer Cross-Attention, Horizon=128) │
        └──────────────────────────────────────────────┘
                              │
                              ▼
           Predicted Continuous 128-Step Trajectory
                 4D Actions: [dx, dy, dz, grip]
```

---

## File Manifest

| File | Purpose |
|---|---|
| `full_smolvla_model.py` | Implementation of `FullSmolVLAPolicy` powered by `SmolVLMForConditionalGeneration` and cross-attention action decoder head |
| `train_full_smolvla.py` | Training script using cached SmolVLM multimodal embeddings and flow-matching velocity loss |
| `run_full_smolvla.py` | Real-time simulator loop running closed-loop evaluation |
| `auto_generate_demos.py` | Clean rule-based demonstration generator |
| `fasttrain_full_smolvla.bat`| 1-click script to generate demos, cache SmolVLM tokens, train policy, and launch runner |
| `run_full_smolvla.bat` | 1-click script to launch the controller |
| `delete_model.bat` | Cleans up saved model checkpoints |
| `delete_recordings.bat` | Cleans up demonstration datasets |
| `env/` | Analytical kinematics and 4-DOF Dobot environment |
