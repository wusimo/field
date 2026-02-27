# StarVLA Architecture Deep Dive

## Overview

**StarVLA** is a modular, open-source framework for developing **Vision-Language-Action (VLA)** models — neural networks that take camera images and natural language instructions as input and predict robot actions as output. It follows a "Lego-like" plug-and-play design where each component (VLM backbone, action head, dataloader, trainer) can be swapped independently.

The repository implements **8 distinct VLA architectures**, all built on top of the **Qwen2.5-VL-3B** vision-language model as the shared backbone. They differ in *how they decode actions* from the VLM's representations. Four are the primary variants (OFT, FAST, GR00T, PI), and four are advanced/experimental (Dual, Adapter, M1, LangForce).

---

## Repository Structure

```
starVLA/
├── config/                    # YAML configs (training, DeepSpeed)
│   ├── training/              # starvla_cotrain_oxe.yaml, etc.
│   └── deepseeds/             # DeepSpeed ZeRO-2/3 configs
├── dataloader/                # Data loading pipelines
│   ├── lerobot_datasets.py    # LeRobot-format VLA data
│   ├── vlm_datasets.py        # VLM co-training data (LLaVA JSON)
│   ├── gr00t_lerobot/         # GR00T-style data transforms
│   └── qwenvl_llavajson/      # Qwen-VL data processing
├── model/
│   ├── framework/             # THE 4 VLA ARCHITECTURES (see below)
│   │   ├── base_framework.py  # Base class (PreTrainedModel)
│   │   ├── QwenOFT.py         # Qwen-OFT (MLP regression)
│   │   ├── QwenFast.py        # Qwen-FAST (discrete tokens)
│   │   ├── QwenGR00T.py       # Qwen-GR00T (flow matching)
│   │   ├── QwenPI.py          # Qwen-PI (layerwise flow matching)
│   │   ├── QwenDual.py        # Dual-system variant
│   │   ├── QwenAdapter.py     # Adapter-based variant
│   │   ├── M1.py, LangForce.py
│   │   └── __init__.py        # Framework registry & factory
│   └── modules/               # Reusable building blocks
│       ├── vlm/               # VLM backends (QWen2_5, QWen3, Florence2, CosmosReason2)
│       ├── action_model/      # Action heads (MLP, DiT, FlowMatching, FAST)
│       ├── projector/         # Q-Former cross-attention projector
│       └── dino_model/        # DINOv2 vision encoder
├── training/
│   ├── train_starvla.py       # Single-task training
│   ├── train_starvla_cotrain.py  # Multi-task co-training (VLA + VLM)
│   └── trainer_utils/         # Optimizer, scheduler, logging
├── deployment/                # Model serving (WebSocket policy server)
└── examples/                  # Benchmark configs (LIBERO, RoboCasa, CALVIN, etc.)
```

---

## High-Level Data Flow

All 4 architectures share the same high-level pipeline:

```
Input:                     Processing:                        Output:
┌──────────────┐     ┌──────────────────────┐          ┌──────────────┐
│ Multi-view   │────>│  Qwen2.5-VL Backbone │─────────>│  Action Head │───> Robot Actions
│ Images       │     │  (Vision Encoder +   │          │  (varies by  │    [B, T, action_dim]
│ [PIL images] │     │   Language Model)    │          │  framework)  │
├──────────────┤     │                      │          └──────────────┘
│ Language     │────>│  Produces hidden     │
│ Instruction  │     │  states [B, L, H]    │
│ (str)        │     └──────────────────────┘
├──────────────┤
│ Actions      │ (training labels only)
│ [T, 7]       │
└──────────────┘
```

Each training example is a dictionary:
```python
{
    "image": [PIL.Image, ...],           # Multi-view camera images
    "lang": "Pick up the red block",     # Natural language instruction
    "action": np.ndarray (T, action_dim),# Ground-truth action trajectory
    "state": np.ndarray (1, state_dim),  # Optional proprioceptive state
}
```

---

## The 4 Primary VLA Architectures

### 1. Qwen-OFT (Parallel MLP Regression)

**File:** `starVLA/model/framework/QwenOFT.py`
**Inspired by:** [OpenVLA-OFT](https://github.com/moojink/openvla-oft)

**How it works:**
1. Append special "action tokens" (the emoji character repeated `chunk_len` times) to the instruction text
2. Feed images + augmented instruction through Qwen2.5-VL
3. Extract hidden states at the positions of the action tokens from the last layer
4. Pass these hidden states through an **MLP ResNet** action head to predict continuous actions
5. Train with **L1 loss** between predicted and ground-truth actions

```
Images + "Pick up block. Predict next 16 actions: <action>🔍🔍🔍...🔍<action>"
                    │
                    ▼
            ┌───────────────┐
            │  Qwen2.5-VL   │
            │  (full model)  │
            └───────┬───────┘
                    │ hidden_states[-1] at action token positions
                    ▼
            ┌───────────────┐
            │  MLP ResNet   │  (LayerNorm → Linear → ReLU → ResBlocks → Linear)
            │  Action Head  │
            └───────┬───────┘
                    │
                    ▼
            [B, chunk_len, 7]  ← continuous action predictions
```

**Key detail:** The action head is a simple 2-block MLP ResNet. Each block does `LayerNorm → Linear → ReLU + residual`. The input dim matches the VLM's hidden size (2048 for Qwen2.5-VL-3B).

**Action token extraction:** Uses vectorized `torch.topk` to gather the last `chunk_len` action token embeddings from the hidden states, sorted by sequence position.

---

### 2. Qwen-FAST (Autoregressive Discrete Tokens)

**File:** `starVLA/model/framework/QwenFast.py`
**Inspired by:** [pi0-FAST](https://www.physicalintelligence.company/blog/pi0-fast) (Physical Intelligence)

**How it works:**
1. Use the **FAST tokenizer** (from Physical Intelligence) to convert continuous actions into discrete token sequences
2. Map these discrete tokens to special VLM tokens: `<robot_action_0>`, `<robot_action_1>`, ..., `<robot_action_N>`
3. Train the VLM with standard **next-token prediction loss** (cross-entropy) — the same loss used for language
4. At inference, use `model.generate()` to autoregressively sample action tokens, then decode back to continuous actions

```
Training:
  actions [16, 7] ──→ FAST tokenizer ──→ "<robot_action_12><robot_action_3>..."
                                                    │
  Images + Instruction ──→ Qwen2.5-VL ──→ next-token prediction loss (CE)
                                                    │
                                            (learns to generate action tokens)

Inference:
  Images + Instruction ──→ Qwen2.5-VL.generate() ──→ "<robot_action_12><robot_action_3>..."
                                                              │
                                          FAST tokenizer.decode() ──→ continuous actions [T, 7]
```

**Key detail:** The FAST tokenizer (loaded from HuggingFace `physical-intelligence/fast`) discretizes continuous robot actions into a vocabulary of tokens. Special tokens `<robot_action_0>` through `<robot_action_N>` are added to the Qwen2.5-VL tokenizer vocabulary (token IDs 151665–153712).

**No separate action head** — the VLM's own language modeling head generates action tokens directly.

---

### 3. Qwen-GR00T (Flow Matching)

**File:** `starVLA/model/framework/QwenGR00T.py`
**Inspired by:** [NVIDIA GR00T N1.5](https://developer.nvidia.com/isaac/gr00t)

**How it works:**
1. Feed images + instruction through Qwen2.5-VL to get hidden states
2. Pass the **last-layer hidden states** as conditioning to a **Flow Matching action head**
3. The action head is a **DiT (Diffusion Transformer)** with cross-attention to VLM features
4. During training: sample random timesteps, add noise to actions, predict the velocity field (flow matching loss)
5. During inference: iterative denoising from pure noise using learned flow

```
Images + Instruction
        │
        ▼
┌───────────────┐
│  Qwen2.5-VL   │ ──→ hidden_states[-1]: [B, L, 2048]
│  (System 2)   │                │
└───────────────┘                │ cross-attention conditioning
                                 ▼
                    ┌──────────────────────┐
                    │  Flow Matching Head   │
                    │  (DiT Transformer)    │
                    │  - ActionEncoder      │  ← encodes noisy actions + timestep
                    │  - N × {              │
                    │      AdaLN            │  ← timestep-conditioned normalization
                    │      Self-Attention   │
                    │      Cross-Attention  │  ← attends to VLM hidden states
                    │      FeedForward      │
                    │    }                  │
                    │  - Final projection   │
                    │  (System 1)           │
                    └──────────┬───────────┘
                               │
                               ▼
                    [B, chunk_len, action_dim]  ← denoised continuous actions
```

**Key details:**
- Uses **Beta distribution** noise scheduling (not uniform) for timestep sampling
- Supports **repeated diffusion steps** (default 4) — the same batch is processed multiple times with different noise levels to amortize the VLM forward cost
- The DiT uses **AdaLayerNorm** conditioned on the diffusion timestep
- Optional **state conditioning** (proprioceptive robot state) via concatenation
- Supports **multi-embodiment** via `CategorySpecificLinear` layers (per-robot-type weights)
- At inference, uses a small number of denoising steps (default 4) for real-time performance

---

### 4. Qwen-PI (Layerwise Flow Matching)

**File:** `starVLA/model/framework/QwenPI.py`
**Inspired by:** [pi0](https://www.physicalintelligence.company/blog/pi0) (Physical Intelligence)

**How it works:**
Similar to GR00T but with a critical difference: instead of only using the last VLM layer, it uses **multiple VLM hidden layers** — each DiT transformer block cross-attends to a *different* VLM layer.

```
Images + Instruction
        │
        ▼
┌───────────────┐
│  Qwen2.5-VL   │ ──→ hidden_states: [layer_0, layer_1, ..., layer_N]
│  (all layers)  │           │         │              │
└───────────────┘           │         │              │
                             ▼         ▼              ▼
                    ┌────────────────────────────────────┐
                    │  Layerwise Flow Matching Head       │
                    │  DiT Block 0 ← cross-attn layer_k  │
                    │  DiT Block 1 ← cross-attn layer_k+1│
                    │  ...                                │
                    │  DiT Block N ← cross-attn layer_N   │
                    └──────────────┬─────────────────────┘
                                   │
                                   ▼
                    [B, chunk_len, action_dim]
```

**Key difference from GR00T:** The number of DiT blocks equals the number of VLM layers used. Each DiT block receives conditioning from a *different* depth of the VLM, giving the action head access to both low-level visual features (early layers) and high-level semantic features (later layers).

---

---

## 4 Advanced/Experimental Architectures

### 5. Qwen-Dual (Dual Encoder: Qwen VL + DINOv2)

**File:** `starVLA/model/framework/QwenDual.py`
**Registry name:** `"QwenDual"`

**How it works:**
Extends GR00T by adding a **DINOv2** vision encoder alongside the VLM. The DINOv2 features provide dense spatial tokens that complement the VLM's semantic understanding.

1. Run Qwen2.5-VL to get hidden states at a configurable layer
2. Process images (or separate wrist camera views) through DINOv2
3. Project DINO features to VLM hidden size via a linear layer
4. **Concatenate** VLM + DINO features along the sequence dimension → `[B, L_vlm + L_dino, H]`
5. This combined tensor conditions the same Flow Matching action head as GR00T

**Key difference:** Adds dense spatial features from a frozen DINOv2 encoder. Supports separate wrist-camera views for manipulation tasks.

---

### 6. Qwen-Adapter (Learnable Action Queries + Deep MLP)

**File:** `starVLA/model/framework/QwenAdapter.py`
**Registry name:** `"QwenAdapter"`

**How it works:**
The most architecturally complex variant. Uses **learnable action query tokens** injected directly into the VLM's embedding space via forward hooks, and a **deep 24-block MLP ResNet** action head with per-layer cross-attention.

1. Insert `action_query_num` (default 64) dummy action tokens into the instruction
2. Register a **forward hook** on the VLM embedding layer to replace dummy embeddings with learnable `action_query` parameters
3. Run VLM, extract **all hidden layers**
4. For each layer, extract vision features (image positions) and action query features (action token positions)
5. Stack into `[B, num_layers, L, H]` and feed through the 24-block MLP ResNet
6. Each block does **multi-head cross-attention** to the corresponding VLM layer's vision + action query features, then residual MLP
7. L1 loss

**Key difference:** Injects learnable queries into the VLM via embedding hooks (not special tokens). The 24-block deep action head with per-layer cross-attention is far larger than OFT's 2-block MLP.

---

### 7. InternVLA-M1 (Full Pipeline: QFormer + DINO + DiT Diffusion)

**File:** `starVLA/model/framework/M1.py`
**Registry name:** `"InternVLA-M1"`

**How it works:**
The original/fullest architecture. Combines Qwen VL + DINOv2 + a **Q-Former bottleneck** for multi-layer feature aggregation + traditional **DDPM/DDIM diffusion** (not flow matching).

1. Run VLM with `output_hidden_states=True`
2. Run DINOv2, project to VLM hidden size
3. For each layer in `[start_layer, end_layer]`, concatenate VLM + DINO features
4. Pass through **layer-wise Q-Former** → `[B, 64, D_action]` action condition embeddings
5. DDPM forward: add noise at random timestep, predict noise with DiT conditioned on QFormer output
6. MSE loss on noise prediction
7. At inference: DDIM sampling with configurable steps and **classifier-free guidance** (CFG)

**Key difference:** Uses traditional DDPM/DDIM diffusion, a Q-Former bottleneck, and supports classifier-free guidance. Most parameter-heavy variant.

---

### 8. LangForce (Bayesian Dual-Branch with Language LLR Regularization)

**File:** `starVLA/model/framework/LangForce.py`
**Registry name:** `"LangForce"`

**How it works:**
Implements a Bayesian decomposition with two branches that differ in **input ordering**:

- **Prior branch:** `[Vision + ActionQuery + Language]` — sees action queries before language, forming a vision-conditioned prior p(a|v)
- **Posterior branch:** `[Vision + Language + ActionQuery]` — standard order, forming the policy pi(a|v,l)

Both branches use the same VLM + Flow Matching head. A **Language Log-Likelihood Ratio (LLR)** regularizer encourages the prior's action queries to encode information useful for predicting the instruction:

```
LLR = log p(L|V, A_prior) - log p(L|V)
Total Loss = (1 - w) * posterior_loss + w * prior_loss - kl_weight * LLR
```

Features hard-token LLR (top-k hardest tokens only) and an adaptive shortcut gate.

**Key difference:** Only framework with a dual-branch design and information-theoretic regularizer. Based on arXiv 2601.15197. Only the posterior branch is used at inference time.

---

## Full Comparison Table

| Framework | Action Head | Action Repr. | VLM Layers | Extra Encoders | Inference | Loss |
|-----------|------------|-------------|-----------|----------------|-----------|------|
| **OFT** | MLP ResNet (2 blocks) | Continuous | Last | None | Single forward | L1 |
| **FAST** | None (VLM head) | Discrete tokens | All (autoregressive) | None | Token generation | Cross-entropy |
| **GR00T** | DiT + Flow Matching | Continuous | Last | None | Euler integration | Flow matching |
| **PI** | Layerwise DiT + FM | Continuous | Last N (per-block) | None | Euler integration | Flow matching |
| **Dual** | DiT + Flow Matching | Continuous | Configurable | DINOv2 | Euler integration | Flow matching |
| **Adapter** | Deep MLP (24 blocks) + cross-attn | Continuous | All layers | None | Single forward | L1 |
| **M1** | DiT + DDPM diffusion | Continuous | Layer range | DINOv2 + QFormer | DDIM sampling + CFG | MSE (noise) |
| **LangForce** | DiT + FM (x2 branches) | Continuous | Last | None | Euler (posterior) | FM + LLR |

---

## Shared VLM Backbone: Qwen2.5-VL Interface

**File:** `starVLA/model/modules/vlm/QWen2_5.py`

All frameworks use `_QWen_VL_Interface`, a wrapper around `Qwen2_5_VLForConditionalGeneration`:

- Loads pre-trained Qwen2.5-VL-3B with Flash Attention 2
- Provides `build_qwenvl_inputs()` to format multi-view images + text into the Qwen chat template
- Handles tokenization, image preprocessing, and attention masking
- For FAST: adds special `<robot_action_N>` tokens to the vocabulary (IDs 151665–153712)
- Left-pads sequences for batch processing

---

## Base Framework Class

**File:** `starVLA/model/framework/base_framework.py`

All 8 architectures inherit from `baseframework(PreTrainedModel)`:
- `from_pretrained()` — loads config, normalization stats, and weights from a checkpoint
- `unnormalize_actions()` — maps normalized actions ([-1,1]) back to original scale using q01/q99 percentile statistics
- `trainable_module_keys` — auto-discovers which submodules have trainable parameters
- Framework registry pattern: each architecture registers itself via `@FRAMEWORK_REGISTRY.register("name")`

---

## Training Pipeline

**File:** `starVLA/training/train_starvla_cotrain.py`

- Uses HuggingFace Accelerate for distributed training
- Supports DeepSpeed ZeRO-2 and ZeRO-3
- **Co-training**: can jointly train on VLA data (robot actions) and VLM data (image captioning, VQA) with configurable loss weighting (e.g., `vla: 1.0, vlm: 0.1`)
- Configurable per-module learning rates (e.g., `action_model: 1e-4`, `qwen_vl_interface: 1e-5`)
- Gradient checkpointing for memory efficiency
- Cosine LR scheduler with warmup

---

## Data Pipeline

Two data sources are supported simultaneously:

1. **VLA data** (`lerobot_datasets.py`): Robot manipulation data in LeRobot format
   - Returns `{image, lang, action, state}` dictionaries
   - Supports multiple datasets mixed together (e.g., `bridge_rt_1`)
   - Actions normalized to [-1, 1] using q01/q99 percentile statistics

2. **VLM data** (`vlm_datasets.py`): Vision-language data in LLaVA JSON format
   - Used for co-training to preserve VLM capabilities
   - Includes captioning, VQA, grounding tasks

---

## Configuration System

Single YAML entry point (e.g., `starvla_cotrain_oxe.yaml`):

```yaml
framework:
  name: QwenFM          # Selects which architecture to build
  qwenvl:
    base_vlm: ./Qwen2.5-VL-3B-Instruct  # VLM checkpoint path
  action_model:
    action_dim: 7                         # Robot action dimensionality
    future_action_window_size: 15         # How many future steps to predict
    past_action_window_size: 0
    # ... action-head-specific params

datasets:
  vla_data:
    data_mix: bridge_rt_1
    image_size: [224, 224]
  vlm_data:
    dataset_use: coco_karpathy, vqav2_en, ...

trainer:
  learning_rate:
    base: 1e-05
    action_model: 1e-04
  loss_scale:
    vla: 1.0
    vlm: 0.1
```

---

## Summary

StarVLA provides a clean abstraction for VLA research. The key insight is that a strong VLM backbone (Qwen2.5-VL) provides rich multimodal representations, and different action decoding strategies can be plugged in:

- **OFT**: Simplest — just regress actions from hidden states via MLP
- **FAST**: Reuses the VLM's own language modeling — zero extra parameters
- **GR00T**: Powerful diffusion-based decoding with cross-attention to VLM features
- **PI**: Most expressive primary variant — layerwise cross-attention gives the action head multi-scale VLM features
- **Dual**: Adds DINOv2 spatial features on top of GR00T
- **Adapter**: Deep action head with learnable query injection and per-layer cross-attention
- **M1**: Full pipeline with QFormer bottleneck, DINOv2, and traditional DDPM diffusion with CFG
- **LangForce**: Bayesian dual-branch with language-grounded regularization

All share the same data pipeline, training infrastructure, and deployment tooling.
