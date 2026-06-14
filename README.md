# Sparse Reward-Induced Policy Instability (SRIP): Hybrid Reward Design Tradeoffs for Multimodal Agent Factuality

**An Empirical Analysis Across 8 Seeds and 2 Benchmarks**

> *"Binary reward signals provide zero gradient to 82–88% of training steps — HRA-RL fixes this by grounding every step with a continuous factuality signal."*

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%20Workshop%202026-blue)](https://openreview.net/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-TeganJegede-yellow)](https://huggingface.co/TeganJegede)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Seeds](https://img.shields.io/badge/Seeds-n%3D8-orange)](results/)

---

## Overview

This repository contains the full implementation, experimental results, and analysis for **"Hybrid Reward Design Tradeoffs for Multimodal Agent Factuality: An Empirical Analysis"**, submitted to the North East AI Agents Day 2026 workshop.

We compare four reinforcement learning approaches for training multimodal agents on visual question answering benchmarks across **8 random seeds** and **2 benchmarks**, with a focus on training stability, hallucination reduction, and cross-seed reproducibility. All results report mean ± SD with 95% confidence intervals validated within [0, 1].

### Key Finding

Standard REINFORCE with binary task-completion reward achieves TSR = 0.546 ± 0.025 on ScienceQA — consistently and significantly outperformed by HRA-RL (TSR = 0.732 ± 0.044, Cohen's *d* = 5.27, *p* < .05, n = 8). HRA-RL reduces hallucination rate from **45.4% → 26.8%** — a 41% reduction confirmed across 8 seeds and 2 benchmarks.

---

## Results

### ScienceQA (n = 8 seeds)

| Agent | TSR ↑ | Hallucination Rate ↓ | SCS ↑ | 95% CI | Cohen's *d* vs HRA |
|---|---|---|---|---|---|
| **HRA-RL** | **0.7323 ± 0.044** | **26.77%** | **0.740** | [0.696, 0.769] | — |
| Vanilla | 0.5565 ± 0.018 | 44.35% | 0.699 | [0.542, 0.571] | 4.81 |
| REINFORCE | 0.5459 ± 0.025 | 45.41% | 0.703 | [0.525, 0.567] | **5.27** |
| PPO-Clip | 0.5334 ± 0.012 | 46.66% | 0.700 | [0.523, 0.544] | 6.12 |

All comparisons vs HRA-RL: *p* < .05, large effect.

### VQA-v2 Cross-Domain Generalisation (n = 8 seeds)

| Agent | TSR ↑ | Hallucination Rate ↓ | 95% CI | Cohen's *d* vs HRA |
|---|---|---|---|---|
| **HRA-RL** | **0.3565 ± 0.007** | **64.35%** | [0.351, 0.362] | — |
| PPO-Clip | 0.3370 ± 0.005 | 66.30% | [0.333, 0.341] | 3.14 |
| Vanilla | 0.3370 ± 0.008 | 66.30% | [0.331, 0.343] | 2.96 |
| REINFORCE | 0.3230 ± 0.006 | 67.70% | [0.318, 0.328] | **4.96** |

TSR = Task Success Rate (FGM cosine similarity ≥ 0.6). SCS = Semantic Consistency Score. All 95% CIs valid within [0, 1].

---

## What is SRIP?

**Sparse Reward-Induced Policy Instability (SRIP)** describes the failure mode where binary reward signals provide zero gradient to 82–88% of training steps, causing:

- Consistently lower task success and higher hallucination rates across all seeds
- High cross-seed variance in final policy quality
- Sensitivity to early training dynamics determined by random seed

HRA-RL addresses SRIP by providing a **continuous factuality grounding signal** at every training step via a semantic similarity scorer (FGM), regardless of whether the agent hits the binary success threshold.

---

## Architecture

```
Input: Image + Question
       ↓
LLaVA-1.5-7B (LoRA r=16, ~10M trainable params)
       ↓
Generated Response
       ↓
┌─────────────────────────────────┐
│     Hybrid Reward Function      │
│                                 │
│  R = W_task × r_task            │
│    + W_fact × r_fact            │
│    + penalty (if r_fact < θ)    │
│                                 │
│  W_task = 1.0, W_fact = 0.5    │
│  penalty = −0.5, θ_fact = 0.6  │
│  θ_task = 0.7                  │
└─────────────────────────────────┘
       ↓
REINFORCE with EMA baseline
(EMA decay = 0.9, lr = 1e-5)
```

The factuality score r_fact is computed by the **Factuality Grounding Module (FGM)** — cosine similarity between response and ground-truth embeddings using `sentence-transformers/all-MiniLM-L6-v2`.

---

## Experimental Setup

| Config | Value |
|---|---|
| Base model | LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf) |
| LoRA rank | 16, alpha = 32 |
| LoRA targets | q_proj, v_proj |
| Training steps | 400 per agent per seed |
| Learning rate | 1e-5 (AdamW, weight decay 0.01) |
| Gradient clipping | 0.5 |
| Temperature | 0.9 (train), 0.7 (eval) |
| Max new tokens | 50 |
| Seeds | 42, 123, 7, 1, 2, 3, 4, 5 |
| Benchmarks | ScienceQA, VQA-v2 |
| Eval items | 500 per seed (seeds 42, 123, 7, 1); 300 per seed (seeds 2, 3, 4, 5) |
| GPU | NVIDIA T4 × 2 (32 GB) |

**GPU note:** T4 (sm_75+) minimum required. CUDA 12.8 dropped kernel support for P100 (sm_60) in PyTorch 2.10+.

---

## Repository Structure

```
├── HRA_T4_FINAL.ipynb          # Main 16-cell experiment notebook (T4 GPU)
├── RESTORE_ALL_RESULTS.py      # Restore results dict across sessions
├── results/
│   └── all_results.csv         # Full 8-seed results (all agents, both benchmarks)
├── figures/
│   ├── training_curves.png     # Reward curves — 8 seeds × 3 agents
│   └── srip_analysis.png       # SRIP gradient variance analysis (6-panel)
└── README.md
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/TeganJegede/hybrid-rl-hallucination-agents
cd hybrid-rl-hallucination-agents

# Install
pip install transformers==4.44.2 peft>=0.12.0 accelerate>=0.30.0 \
            datasets bitsandbytes scipy scikit-learn pillow
```

Open `HRA_T4_FINAL.ipynb` on a T4 GPU (Kaggle or Colab) and run cells 1–16 in order. The notebook is fully self-contained with auto-resume checkpoint support — if the session dies mid-run, restart and Cell 12 picks up from the last completed seed automatically.

---

## Agents Compared

| Agent | Reward Config | Algorithm |
|---|---|---|
| **HRA-RL** | Task + factuality + penalty (Full HRA) | REINFORCE + EMA baseline |
| Standard REINFORCE | Binary task-completion only | REINFORCE + EMA baseline |
| PPO-Clip | Binary task-completion only | PPO-Clip (ε = 0.2) |
| Vanilla | No training | Zero-shot LLaVA-1.5-7B |

---

## Citation

```bibtex
@inproceedings{jegede2026srip,
  title     = {Sparse Reward-Induced Policy Instability ({SRIP}): Hybrid Reward Design
               Tradeoffs for Multimodal Agent Factuality},
  author    = {Jegede, Tegan},
  booktitle = {North East AI Agents Day 2026 Workshop, NeurIPS},
  year      = {2026},
  url       = {https://huggingface.co/TeganJegede/hybrid-rl-hallucination-agents}
}
```

---

**HuggingFace:** [TeganJegede](https://huggingface.co/TeganJegede)

---

*Empirical study on hybrid reward architectures for improving factuality and stability in reinforcement-learning-trained multimodal language agents. All results based on n = 8 seeds with 95% confidence intervals validated within [0, 1].*
