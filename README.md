# Forward-Forward Algorithm Research

> Systematic study of negative sample strategies and transfer learning in Hinton's Forward-Forward algorithm.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project investigates two key aspects of the Forward-Forward (FF) algorithm:

1. **Negative Sample Strategies** — First systematic comparison of 10 different strategies
2. **Transfer Learning** — Why FF fails at transfer and whether Layer Collaboration helps

**Baseline Verification:** mpezeshki implementation achieves **93.25% MNIST** (1000 epochs/layer)

---

## Core Findings

### 1. CKA Analysis: Catastrophic Layer Disconnection

| Metric | FF | BP | Gap |
|--------|----|----|-----|
| Layer 0 ↔ Layer 2 | **0.025** | 0.39 | 15.6× worse |
| Avg Self-CKA | 0.264 | 0.592 | 2.2× worse |
| Layer 2 vs BP L2 | 0.038 | — | Nearly alien |

**Insight:** FF layers learn in isolation — no information flows between layers.

```
FF:  L0 ←0.72→ L1 ←0.05→ L2   (broken chain)
BP:  L0 ←0.63→ L1 ←0.74→ L2   (coherent flow)
```

### 2. Transfer Learning: FF Weights Are Harmful

MNIST → Fashion-MNIST transfer:

| Method | Transfer Acc | vs Random Init |
|--------|--------------|----------------|
| BP pretrained | 73.19% | −7.41% |
| Random init | **80.60%** | baseline |
| FF pretrained | 13.47% | **−67.13%** 🔴 |
| FF + Layer Collab | 10.21% | −70.39% |

**Insight:** FF pretrained weights perform worse than random — they're harmful, not helpful.

### 3. Negative Sample Strategy Comparison

| Strategy | Accuracy | Uses Labels | Note |
|----------|----------|-------------|------|
| label_embedding | **38.81%** | ✓ | Hinton's original |
| class_confusion | **38.81%** | ✓ | 30% faster |
| random_noise | 9.80% | ✗ | Random chance |
| image_mixing | 9.80% | ✗ | Random chance |
| masking | 8.75% | ✗ | Random chance |
| adversarial | 8.75% | ✗ | Random chance |
| mono_forward | 1.10% | ✓ | No negatives → fails |

**Insight:** Label embedding is required for standard FF evaluation. Non-label strategies need linear probe evaluation.

---

## Experiment Status

| Experiment | Status | Key Result |
|------------|--------|------------|
| Baseline verification | ✅ Done | 93.25% MNIST |
| CKA analysis | ✅ Done | L0-L2 CKA = 0.025 |
| Transfer learning | ✅ Done | FF worse than random |
| Layer Collaboration | ✅ Done | Doesn't help transfer |
| Strategy comparison (9/10) | ✅ Done | Label embedding wins |
| self_contrastive | 🔄 WIP | Needs linear probe |
| Linear probe for all | 📋 Planned | — |
| CIFAR-10 experiments | 📋 Planned | — |

---

## Project Structure

```
ff-research/
├── negative_strategies/    # 10 strategy implementations
│   ├── base.py            # Base class & registry
│   ├── label_embedding.py # Hinton's original
│   ├── class_confusion.py # Wrong label embedding
│   ├── random_noise.py    # Noise baseline
│   ├── self_contrastive.py# SCFF-style augmentation
│   └── ...
├── experiments/           # Experiment scripts
│   ├── ff_baseline.py     # FF baseline
│   ├── transfer_experiment.py
│   ├── strategy_comparison.py
│   └── cka_linear_probe_experiment.py
├── analysis/              # Analysis tools
│   ├── cka_analysis.py    # CKA similarity
│   └── linear_probe.py    # Linear probing
├── results/               # Outputs
│   ├── visualizations/    # CKA heatmaps
│   └── *.json             # Experiment results
├── literature/            # Paper analyses
├── KEY_FINDINGS.md        # Detailed findings
└── EXPERIMENTS.md         # Experiment log
```

---

## Quick Start

### Installation

```bash
cd ff-research
python -m venv venv
source venv/bin/activate
pip install torch torchvision matplotlib seaborn
```

### Usage

```python
from negative_strategies import LabelEmbeddingStrategy, ClassConfusionStrategy

# All strategies share the same interface
strategy = LabelEmbeddingStrategy(num_classes=10)
positive = strategy.create_positive(images, labels)
negative = strategy.generate(images, labels)
```

### Run Experiments

```bash
# Strategy comparison
python experiments/strategy_comparison.py

# CKA analysis
python experiments/cka_linear_probe_experiment.py

# Transfer learning
python experiments/transfer_experiment.py
```

---

## References

- Hinton (2022). [The Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345)
- Lorberbom et al. (2024). [Layer Collaboration in FF](https://ojs.aaai.org/index.php/AAAI/article/view/29307). AAAI 2024
- Brenig et al. (2023). [Self-Contrastive FF](https://arxiv.org/abs/2309.11955)

---

## License

MIT — [Shuaizhi Cheng](https://github.com/koriyoshi2041)
