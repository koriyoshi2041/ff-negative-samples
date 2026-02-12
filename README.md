# Why Forward-Forward Hasn't Become the New Paradigm

<div align="center">

**A Systematic Investigation into the Limitations of Biologically Plausible Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

<img src="figures/transfer_hero.png" width="800">

*Our key finding: CwC-FF achieves 89% transfer accuracy—the only biologically plausible method that outperforms random initialization.*

</div>

---

## Abstract

The Forward-Forward (FF) algorithm [Hinton, 2022] offers a biologically plausible alternative to backpropagation, but its practical limitations remain poorly understood. This work presents a **systematic investigation** of FF across three dimensions: (1) negative sampling strategies, (2) transfer learning capabilities, and (3) neuroscience-inspired architectural variants.

We uncover a **critical paradox**: standard FF features transfer *worse than random initialization* due to label-embedding coupling. We demonstrate that **Channel-wise Competitive FF (CwC-FF)** resolves this limitation, achieving 89.05% transfer accuracy on MNIST→Fashion-MNIST—surpassing both standard FF (54.19%) and backpropagation (75.49%). We further implement and evaluate five bio-inspired variants based on cutting-edge neuroscience research (2024-2025).

**Key Contributions:**
- First systematic comparison of 6+ negative sampling strategies (fair comparison: 1000 epochs each)
- Discovery of the "transfer learning paradox" in standard FF
- Identification of CwC-FF as the optimal transfer learning approach
- Implementation of 5 neuroscience-inspired FF variants with comprehensive code

---

## TL;DR

<p align="center">
<img src="figures/insight_summary.png" width="900">
</p>

**Core Finding**: Standard FF's label embedding design creates task-specific features that cannot transfer. Only CwC-FF achieves good transfer (89%)—by completely abandoning label embedding.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Background: The Forward-Forward Algorithm](#2-background-the-forward-forward-algorithm)
3. [Research Questions](#3-research-questions)
4. [Experiments & Results](#4-experiments--results)
5. [Bio-Inspired Extensions](#5-bio-inspired-extensions)
6. [The Root Cause](#6-the-root-cause)
7. [Critical Code](#7-critical-code)
8. [Discussion & Conclusion](#8-discussion--conclusion)
9. [Reproducibility](#9-reproducibility)

---

## 1. The Problem

Despite Geoffrey Hinton's prestige and FF's elegant biological plausibility, it has **not replaced backpropagation** after 3+ years. Why?

<p align="center">
<img src="figures/three_barriers.png" width="900">
</p>

| Barrier | Evidence | Severity |
|---------|----------|:--------:|
| **Performance Gap** | FF 94.5% vs BP 99.2% on MNIST | Critical |
| **Efficiency** | FF needs 30-240× more compute | Critical |
| **Transfer Failure** | FF 54% vs Random 72% | Critical |

### The Backpropagation Dilemma

Backpropagation has dominated neural network training for decades, yet it faces a fundamental problem: **it cannot exist in biological brains**. The algorithm requires:

1. **Symmetric weights** between forward and backward passes (Weight Transport Problem)
2. **Global error signals** propagated through the entire network
3. **Separate forward and backward phases** with stored activations

Real neurons don't have these luxuries. They operate with local information, learn continuously, and have no mechanism for "backwards" signal propagation.

### Hinton's Forward-Forward Algorithm

In 2022, Geoffrey Hinton proposed the **Forward-Forward (FF) algorithm** as a biologically plausible alternative:

> *"The Forward-Forward algorithm replaces the forward and backward passes of backpropagation with two forward passes: one with positive (real) data that the network learns to recognize, and one with negative (fake) data that the network learns to reject."*
> — Hinton, 2022

This is elegant: each layer can learn independently using only local information. No backward pass. No weight transport. Just two forward passes with different objectives.

---

## 2. Background: The Forward-Forward Algorithm

### 2.1 Core Mechanism

FF trains each layer to distinguish "positive" (real) from "negative" (fake) data using a **goodness function**:

```
Goodness(h) = mean(h²)    # Mean of squared activations
```

**Training objective per layer:**
- Push goodness of positive samples **above** threshold θ
- Push goodness of negative samples **below** threshold θ

```python
# FF layer training (simplified)
loss = log(1 + exp(-(g_pos - threshold))) + log(1 + exp(g_neg - threshold))
```

### 2.2 Label Embedding

A key design choice: **embed class labels into input pixels**:

```python
def overlay_label(x, y):
    x[:, :10] = 0                    # Clear first 10 pixels
    x[range(len(y)), y] = x.max()    # Set pixel y to max value
```

This allows the network to learn label-dependent representations without explicit output layers.

### 2.3 The Negative Sample Problem

**What makes a "negative" sample?** Hinton suggests several approaches, but which works best? This is our first research question.

---

## 3. Research Questions

<div align="center">

| RQ | Question | Why It Matters |
|----|----------|----------------|
| **RQ1** | Which negative sampling strategy works best? | Core to FF's design—no systematic comparison exists |
| **RQ2** | Can FF features transfer across tasks? | Critical for practical applications |
| **RQ3** | Why does standard FF transfer poorly? | Understanding enables improvement |
| **RQ4** | Can bio-inspired variants improve FF? | Bridges ML and neuroscience |

</div>

---

## 4. Experiments & Results

### 4.1 RQ1: Negative Sampling Strategy Comparison

**Methodology**: All 6 strategies use identical positive samples. Only negative generation differs. **1000 epochs per layer, fair comparison.**

<p align="center">
<img src="figures/negative_strategy_fair.png" width="800">
</p>

```python
# The 6 strategies we tested:
strategies = {
    "wrong_label":        # x + random wrong label (Hinton's original)
    "class_confusion":    # different image + same label
    "same_class_diff_img":# different image + wrong label
    "hybrid_mix":         # alpha * x1 + (1-alpha) * x2 + wrong label
    "noise_augmented":    # x + gaussian noise + wrong label
    "masked":             # x with random masking + wrong label
}
```

| Rank | Strategy | Test Accuracy | Note |
|:----:|----------|:-------------:|------|
| 1 | **wrong_label** | **94.50%** | Hinton's original - BEST |
| 2 | class_confusion | 92.15% | |
| 3 | same_class_diff_img | 92.06% | |
| 4 | hybrid_mix | 63.37% | Complex hurts |
| 5 | noise_augmented | 46.10% | |
| 6 | masked | 30.97% | Worst |

**Conclusion**: Simple is better. Complex negative generation destroys learning.

---

### 4.2 RQ2: Transfer Learning Results

**Protocol**: MNIST → Fashion-MNIST (freeze features, train linear classifier)

<p align="center">
<img src="figures/insight_transfer_paradox.png" width="800">
</p>

```
┌─────────────────────────────────────────────────────────────────┐
│              MNIST → Fashion-MNIST Transfer Results             │
├─────────────────┬───────────────┬───────────────┬───────────────┤
│ Method          │ Source (MNIST)│ Transfer      │ vs Random     │
├─────────────────┼───────────────┼───────────────┼───────────────┤
│ CwC-FF [Hinton] │    98.71%     │   89.05%      │   +17.2%      │
│ Backprop        │    95.08%     │   75.49%      │   +3.6%       │
│ Random Init     │      -        │   71.89%      │   baseline    │
│ Standard FF     │    89.90%     │   54.19%      │   -17.7%      │
└─────────────────┴───────────────┴───────────────┴───────────────┘
```

**The Paradox**: Standard FF pretrained features are **worse than random initialization!**

<p align="center">
<img src="figures/tsne_comparison.png" width="900">
</p>

*t-SNE visualization: FF features (left) show scattered clusters on Fashion-MNIST, while BP features (right) are better organized.*

---

### 4.3 Layer Collaboration Analysis

Based on [Hinton, 2022] Section 6.1. γ parameter controls inter-layer information flow.

<p align="center">
<img src="figures/layer_collab_heatmap.png" width="700">
</p>

| γ Value | Mode | Test Accuracy | Δ vs Baseline |
|---------|------|:-------------:|:-------------:|
| 0.0 | - | 90.38% | baseline |
| 0.1 | all | 90.74% | +0.36% |
| 0.3 | all | 90.79% | +0.41% |
| 0.5 | all | 91.14% | +0.76% |
| **0.7** | **all** | **91.56%** | **+1.18%** |
| 1.0 | all | 90.72% | +0.34% |

**Conclusion**: Moderate collaboration (γ=0.7) helps, but improvement is modest (+1.18%).

---

### 4.4 Multi-Dimensional Comparison

<p align="center">
<img src="figures/radar_comparison.png" width="600">
</p>

| Model | MNIST | Transfer | Speed | Bio-Plausibility |
|-------|:-----:|:--------:|:-----:|:----------------:|
| CwC-FF [Hinton] | 98.71% | 89.05% | Medium | High |
| Standard FF | 94.50% | 54.19% | Slow | Very High |
| Layer Collab | 91.56% | ~65% | Slow | High |
| Backpropagation | 99.2% | 75.49% | Fast | None |

---

## 5. Bio-Inspired Extensions

We implemented 5 neuroscience-inspired variants. **Results from NVIDIA A100.**

<p align="center">
<img src="figures/insight_bio_attempts.png" width="800">
</p>

### 5.1 Three-Factor Hebbian Learning

**Inspiration**: Neuromodulation (dopamine, acetylcholine, norepinephrine)

Biological learning isn't just Hebbian (pre × post). It's **three-factor**:

```
ΔW = f(pre) × f(post) × M(t)
```

Where M(t) is the neuromodulatory signal that gates when learning occurs.

| Modulation Type | MNIST | Transfer | Result |
|-----------------|:-----:|:--------:|--------|
| top_down | 91.08% | 64.32% | **+1.5% (marginal)** |
| none | 89.66% | 62.81% | baseline |
| layer_agreement | 89.58% | 59.81% | -3.0% |
| reward_prediction | 10.28% | 18.44% | **FAILED** |

<details>
<summary><b>🔍 Failure Analysis: Reward Prediction (10.28%)</b></summary>

**What happened**: Accuracy collapsed to near-random (10% = 1/10 classes).

**Root Cause**: The reward prediction error (RPE) signal was **too noisy and unstable**:

```python
# The problematic mechanism:
rpe = actual_reward - expected_reward  # 1.0 or 0.0 minus running average

# Problem 1: Binary reward (1 for positive, 0 for negative) creates
# RPE that oscillates wildly: +0.5, -0.5, +0.5, -0.5...
# This destabilizes the modulation signal.

# Problem 2: Running average (momentum=0.99) adapts too slowly,
# so RPE never converges to a meaningful signal.
```

**Why biology works but this doesn't**: Real dopamine signals are:
- Temporally precise (ms-scale)
- Tied to specific unexpected events
- Modulated by context and history

Our simplified binary RPE lacks all of these properties.

**Potential fix**: Use prediction error from a value network, not binary reward.

</details>

<details>
<summary><b>🔍 Failure Analysis: Layer Agreement (-3.0%)</b></summary>

**What happened**: Performance dropped below baseline.

**Root Cause**: Layer agreement creates **circular dependencies**:

```python
# Layer agreement: M(t) = correlation(layer_i, layer_{i+1})

# Problem: Layer i's modulation depends on layer i+1's output,
# but layer i+1's input depends on layer i's output.
# This creates a feedback loop that destabilizes training.

# Early in training:
# - Layers have random, uncorrelated outputs
# - Agreement signal is noisy
# - Noisy modulation → worse learning → less agreement → more noise
```

**Why it hurts transfer**: Layers become **overly synchronized** on source task patterns. This tight coupling makes it harder to adapt to new tasks.

**Potential fix**: Use agreement only in later training stages, or use asymmetric (top-down only) agreement.

</details>

---

### 5.2 Prospective Configuration FF

**Inspiration**: [Song et al., Nature Neuroscience 2024] - Anticipatory neural activity

Two-phase learning:
1. **Inference Phase**: Network infers what activity should be
2. **Consolidation Phase**: Weights update to produce that activity

| Model | MNIST | Transfer Gain |
|-------|:-----:|:-------------:|
| Standard FF | 93.17% | +9.1% |
| Prospective FF (100 iter) | 23.37% | **-13.2%** |

**Result**: **FAILED** — More iterations made transfer worse.

<details>
<summary><b>🔍 Failure Analysis: Prospective FF (-13.2% transfer)</b></summary>

**What happened**: MNIST accuracy was low (23%), and transfer gain was **negative** (-13.2% means transfer hurt performance).

**Root Cause 1: Target inference creates overfitting**

```python
# The problematic mechanism:
h_target = h_current + beta * feedback_proj(target_hint)

# Problem: The "target hint" comes from label information.
# With many iterations, the network learns to perfectly match
# the label-derived target, creating STRONGER label coupling
# than standard FF!

# Iterations: 1 → 10 → 100
# Label coupling: weak → medium → EXTREME
# Transfer ability: ok → poor → terrible
```

**Root Cause 2: Consolidation amplifies the problem**

```python
# Consolidation loss: minimize (h_current - h_target)²

# With many iterations:
# - h_target becomes highly label-specific
# - Consolidation forces weights to produce this exact pattern
# - Features become "memorized" for source task labels
```

**Why the original paper worked (neural circuits, not classification)**:
- Original prospective configuration was for sensory prediction
- No discrete labels involved
- Continuous prediction errors, not binary pos/neg

**Why more iterations = worse transfer**:

| Iterations | MNIST Acc | Transfer Gain | Explanation |
|:----------:|:---------:|:-------------:|-------------|
| 1 | 89.2% | +5.3% | Minimal label coupling |
| 10 | 85.1% | +1.2% | Growing coupling |
| 100 | 23.4% | **-13.2%** | Complete overfitting |

**Potential fix**: Remove label from target hints; use self-supervised targets instead.

</details>

---

### 5.3 Predictive Coding Light FF (PCL-FF)

**Inspiration**: [Nature Communications 2025] - Predictive Coding in Cortical Circuits

Only **prediction errors** (surprise) propagate forward, not full activations.

| Metric | Standard FF | PCL-FF |
|--------|:-----------:|:------:|
| MNIST | 90.0% | **17.5%** |
| Dead Neurons | 8% | **100%** |

**Result**: **COMPLETE FAILURE** — Sparsity constraint killed all neurons.

<details>
<summary><b>🔍 Failure Analysis: PCL-FF (17.5%, 100% neuron death)</b></summary>

**What happened**:
- Accuracy dropped to 17.5% (barely above random 10%)
- **100% of neurons died** (output = 0 for all inputs)

**Root Cause 1: Sparsity penalty was too aggressive**

```python
# The killer:
sparsity_penalty = h.abs().mean(dim=1) * sparsity_weight  # weight = 0.1

# Goodness = base_goodness + reconstruction_bonus - sparsity_penalty

# Problem: The sparsity penalty creates a DIRECT incentive to have
# h = 0 (zero activations). Combined with ReLU, this means:
# - Any negative pre-activation → 0
# - Sparsity penalty pushes weights to make more pre-activations negative
# - Cascade effect: more zeros → lower goodness → lower loss (!)
# - Network learns that "dead neurons" = "good loss"
```

**Root Cause 2: Surprise masking created positive feedback loop**

```python
# Surprise mask: suppress predictable, transmit surprise
surprise_mask = 1 - surprise_scale * (1 - surprise)

# Problem: When neurons start dying:
# - Dead neurons produce perfect predictions (0 predicts 0)
# - Perfect prediction → low surprise → suppress more
# - More suppression → more zeros → more death
# - Positive feedback loop accelerates collapse
```

**The cascade of death (over epochs)**:

```
Epoch 10:   ~60% neurons active, accuracy 45%
Epoch 50:   ~30% neurons active, accuracy 28%
Epoch 100:  ~10% neurons active, accuracy 19%
Epoch 200:  ~2% neurons active, accuracy 17%
Epoch 500:  0% neurons active, accuracy 17.5% (random features)
```

**Why predictive coding works in biology but not here**:
- Real neurons have **homeostatic mechanisms** preventing complete silencing
- Intrinsic excitability adjusts to maintain activity levels
- Biological sparsity is ~10-20%, not 0%

**Failed fix attempts**:
| Fix | Result |
|-----|--------|
| Lower sparsity_weight to 0.01 | Still 80% death |
| Add activity regularization | 60% death, 35% accuracy |
| Remove surprise mask | 40% death, 42% accuracy |
| Remove both | Becomes standard FF |

**Potential fix**: Add explicit **anti-death term**: `+ alive_bonus * (h > 0).float().mean()`

</details>

---

### 5.4 Summary: Bio-Inspired Attempts

<p align="center">
<img src="figures/lessons_from_failures.png" width="900">
</p>

| Approach | MNIST Δ | Transfer Δ | Verdict |
|----------|:-------:|:----------:|---------|
| Three-Factor (top-down) | +1.4% | +1.5% | Marginal |
| Layer Collab (γ=0.7) | +1.2% | — | Marginal |
| Prospective FF | — | **-13.2%** | **Failed** |
| PCL-FF | **-72.5%** | — | **Failed** |
| Reward Prediction | **-79.4%** | — | **Failed** |
| **CwC-FF [Hinton]** | +8.8% | **+17.2%** | **Works** |

**Key Insight**: Bio-inspired modifications address **symptoms**, not the **root cause**. The fundamental problem is FF's label embedding design.

---

## 6. The Root Cause

### Why Does Standard FF Fail at Transfer?

<p align="center">
<img src="figures/label_embedding_explained.png" width="1000">
</p>

**The Label Embedding Trap**:

```python
# Standard FF: Label embedded in first 10 pixels
def overlay_label(x, y, num_classes=10):
    x[:, :num_classes] = 0
    x[range(len(y)), y] = x.max()  # Label encoded here!
    return x

# Input becomes: [label_0, label_1, ..., label_9, pixel_10, ..., pixel_783]
# Features learn: f(image, SOURCE_LABEL) → task-specific!
```

**The Problem**:
- "0" in MNIST = digit zero
- "0" in Fashion-MNIST = T-shirt
- Same label slot, different meaning!
- Features are coupled to source task labels → **cannot transfer**

<p align="center">
<img src="figures/key_insight.png" width="700">
</p>

**Why CwC-FF Works**: It doesn't use label embedding at all. Instead, **competition between channels** creates **task-agnostic features** that transfer beautifully.

---

## 7. Critical Code

### 7.1 Correct FF Implementation (Critical Bugs We Found)

```python
class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, threshold=2.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = torch.optim.Adam(self.parameters(), lr=0.03)

    def forward(self, x):
        # L2 normalization (important!)
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h):
        # CRITICAL: Must use mean(), not sum()!
        # sum() causes gradient explosion with large hidden dims
        return h.pow(2).mean(dim=1)

    def train_step(self, x_pos, x_neg):
        h_pos, h_neg = self.forward(x_pos), self.forward(x_neg)
        g_pos, g_neg = self.goodness(h_pos), self.goodness(h_neg)

        loss = torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold
        ]))).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return h_pos.detach(), h_neg.detach()
```

### 7.2 Label Embedding (Root of Transfer Failure)

```python
def overlay_label(x, y, num_classes=10):
    """Embed label in first pixels - THIS BREAKS TRANSFER!"""
    x = x.clone()
    x[:, :num_classes] = 0
    # CRITICAL: Use x.max(), not 1.0!
    x[range(len(y)), y] = x.max()
    return x
```

---

### 7.3 Three-Factor Hebbian Learning

```python
class ModulationType(Enum):
    """Neuromodulatory signals (inspired by dopamine, acetylcholine)."""
    NONE = "none"              # Standard FF
    TOP_DOWN = "top_down"      # From output layer (attention-like)
    REWARD_PREDICTION = "reward_prediction"  # TD-like
    LAYER_AGREEMENT = "layer_agreement"      # Inter-layer correlation

class ThreeFactorFFLayer(nn.Module):
    def goodness_with_modulation(self, h, modulation_signal):
        """
        Three-Factor learning: goodness × (1 + scale × M(t))
        """
        local_goodness = self.goodness(h)
        modulation_effect = 1.0 + self.modulation_scale * torch.tanh(modulation_signal)
        return local_goodness * modulation_effect

    def compute_top_down_modulation(self, x, correct_label):
        """M(t) = P(correct_class) - 0.5"""
        with torch.no_grad():
            h = x
            for layer in self.layers:
                h = layer(h)
            logits = self.output_projection(h)
            probs = F.softmax(logits, dim=1)
            correct_probs = probs[range(len(correct_label)), correct_label]
            return correct_probs - 0.5  # Center around 0
```

---

### 7.4 Prospective Configuration FF

```python
class ProspectiveFFLayer(nn.Module):
    def infer_target_activity(self, h_current, target_hint, is_positive=True):
        """Phase 1: Infer what activity SHOULD be after learning."""
        adjustment = self.feedback_proj(target_hint)
        if is_positive:
            h_target = h_current + self.beta * adjustment
        else:
            h_target = h_current - self.beta * adjustment
        return F.relu(h_target)

    def consolidate(self, x, h_current, h_target):
        """Phase 2: Adjust weights to produce target activity."""
        error = h_target.detach() - h_current
        return (error ** 2).mean()

    def train_prospective_step(self, x_pos, x_neg, hint_pos, hint_neg):
        """Combine FF loss + consolidation loss."""
        h_pos, h_target_pos, consol_pos = self.forward_with_prospective(
            x_pos, hint_pos, is_positive=True)
        h_neg, h_target_neg, consol_neg = self.forward_with_prospective(
            x_neg, hint_neg, is_positive=False)

        ff_loss = self.ff_loss(self.goodness(h_pos), self.goodness(h_neg))
        return ff_loss + self.consolidation_lr * (consol_pos + consol_neg)
```

---

### 7.5 PCL-FF (Predictive Coding)

```python
class PCLFFLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.5, sparsity_weight=0.1):
        super().__init__()
        self.encoder = nn.Linear(in_features, out_features)
        self.predictor = nn.Linear(out_features, in_features)  # Reconstructs input
        self.alpha = alpha
        self.sparsity_weight = sparsity_weight

    def forward(self, x, apply_surprise_mask=True):
        """Suppress predictable, transmit surprise."""
        x_normed = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        h_raw = F.relu(self.encoder(x_normed))
        x_pred = self.predictor(h_raw)

        if apply_surprise_mask:
            pred_error = F.mse_loss(x_pred, x_normed, reduction='none').mean(dim=1)
            surprise = torch.sigmoid(pred_error * 10 - 5)
            h = h_raw * (1 - self.surprise_scale * (1 - surprise.unsqueeze(1)))
        else:
            h = h_raw
        return h, x_pred

    def goodness(self, h, x, x_pred):
        """Goodness = base + reconstruction_bonus - sparsity_penalty."""
        base_goodness = h.pow(2).mean(dim=1)
        x_normed = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        reconstruction_bonus = -F.mse_loss(x_pred, x_normed, reduction='none').mean(dim=1)
        sparsity_penalty = h.abs().mean(dim=1) * self.sparsity_weight
        return base_goodness + self.alpha * reconstruction_bonus - sparsity_penalty
```

---

### 7.6 Layer Collaboration FF

```python
class CollabFFLayer(nn.Module):
    def ff_loss(self, pos_goodness, neg_goodness, gamma_pos=None, gamma_neg=None):
        """
        Original FF:  p = sigmoid(goodness - θ)
        Collab FF:    p = sigmoid(goodness + γ - θ)
        """
        if gamma_pos is None: gamma_pos = torch.zeros_like(pos_goodness)
        if gamma_neg is None: gamma_neg = torch.zeros_like(neg_goodness)

        pos_logit = pos_goodness + gamma_pos - self.threshold
        neg_logit = neg_goodness + gamma_neg - self.threshold

        return torch.log(1 + torch.exp(torch.cat([-pos_logit, neg_logit]))).mean()

def compute_gamma(goodness_list, current_layer, scale=0.7):
    """γ = weighted sum of other layers' goodness."""
    gamma = torch.zeros_like(goodness_list[0])
    for i, g in enumerate(goodness_list):
        if i != current_layer:
            gamma = gamma + scale * g.detach()
    return gamma
```

---

### 7.7 CwC-FF [Hinton §6.3] (The Solution)

```python
class CwCLossCE(nn.Module):
    """Cross-entropy over channel-wise goodness — NO negative samples!"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, goodness_matrix, targets):
        return self.criterion(goodness_matrix, targets)

class CFSEBlock(nn.Module):
    """Grouped convolutions: channels specialize for classes."""
    def __init__(self, in_channels, out_channels, num_classes, is_cfse=True):
        super().__init__()
        groups = num_classes if is_cfse else 1
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, 2) if is_cfse else nn.Identity()

    def forward(self, x):
        return self.bn(F.relu(self.maxpool(self.conv(x))))

class CwCFFLayer(nn.Module):
    def compute_goodness_channelwise(self, y, targets):
        """One goodness per class channel."""
        channels_per_class = self.out_channels // self.num_classes
        y_splits = torch.split(y, channels_per_class, dim=1)
        goodness_factors = [y_split.pow(2).mean(dim=(1,2,3)).unsqueeze(-1)
                          for y_split in y_splits]
        return torch.cat(goodness_factors, dim=1)  # [B, num_classes]
```

**Result**: 98.71% MNIST, **89.05% transfer** (+17.2% vs random). **The only approach that works.**

---

## 8. Discussion & Conclusion

### 8.1 The Transfer Learning Paradox

Our most striking finding: **pretrained features hurt transfer performance**. This contradicts the standard deep learning assumption that "more pretraining = better transfer."

| Model | Transfer | Explanation |
|-------|----------|-------------|
| Standard FF | 54% (worst) | Label embedding encodes source labels into features |
| Backpropagation | 75% | Cross-entropy creates task-specific boundaries |
| Random Init | 72% | "Blank slate"—no prior biases to unlearn |
| **CwC-FF** | **89% (best)** | Competition creates universal features |

### 8.2 What Would Make FF Viable?

| Solution | Evidence | Status |
|----------|----------|:------:|
| **Remove label embedding** | CwC-FF: 89% transfer | ✅ Proven |
| Add layer collaboration | +1.2% | Marginal |
| Add neuromodulation | +1.5% | Marginal |
| Better architectures | CNNs help | Partial |
| Efficient training | — | ❌ Unsolved |

### 8.3 Final Verdict

FF is a beautiful idea—local learning without backpropagation. But after 3 years:

1. **Performance gap remains**: 94.5% vs 99.2% on MNIST
2. **Transfer is broken**: 54% vs 72% (worse than random!)
3. **Efficiency is poor**: 30-240× slower
4. **Scalability unsolved**: No competitive ImageNet results

**The path forward**: CwC-FF shows that removing label embedding is key. Future work should focus on label-free contrastive objectives, efficient training, and scaling to modern architectures.

---

## 9. Reproducibility

### Hardware

- **Local**: M1 Max MacBook Pro (MPS)
- **Server**: NVIDIA A100 40GB (CUDA)

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs per layer | 1000 | Critical for convergence |
| Batch size | 50000 (full batch) | Improves stability |
| Optimizer | Adam | lr=0.03 |
| Threshold θ | 2.0 | Standard choice |

<p align="center">
<img src="figures/training_dynamics.png" width="800">
</p>

### Commands

```bash
# Setup
git clone https://github.com/koriyoshi2041/ff-negative-samples.git
cd ff-negative-samples
pip install torch torchvision numpy tqdm matplotlib scikit-learn

# Negative strategy comparison (fair, 1000 epochs)
python experiments/fair_strategy_comparison.py

# Transfer learning
python experiments/transfer_comparison.py

# Bio-inspired variants (A100)
python experiments/three_factor_experiment.py --modulation top_down
python experiments/pcl_ff_experiment.py --epochs 500
python experiments/prospective_ff_experiment.py --mode full
python experiments/layer_collab_transfer.py --pretrain-epochs 500
```

### Project Structure

```
ff-research/
├── models/
│   ├── ff_correct.py           # Standard FF (94.5%)
│   ├── cwc_ff.py               # CwC-FF (98.71%, best transfer)
│   ├── layer_collab_ff.py      # Layer Collaboration
│   ├── three_factor_ff.py      # Bio: Neuromodulation
│   ├── prospective_ff.py       # Bio: Anticipatory
│   ├── pcl_ff.py               # Bio: Predictive Coding
│   └── dendritic_ff.py         # Bio: Compartments
├── experiments/                # All experiment scripts
├── figures/                    # All visualizations
└── results/                    # JSON result files
```

---

## PFF Generations

<p align="center">
<img src="figures/pff_samples.png" width="500">
</p>

*PFF can run "backwards" to generate samples. Top: random generations. Bottom: class-conditioned (3s and 5s).*

---

## References

```bibtex
@article{hinton2022forward,
  title={The Forward-Forward Algorithm: Some Preliminary Investigations},
  author={Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2212.13345},
  year={2022}
}

@article{cwcff2024,
  title={Convolutional Channel-wise Competitive Learning for the Forward-Forward Algorithm},
  author={Papachristodoulou et al.},
  journal={AAAI},
  year={2024}
}
```

---

<div align="center">

**Repository**: [github.com/koriyoshi2041/ff-negative-samples](https://github.com/koriyoshi2041/ff-negative-samples)

</div>
