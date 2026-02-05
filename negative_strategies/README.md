# Negative Sample Strategies for Forward-Forward Algorithm

This module provides a collection of negative sample generation strategies for the Forward-Forward (FF) algorithm, with a unified interface for easy experimentation.

## Overview

The Forward-Forward algorithm trains neural networks using two forward passes instead of backpropagation:
1. **Positive pass**: Real data with correct labels → maximize "goodness"
2. **Negative pass**: Fake/corrupted data → minimize "goodness"

The **quality of negative samples** critically affects FF performance. This module implements 10 different strategies from the literature and research.

## Quick Start

```python
from negative_strategies import create_strategy, list_strategies

# List all available strategies
list_strategies()

# Create a strategy
strategy = create_strategy('label_embedding', num_classes=10)

# Generate negative samples
negatives = strategy.generate(images, labels)

# Some strategies also customize positive samples
positives = strategy.create_positive(images, labels)
```

## Strategies

### 1. Label Embedding (`label_embedding`)
**Hinton's original method** from the FF paper.

- Embeds a one-hot label into the first N pixels of the image
- Positive: correct label embedded
- Negative: random wrong label embedded

```python
strategy = create_strategy('label_embedding', num_classes=10, label_scale=1.0)
```

### 2. Image Mixing (`image_mixing`)
**Mix two different images** pixel-wise.

- `interpolate`: `neg = α * img1 + (1-α) * img2`
- `mask`: Random binary mask selecting pixels from each image

```python
strategy = create_strategy('image_mixing', 
    alpha_range=(0.3, 0.7),
    mixing_mode='interpolate'  # or 'mask'
)
```

Variant: `class_aware_mixing` ensures mixed images come from different classes.

### 3. Random Noise (`random_noise`)
**Simplest baseline**: pure random noise.

- Types: `gaussian`, `uniform`, `matched` (matches input statistics)
- Provides minimal learning signal

```python
strategy = create_strategy('random_noise', 
    noise_type='gaussian',
    noise_mean=0.0,
    noise_std=1.0
)
```

### 4. Class Confusion (`class_confusion`)
**Correct image + wrong label** (label-only perturbation).

- Modes: `random`, `adjacent`, `hardest` (requires confusion matrix)
- Doesn't corrupt visual features

```python
strategy = create_strategy('class_confusion',
    confusion_mode='random'
)
```

Variant: `class_confusion_embedded` combines with label embedding.

### 5. Self-Contrastive (`self_contrastive`)
**SCFF method**: strong augmentation as negatives.

Based on "Self-Contrastive Forward-Forward Algorithm" (arXiv:2409.12184).

- Applies multiple augmentations: noise, cutout, shuffle, jitter
- Self-supervised: no labels needed

```python
strategy = create_strategy('self_contrastive',
    strong_aug_prob=0.8,
    noise_std=0.3,
    cutout_ratio=0.3
)
```

### 6. Masking (`masking`)
**Random masking** of image portions.

- Mask modes: `zero`, `noise`, `shuffle`, `mean`
- Mask types: `random`, `block`, `structured`

```python
strategy = create_strategy('masking',
    mask_ratio=0.5,
    mask_mode='zero',
    mask_type='block',
    block_size=4
)
```

### 7. Layer-Wise (`layer_wise`)
**Layer-specific generation** using activations.

Based on "Layer Collaboration in the Forward-Forward Algorithm" (arXiv:2305.12393).

- Generates negatives adapted to each layer
- Can use learned generators or adaptive perturbation

```python
strategy = create_strategy('layer_wise',
    layer_dims=[784, 500, 500],
    perturbation_scale=0.5
)

# Use for specific layer
strategy.set_layer(layer_idx=1)
negatives = strategy.generate_from_activations(activations, layer_idx=1)
```

### 8. Adversarial (`adversarial`)
**Gradient-based perturbation** (FGSM/PGD style).

- Creates maximally hard negatives
- Requires model reference for gradient computation

```python
strategy = create_strategy('adversarial',
    epsilon=0.1,
    alpha=0.01,
    num_steps=3,  # PGD steps
    perturbation_norm='linf'  # or 'l2'
)

# Set model for gradient computation
strategy.set_model(model)
negatives = strategy.generate(images, labels)
```

Variant: `fast_adversarial` for single-step FGSM.

### 9. Hard Mining (`hard_mining`)
**Select hardest negatives** from a pool.

- Mining modes: `goodness`, `distance`, `class`, `feature`
- Can use memory bank for harder negatives over time

```python
strategy = create_strategy('hard_mining',
    mining_mode='goodness',
    pool_size=128,
    top_k_ratio=0.5,
    use_memory_bank=True
)

strategy.set_model(model)
negatives = strategy.generate(images, labels)
```

### 10. Mono-Forward (`mono_forward`)
**No negative samples** variant.

Based on "Mono-Forward: Backpropagation-Free Algorithm" (arXiv:2501.08756).

- Uses alternative loss functions (MSE, hinge, softplus to target goodness)
- Requires modified training loop

```python
strategy = create_strategy('mono_forward',
    target_goodness=2.0,
    loss_type='mse'
)

# Use mono_loss instead of FF contrastive loss
loss = strategy.compute_mono_loss(activations, labels)
```

Variant: `energy_minimization` for energy-based training.

## Unified Interface

All strategies implement:

```python
class NegativeStrategy:
    def generate(self, images, labels, **kwargs) -> torch.Tensor:
        """Generate negative samples."""
        pass
    
    def create_positive(self, images, labels, **kwargs) -> torch.Tensor:
        """Create positive samples (default: flatten)."""
        pass
    
    @property
    def requires_labels(self) -> bool:
        """Whether labels are needed."""
        pass
    
    def get_config(self) -> dict:
        """Return configuration for logging."""
        pass
```

## Registry

Strategies are auto-registered and can be accessed by name:

```python
from negative_strategies import StrategyRegistry

# List all
print(StrategyRegistry.list_strategies())

# Create by name
strategy = StrategyRegistry.create('image_mixing', num_classes=10)

# Get class
cls = StrategyRegistry.get('masking')
```

## Testing

Run the test suite:

```bash
# Test all strategies
python test_strategies.py

# Test specific strategy
python test_strategies.py --strategy label_embedding

# List available strategies
python test_strategies.py --list
```

## Comparison Table

| Strategy | Labels | Complexity | Best For |
|----------|--------|------------|----------|
| label_embedding | ✓ | Low | Supervised, simple tasks |
| image_mixing | ✗ | Low | Unsupervised, any task |
| random_noise | ✗ | Lowest | Baseline comparison |
| class_confusion | ✓ | Low | Label-aware learning |
| self_contrastive | ✗ | Medium | Self-supervised |
| masking | ✗ | Low | Feature learning |
| layer_wise | ✗ | High | Deep networks |
| adversarial | ✗ | High | Robust features |
| hard_mining | ✓ | Medium | Difficult tasks |
| mono_forward | ✓ | Low | Energy efficiency |

## Integration with FF Baseline

```python
from negative_strategies import create_strategy
from experiments.ff_baseline import FFNetwork, train_ff_epoch

# Create strategy
strategy = create_strategy('image_mixing')

# Use in training
def train_step(model, images, labels, strategy):
    pos_data = strategy.create_positive(images, labels)
    neg_data = strategy.generate(images, labels)
    return model.train_step(pos_data, neg_data)
```

## References

1. Hinton (2022). The Forward-Forward Algorithm. arXiv:2212.13345
2. Chen et al. (2024). Self-Contrastive Forward-Forward. arXiv:2409.12184
3. Gat et al. (2023). Layer Collaboration in FF. arXiv:2305.12393
4. Gong et al. (2025). Mono-Forward. arXiv:2501.08756

## File Structure

```
negative_strategies/
├── __init__.py          # Exports and registry
├── base.py              # Base class and registry
├── label_embedding.py   # Hinton's original
├── image_mixing.py      # Pixel mixing
├── random_noise.py      # Noise baseline
├── class_confusion.py   # Wrong labels
├── self_contrastive.py  # SCFF augmentation
├── masking.py           # Random masking
├── layer_wise.py        # Layer-specific
├── adversarial.py       # Gradient-based
├── hard_mining.py       # Hard negatives
├── mono_forward.py      # No negatives
├── test_strategies.py   # Test suite
└── README.md            # This file
```
