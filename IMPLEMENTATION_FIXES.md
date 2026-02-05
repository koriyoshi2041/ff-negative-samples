# Forward-Forward Implementation Fixes

## Summary

The original implementation achieved only **56% accuracy** on MNIST. After fixes, we now achieve **~93% accuracy** - a massive improvement and within range of the reported ~98.6% from Hinton's paper.

## Critical Issues Found

### 1. Goodness Calculation: MEAN vs SUM ⭐ CRITICAL

**Bug:** `goodness = (x ** 2).sum(dim=1)`  
**Fix:** `goodness = x.pow(2).mean(dim=1)`

This is the **most critical fix**. Using SUM makes the threshold scale-dependent on layer width. With 500 neurons, the sum of squared activations is ~500x larger than the mean, making the fixed threshold=2.0 completely inappropriate.

**Reference (mpezeshki):**
```python
g_pos = self.forward(x_pos).pow(2).mean(1)  # MEAN!
```

### 2. Training Strategy: Layer-by-Layer vs Mini-Batch ⭐ CRITICAL

**Bug:** Training all layers simultaneously with mini-batches
```python
for epoch in range(num_epochs):
    for batch in train_loader:
        for layer in layers:
            layer.train_step(batch)  # WRONG!
```

**Fix:** Greedy layer-by-layer training
```python
for layer in layers:
    for epoch in range(1000):  # Train this layer to convergence
        layer.train_step(all_data)
    # Then freeze this layer and move to next
```

This is how FF is designed: each layer learns to be a good feature extractor independently, then passes its output to the next layer.

### 3. Label Embedding Value

**Bug:** Using 1.0 for one-hot encoding
```python
one_hot.scatter_(1, labels.unsqueeze(1), 1.0)  # Value = 1.0
```

**Fix:** Using `x.max()` for one-hot encoding
```python
x_[range(x.shape[0]), y] = x.max()  # Value = x.max() ≈ 2.82
```

The label signal should be comparable in magnitude to the normalized image data. After normalization, MNIST pixels range from ~-0.42 to ~2.82.

### 4. Batch Size

**Bug:** Small mini-batches (64)  
**Fix:** Full-batch training (50,000)

The original mpezeshki implementation loads the entire MNIST training set in one batch. While mini-batch can work, it requires different hyperparameters.

### 5. Negative Sample Generation

**Bug:** Random wrong labels
```python
wrong_labels = torch.randint(0, 10, (batch_size,))
```

**Fix:** Shuffle labels across batch
```python
rnd = torch.randperm(x.size(0))
x_neg = overlay_y_on_x(x, y[rnd])  # Same images, shuffled labels
```

This ensures each image has a plausible but incorrect label pairing.

## Results

| Implementation | MNIST Test Accuracy |
|----------------|---------------------|
| Original (broken) | 56% |
| Fixed | 93% |
| Reference (Hinton paper) | ~98.6% |
| Reference (mpezeshki repo) | ~98.64% |

## Remaining Gap

The 5% gap between our 93% and the reference 98.6% could be due to:

1. **Device differences**: MPS (Apple Silicon) vs CUDA may have numerical precision differences
2. **Random seed**: Different initialization could affect convergence
3. **More training**: The reference uses 1000 epochs per layer; we used 1000 as well but convergence may vary
4. **Architecture**: Could try deeper networks (4 layers) or wider layers (2000 neurons)

## Files Changed

- `models/layer_collab_ff.py` - Fixed layer collaboration implementation
- `experiments/ff_baseline.py` - Fixed baseline FF implementation
- `models/ff_correct.py` - Clean reference implementation
- `models/ff_debug.py` - Debug/verification script

## How to Verify

```bash
cd ~/Desktop/Rios/ff-research
source venv/bin/activate

# Run the corrected baseline (takes ~40 min)
python models/ff_debug.py

# Expected output:
# Train accuracy: ~93%
# Test accuracy: ~93%
```

## References

- Hinton, G. (2022). "The Forward-Forward Algorithm: Some Preliminary Investigations"
- mpezeshki/pytorch_forward_forward: https://github.com/mpezeshki/pytorch_forward_forward
- Lorberbom et al. (2024). "Layer Collaboration in the Forward-Forward Algorithm"
