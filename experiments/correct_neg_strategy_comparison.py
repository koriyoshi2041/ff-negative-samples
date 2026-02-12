"""
Correct Negative Sample Strategy Comparison Experiment

Uses the CORRECT Forward-Forward implementation:
1. Goodness = MEAN of squared activations (not sum)
2. Layer-by-layer greedy training
3. Label embedding uses x.max() (not 1.0)
4. Full-batch training (batch_size = 50000)

Tested Strategies:
1. label_embedding - Hinton's original (shuffled wrong labels)
2. class_confusion - Correct image + wrong label embedded
3. random_noise - Matched statistics noise baseline
4. image_mixing - Pixel-wise mixing of two images
5. masking - Random pixel masking

For non-label strategies (random_noise, image_mixing, masking),
we use Linear Probe to assess learned representations.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple
import json
import time
from pathlib import Path
import sys

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Negative Sample Strategies
# ============================================================

def overlay_label(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Label Embedding Strategy (Hinton's original).

    Replace the first 10 pixels with one-hot-encoded label.
    CRITICAL: Use x.max() as the label value, not 1.0!
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


def class_confusion_strategy(x: torch.Tensor, y: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """
    Class Confusion Strategy.

    Same image with a randomly selected wrong label embedded.
    Different from label_embedding which uses shuffled labels.
    """
    batch_size = x.size(0)
    device = x.device

    # Generate random wrong labels (not the correct one)
    wrong_labels = torch.randint(0, num_classes, (batch_size,), device=device)
    mask = wrong_labels == y
    wrong_labels[mask] = (wrong_labels[mask] + 1) % num_classes

    return overlay_label(x, wrong_labels)


def random_noise_strategy(x: torch.Tensor) -> torch.Tensor:
    """
    Random Noise Strategy.

    Generate noise with matched statistics (mean and std) of input data.
    """
    mean = x.mean()
    std = x.std() + 1e-8
    noise = torch.randn_like(x) * std + mean
    return noise


def image_mixing_strategy(x: torch.Tensor, alpha_range: Tuple[float, float] = (0.3, 0.7)) -> torch.Tensor:
    """
    Image Mixing Strategy (Hinton's unsupervised variant).

    Mix two different images: neg = alpha * img1 + (1-alpha) * img2
    Creates chimera images that don't belong to any real class.
    """
    batch_size = x.size(0)

    # Shuffle to get different images
    perm = torch.randperm(batch_size, device=x.device)
    other_images = x[perm]

    # Random mixing coefficients
    alpha = torch.rand(batch_size, 1, device=x.device)
    alpha = alpha * (alpha_range[1] - alpha_range[0]) + alpha_range[0]

    return alpha * x + (1 - alpha) * other_images


def masking_strategy(x: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
    """
    Masking Strategy.

    Randomly mask (zero out) a portion of pixels.
    Creates corrupted versions of real images as negatives.
    """
    mask = (torch.rand_like(x) < mask_ratio).float()
    return x * (1 - mask)


# ============================================================
# FF Layer and Network (Correct Implementation)
# ============================================================

class FFLayer(nn.Module):
    """
    Forward-Forward Layer - Correct Implementation.

    Key points:
    - Goodness = MEAN of squared activations (not sum!)
    - Threshold default = 2.0
    - Layer normalization before linear
    """

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with layer normalization."""
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute goodness - MEAN of squared activations.

        CRITICAL: This must be MEAN, not SUM!
        """
        return h.pow(2).mean(dim=1)

    def train_layer(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                    num_epochs: int = 200, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train this layer to convergence."""
        iterator = tqdm(range(num_epochs), desc="Training layer") if verbose else range(num_epochs)

        for epoch in iterator:
            h_pos = self.forward(x_pos)
            h_neg = self.forward(x_neg)

            g_pos = self.goodness(h_pos)
            g_neg = self.goodness(h_neg)

            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold
            ]))).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Show progress
            if verbose and hasattr(iterator, 'set_postfix'):
                pos_acc = (g_pos > self.threshold).float().mean().item()
                neg_acc = (g_neg < self.threshold).float().mean().item()
                iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pos_acc': f'{pos_acc:.2%}',
                    'neg_acc': f'{neg_acc:.2%}'
                })

        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFNetwork(nn.Module):
    """Forward-Forward Network - Correct Implementation."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dims = dims
        for d in range(len(dims) - 1):
            self.layers.append(FFLayer(dims[d], dims[d + 1], threshold, lr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_all_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get activations from all layers."""
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations

    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 200, verbose: bool = True):
        """Greedy layer-by-layer training."""
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\nTraining layer {i}...')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg, epochs_per_layer, verbose)

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels and picking highest goodness."""
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(num_classes):
            h = overlay_label(x, torch.full((batch_size,), label, device=x.device))

            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))

            total_goodness = sum(goodness)
            goodness_per_label.append(total_goodness.unsqueeze(1))

        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor,
                     num_classes: int = 10) -> float:
        """Compute accuracy."""
        predictions = self.predict(x, num_classes)
        return (predictions == y).float().mean().item()


# ============================================================
# Linear Probe for unsupervised strategies
# ============================================================

class LinearProbe(nn.Module):
    """Simple linear probe for representation quality."""

    def __init__(self, input_dim: int, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_probe(
    model: FFNetwork,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    Train a linear probe on the concatenated activations from all layers.

    Returns (train_acc, test_acc)
    """
    device = x_train.device

    # Get representations (concatenate all layer activations)
    model.train(False)
    with torch.no_grad():
        train_acts = model.get_all_activations(x_train)
        test_acts = model.get_all_activations(x_test)

        # Concatenate activations from all layers
        train_repr = torch.cat(train_acts, dim=1)
        test_repr = torch.cat(test_acts, dim=1)

    # Train linear probe
    probe = LinearProbe(train_repr.size(1), num_classes=10).to(device)
    optimizer = Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    iterator = tqdm(range(epochs), desc="Training probe") if verbose else range(epochs)

    for _ in iterator:
        probe.train()
        optimizer.zero_grad()
        logits = probe(train_repr)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # Final assessment
    probe.train(False)
    with torch.no_grad():
        train_logits = probe(train_repr)
        train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()

        test_logits = probe(test_repr)
        test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

    return train_acc, test_acc


# ============================================================
# Data Loading
# ============================================================

def get_mnist_data(batch_size: int = 50000):
    """Get MNIST data with full batch."""
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])

    train_loader = DataLoader(
        MNIST('./data/', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(
        MNIST('./data/', train=False, download=True, transform=transform),
        batch_size=10000, shuffle=False
    )

    return train_loader, test_loader


# ============================================================
# Training Functions for Each Strategy
# ============================================================

def train_label_embedding(
    x: torch.Tensor, y: torch.Tensor,
    device: torch.device,
    epochs_per_layer: int = 200,
    verbose: bool = True
) -> Tuple[FFNetwork, Dict]:
    """Train with label embedding strategy (Hinton's original)."""
    print("\n" + "="*60)
    print("Strategy: LABEL EMBEDDING (Hinton's original)")
    print("="*60)

    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Positive: correct label embedded
    x_pos = overlay_label(x, y)

    # Negative: shuffled labels (wrong labels from same batch)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_label(x, y[rnd])

    start_time = time.time()
    model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    train_time = time.time() - start_time

    train_acc = model.get_accuracy(x, y)

    return model, {
        'train_acc': train_acc,
        'train_time': train_time,
        'uses_labels': True
    }


def train_class_confusion(
    x: torch.Tensor, y: torch.Tensor,
    device: torch.device,
    epochs_per_layer: int = 200,
    verbose: bool = True
) -> Tuple[FFNetwork, Dict]:
    """Train with class confusion strategy (random wrong labels)."""
    print("\n" + "="*60)
    print("Strategy: CLASS CONFUSION (random wrong label)")
    print("="*60)

    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Positive: correct label embedded
    x_pos = overlay_label(x, y)

    # Negative: same image with random wrong label embedded
    x_neg = class_confusion_strategy(x, y, num_classes=10)

    start_time = time.time()
    model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    train_time = time.time() - start_time

    train_acc = model.get_accuracy(x, y)

    return model, {
        'train_acc': train_acc,
        'train_time': train_time,
        'uses_labels': True
    }


def train_random_noise(
    x: torch.Tensor, y: torch.Tensor,
    device: torch.device,
    epochs_per_layer: int = 200,
    verbose: bool = True
) -> Tuple[FFNetwork, Dict]:
    """Train with random noise strategy."""
    print("\n" + "="*60)
    print("Strategy: RANDOM NOISE (matched statistics)")
    print("="*60)

    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Positive: correct label embedded (for fair comparison)
    x_pos = overlay_label(x, y)

    # Negative: random noise with matched statistics
    x_neg = random_noise_strategy(x)

    start_time = time.time()
    model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    train_time = time.time() - start_time

    # Note: For noise strategy, we cannot use goodness-based prediction
    # because negatives don't have label structure
    # We'll use linear probe accuracy instead
    train_acc = model.get_accuracy(x, y)  # This may be low/random

    return model, {
        'train_acc': train_acc,
        'train_time': train_time,
        'uses_labels': False,
        'needs_linear_probe': True
    }


def train_image_mixing(
    x: torch.Tensor, y: torch.Tensor,
    device: torch.device,
    epochs_per_layer: int = 200,
    verbose: bool = True
) -> Tuple[FFNetwork, Dict]:
    """Train with image mixing strategy."""
    print("\n" + "="*60)
    print("Strategy: IMAGE MIXING (chimera images)")
    print("="*60)

    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Positive: correct label embedded (for fair comparison)
    x_pos = overlay_label(x, y)

    # Negative: mixed images (no labels - chimera images)
    x_neg = image_mixing_strategy(x)

    start_time = time.time()
    model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    train_time = time.time() - start_time

    train_acc = model.get_accuracy(x, y)

    return model, {
        'train_acc': train_acc,
        'train_time': train_time,
        'uses_labels': False,
        'needs_linear_probe': True
    }


def train_masking(
    x: torch.Tensor, y: torch.Tensor,
    device: torch.device,
    epochs_per_layer: int = 200,
    mask_ratio: float = 0.5,
    verbose: bool = True
) -> Tuple[FFNetwork, Dict]:
    """Train with masking strategy."""
    print("\n" + "="*60)
    print(f"Strategy: MASKING (ratio={mask_ratio})")
    print("="*60)

    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Positive: correct label embedded
    x_pos = overlay_label(x, y)

    # Negative: masked images with correct label
    x_masked = masking_strategy(x, mask_ratio)
    x_neg = overlay_label(x_masked, y)

    start_time = time.time()
    model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    train_time = time.time() - start_time

    train_acc = model.get_accuracy(x, y)

    return model, {
        'train_acc': train_acc,
        'train_time': train_time,
        'uses_labels': True,
        'mask_ratio': mask_ratio
    }


# ============================================================
# Main Experiment
# ============================================================

def run_comparison(epochs_per_layer: int = 200, seed: int = 1234):
    """Run strategy comparison experiment."""
    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # Load data (full batch)
    train_loader, test_loader = get_mnist_data(batch_size=50000)

    x_train, y_train = next(iter(train_loader))
    x_train, y_train = x_train.to(device), y_train.to(device)

    x_test, y_test = next(iter(test_loader))
    x_test, y_test = x_test.to(device), y_test.to(device)

    print("\n" + "="*60)
    print("CORRECT NEGATIVE STRATEGY COMPARISON EXPERIMENT")
    print("="*60)
    print(f"Architecture: [784, 500, 500]")
    print(f"Epochs per layer: {epochs_per_layer}")
    print(f"Batch size: {x_train.size(0)} (full batch)")
    print(f"Threshold: 2.0")
    print(f"Learning rate: 0.03")
    print(f"Goodness: MEAN of squared activations")
    print(f"Label embedding: x.max()")

    results = {}

    # ============================================================
    # Strategy 1: Label Embedding (Hinton's original)
    # ============================================================
    model_le, stats_le = train_label_embedding(
        x_train, y_train, device, epochs_per_layer
    )
    test_acc_le = model_le.get_accuracy(x_test, y_test)
    results['label_embedding'] = {
        'train_acc': stats_le['train_acc'],
        'test_acc': test_acc_le,
        'train_time': stats_le['train_time'],
        'uses_labels': True,
        'description': "Hinton's original - shuffled wrong labels"
    }
    print(f"\nLabel Embedding - Train: {stats_le['train_acc']*100:.2f}%, Test: {test_acc_le*100:.2f}%")

    # ============================================================
    # Strategy 2: Class Confusion (random wrong label)
    # ============================================================
    model_cc, stats_cc = train_class_confusion(
        x_train, y_train, device, epochs_per_layer
    )
    test_acc_cc = model_cc.get_accuracy(x_test, y_test)
    results['class_confusion'] = {
        'train_acc': stats_cc['train_acc'],
        'test_acc': test_acc_cc,
        'train_time': stats_cc['train_time'],
        'uses_labels': True,
        'description': "Random wrong label for each sample"
    }
    print(f"\nClass Confusion - Train: {stats_cc['train_acc']*100:.2f}%, Test: {test_acc_cc*100:.2f}%")

    # ============================================================
    # Strategy 3: Random Noise + Linear Probe
    # ============================================================
    model_rn, stats_rn = train_random_noise(
        x_train, y_train, device, epochs_per_layer
    )
    test_acc_rn = model_rn.get_accuracy(x_test, y_test)

    # Linear probe assessment
    print("\nTraining Linear Probe for Random Noise strategy...")
    probe_train_rn, probe_test_rn = train_linear_probe(
        model_rn, x_train, y_train, x_test, y_test, epochs=100, verbose=True
    )

    results['random_noise'] = {
        'train_acc': stats_rn['train_acc'],
        'test_acc': test_acc_rn,
        'train_time': stats_rn['train_time'],
        'uses_labels': False,
        'linear_probe_train_acc': probe_train_rn,
        'linear_probe_test_acc': probe_test_rn,
        'description': "Matched statistics noise baseline"
    }
    print(f"\nRandom Noise - Train: {stats_rn['train_acc']*100:.2f}%, Test: {test_acc_rn*100:.2f}%")
    print(f"  Linear Probe - Train: {probe_train_rn*100:.2f}%, Test: {probe_test_rn*100:.2f}%")

    # ============================================================
    # Strategy 4: Image Mixing + Linear Probe
    # ============================================================
    model_im, stats_im = train_image_mixing(
        x_train, y_train, device, epochs_per_layer
    )
    test_acc_im = model_im.get_accuracy(x_test, y_test)

    # Linear probe assessment
    print("\nTraining Linear Probe for Image Mixing strategy...")
    probe_train_im, probe_test_im = train_linear_probe(
        model_im, x_train, y_train, x_test, y_test, epochs=100, verbose=True
    )

    results['image_mixing'] = {
        'train_acc': stats_im['train_acc'],
        'test_acc': test_acc_im,
        'train_time': stats_im['train_time'],
        'uses_labels': False,
        'linear_probe_train_acc': probe_train_im,
        'linear_probe_test_acc': probe_test_im,
        'description': "Mix two images pixel-wise (chimera)"
    }
    print(f"\nImage Mixing - Train: {stats_im['train_acc']*100:.2f}%, Test: {test_acc_im*100:.2f}%")
    print(f"  Linear Probe - Train: {probe_train_im*100:.2f}%, Test: {probe_test_im*100:.2f}%")

    # ============================================================
    # Strategy 5: Masking
    # ============================================================
    model_mk, stats_mk = train_masking(
        x_train, y_train, device, epochs_per_layer, mask_ratio=0.5
    )
    test_acc_mk = model_mk.get_accuracy(x_test, y_test)

    # Also add linear probe for masking to compare representation quality
    print("\nTraining Linear Probe for Masking strategy...")
    probe_train_mk, probe_test_mk = train_linear_probe(
        model_mk, x_train, y_train, x_test, y_test, epochs=100, verbose=True
    )

    results['masking'] = {
        'train_acc': stats_mk['train_acc'],
        'test_acc': test_acc_mk,
        'train_time': stats_mk['train_time'],
        'uses_labels': True,
        'mask_ratio': 0.5,
        'linear_probe_train_acc': probe_train_mk,
        'linear_probe_test_acc': probe_test_mk,
        'description': "Random pixel masking (50 percent)"
    }
    print(f"\nMasking - Train: {stats_mk['train_acc']*100:.2f}%, Test: {test_acc_mk*100:.2f}%")
    print(f"  Linear Probe - Train: {probe_train_mk*100:.2f}%, Test: {probe_test_mk*100:.2f}%")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Strategy':<20} {'Train Acc':<12} {'Test Acc':<12} {'Probe Test':<12} {'Time (s)':<10}")
    print("-"*80)

    for name, data in results.items():
        if name == 'metadata':
            continue
        probe_acc = data.get('linear_probe_test_acc', '-')
        if isinstance(probe_acc, float):
            probe_str = f"{probe_acc*100:.2f}%"
        else:
            probe_str = probe_acc
        print(f"{name:<20} {data['train_acc']*100:.2f}%{'':<5} {data['test_acc']*100:.2f}%{'':<5} {probe_str:<12} {data['train_time']:.1f}")

    # Add metadata
    results['metadata'] = {
        'epochs_per_layer': epochs_per_layer,
        'batch_size': 50000,
        'architecture': [784, 500, 500],
        'threshold': 2.0,
        'learning_rate': 0.03,
        'goodness': 'mean',
        'label_scale': 'x.max()',
        'training_method': 'layer_by_layer_greedy',
        'seed': seed
    }

    return results


def save_results(results: Dict, output_path: str):
    """Save results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Correct Negative Strategy Comparison')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Epochs per layer (default 200 for about 85 percent accuracy)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()

    results = run_comparison(epochs_per_layer=args.epochs, seed=args.seed)

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'correct_neg_strategy_comparison.json'
    save_results(results, str(output_path))

    return results


if __name__ == "__main__":
    main()
