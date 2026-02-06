"""
Correct Strategy Comparison Experiment

Uses the correct Forward-Forward implementation from models/ff_correct.py:
1. Goodness = MEAN of squared activations (not sum)
2. Layer-by-layer greedy training
3. Label embedding uses x.max() (not fixed scale)
4. Full-batch training (batch_size = 50000)

Compares 3 strategies:
- label_embedding: Hinton's original method
- random_noise: Matched noise baseline
- masking: Random pixel masking
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


def random_noise_strategy(x: torch.Tensor) -> torch.Tensor:
    """
    Random Noise Strategy.

    Generate noise with matched statistics (mean and std) of input data.
    """
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-8
    noise = torch.randn_like(x) * std + mean
    return noise


def masking_strategy(x: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
    """
    Masking Strategy.

    Randomly mask (zero out) a portion of pixels.
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
                    num_epochs: int = 100, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train this layer to convergence."""
        iterator = tqdm(range(num_epochs), desc="Training layer") if verbose else range(num_epochs)

        for _ in iterator:
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

        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFNetwork(nn.Module):
    """Forward-Forward Network - Correct Implementation."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(FFLayer(dims[d], dims[d + 1], threshold, lr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 100, verbose: bool = True):
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

def train_label_embedding(x: torch.Tensor, y: torch.Tensor,
                          device: torch.device,
                          epochs_per_layer: int = 100,
                          verbose: bool = True) -> Tuple[FFNetwork, Dict]:
    """Train with label embedding strategy (Hinton's original)."""
    print("\n" + "="*60)
    print("Strategy: LABEL EMBEDDING")
    print("="*60)

    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Positive: correct label embedded
    x_pos = overlay_label(x, y)

    # Negative: shuffled labels (wrong labels)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_label(x, y[rnd])

    start_time = time.time()
    model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    train_time = time.time() - start_time

    train_acc = model.get_accuracy(x, y)

    return model, {
        'train_acc': train_acc,
        'train_time': train_time
    }


def train_random_noise(x: torch.Tensor, y: torch.Tensor,
                       device: torch.device,
                       epochs_per_layer: int = 100,
                       verbose: bool = True) -> Tuple[FFNetwork, Dict]:
    """Train with random noise strategy."""
    print("\n" + "="*60)
    print("Strategy: RANDOM NOISE (matched statistics)")
    print("="*60)

    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Positive: correct label embedded
    x_pos = overlay_label(x, y)

    # Negative: random noise with matched statistics
    x_neg = random_noise_strategy(x)

    start_time = time.time()
    model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    train_time = time.time() - start_time

    train_acc = model.get_accuracy(x, y)

    return model, {
        'train_acc': train_acc,
        'train_time': train_time
    }


def train_masking(x: torch.Tensor, y: torch.Tensor,
                  device: torch.device,
                  epochs_per_layer: int = 100,
                  mask_ratio: float = 0.5,
                  verbose: bool = True) -> Tuple[FFNetwork, Dict]:
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
        'train_time': train_time
    }


# ============================================================
# Main Experiment
# ============================================================

def run_comparison(epochs_per_layer: int = 100, seed: int = 1234):
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
    print("CORRECT STRATEGY COMPARISON EXPERIMENT")
    print("="*60)
    print(f"Architecture: [784, 500, 500]")
    print(f"Epochs per layer: {epochs_per_layer}")
    print(f"Batch size: {x_train.size(0)} (full batch)")
    print(f"Threshold: 2.0")
    print(f"Learning rate: 0.03")
    print(f"Goodness: MEAN of squared activations")

    results = {}

    # Strategy 1: Label Embedding
    model_le, stats_le = train_label_embedding(
        x_train, y_train, device, epochs_per_layer
    )
    test_acc_le = model_le.get_accuracy(x_test, y_test)
    results['label_embedding'] = {
        'train_acc': stats_le['train_acc'],
        'test_acc': test_acc_le,
        'train_time': stats_le['train_time']
    }
    print(f"\nLabel Embedding - Train: {stats_le['train_acc']*100:.2f}%, Test: {test_acc_le*100:.2f}%")

    # Strategy 2: Random Noise
    model_rn, stats_rn = train_random_noise(
        x_train, y_train, device, epochs_per_layer
    )
    test_acc_rn = model_rn.get_accuracy(x_test, y_test)
    results['random_noise'] = {
        'train_acc': stats_rn['train_acc'],
        'test_acc': test_acc_rn,
        'train_time': stats_rn['train_time']
    }
    print(f"\nRandom Noise - Train: {stats_rn['train_acc']*100:.2f}%, Test: {test_acc_rn*100:.2f}%")

    # Strategy 3: Masking
    model_mk, stats_mk = train_masking(
        x_train, y_train, device, epochs_per_layer, mask_ratio=0.5
    )
    test_acc_mk = model_mk.get_accuracy(x_test, y_test)
    results['masking'] = {
        'train_acc': stats_mk['train_acc'],
        'test_acc': test_acc_mk,
        'train_time': stats_mk['train_time']
    }
    print(f"\nMasking - Train: {stats_mk['train_acc']*100:.2f}%, Test: {test_acc_mk*100:.2f}%")

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Strategy':<20} {'Train Acc':<15} {'Test Acc':<15} {'Time (s)':<10}")
    print("-"*60)
    for name, data in results.items():
        print(f"{name:<20} {data['train_acc']*100:.2f}%{'':<8} {data['test_acc']*100:.2f}%{'':<8} {data['train_time']:.1f}")

    # Add metadata
    results['metadata'] = {
        'epochs_per_layer': epochs_per_layer,
        'batch_size': 50000,
        'architecture': [784, 500, 500],
        'threshold': 2.0,
        'learning_rate': 0.03,
        'goodness': 'mean',
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs per layer (default 100, full test needs 1000)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()

    results = run_comparison(epochs_per_layer=args.epochs, seed=args.seed)

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'correct_comparison.json'
    save_results(results, str(output_path))

    return results


if __name__ == "__main__":
    main()
