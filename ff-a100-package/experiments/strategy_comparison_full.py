#!/usr/bin/env python3
"""
Negative Sample Strategy Full Comparison
=========================================
Fixed version with proper epoch count (1000 epochs/layer for convergence).

Expected accuracy with correct implementation: ~93%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class FFLayer(nn.Module):
    """FF Layer with correct goodness calculation (MEAN not SUM)."""

    def __init__(self, in_features: int, out_features: int, threshold: float = 2.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.optimizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = x / (x.norm(2, dim=1, keepdim=True) + 1e-8)
        return self.relu(self.linear(x_normalized))

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        """CRITICAL: Use MEAN not SUM."""
        return (x ** 2).mean(dim=1)

    def ff_loss(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor) -> torch.Tensor:
        loss_pos = torch.log(1 + torch.exp(self.threshold - pos_goodness)).mean()
        loss_neg = torch.log(1 + torch.exp(neg_goodness - self.threshold)).mean()
        return loss_pos + loss_neg


class FFNetwork(nn.Module):
    def __init__(self, layer_sizes: List[int], threshold: float = 2.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(FFLayer(layer_sizes[i], layer_sizes[i+1], threshold))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations

    def setup_optimizers(self, lr: float = 0.03):
        for layer in self.layers:
            layer.optimizer = optim.Adam(layer.parameters(), lr=lr)


def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Embed label in first 10 pixels using x.max() (not 1.0!)."""
    x = x.clone()
    label_value = x.max()  # CRITICAL: Use x.max() not 1.0
    x[:, :num_classes] = 0
    x[torch.arange(len(y)), y] = label_value
    return x


def create_negative_label_embedding(x: torch.Tensor, y: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Create negative samples by shuffling labels."""
    wrong_labels = torch.randint(0, num_classes - 1, (len(y),), device=y.device)
    wrong_labels = (wrong_labels + y + 1) % num_classes
    return overlay_y_on_x(x, wrong_labels, num_classes)


def create_negative_image_mixing(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Create negative samples by mixing two random images."""
    idx1 = torch.randperm(len(x))
    idx2 = torch.randperm(len(x))
    alpha = torch.rand(len(x), 1, device=x.device) * 0.5 + 0.25
    return x[idx1] * alpha + x[idx2] * (1 - alpha)


def create_negative_random_noise(x: torch.Tensor) -> torch.Tensor:
    """Create negative samples with matched statistics noise."""
    noise = torch.randn_like(x)
    noise = noise * x.std() + x.mean()
    return noise


def train_layer_greedy(
    layer: FFLayer,
    pos_data: torch.Tensor,
    neg_data: torch.Tensor,
    epochs: int = 1000,
    quiet: bool = False
) -> None:
    """Train a single layer to convergence (greedy layer-by-layer)."""
    layer.optimizer = optim.Adam(layer.parameters(), lr=0.03)

    iterator = range(epochs) if quiet else tqdm(range(epochs), desc="Training layer")
    for epoch in iterator:
        layer.optimizer.zero_grad()

        pos_out = layer(pos_data)
        neg_out = layer(neg_data)

        pos_goodness = layer.goodness(pos_out)
        neg_goodness = layer.goodness(neg_out)

        loss = layer.ff_loss(pos_goodness, neg_goodness)
        loss.backward()
        layer.optimizer.step()


def get_accuracy(
    model: FFNetwork,
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 10
) -> float:
    """Test accuracy using goodness-based classification."""
    model.eval_mode = True
    with torch.no_grad():
        all_goodness = []
        for label in range(num_classes):
            test_labels = torch.full((len(x),), label, device=x.device)
            x_with_label = overlay_y_on_x(x, test_labels, num_classes)

            activations = model(x_with_label)
            total_goodness = sum(layer.goodness(act) for layer, act in zip(model.layers, activations))
            all_goodness.append(total_goodness)

        all_goodness = torch.stack(all_goodness, dim=1)
        predictions = all_goodness.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()

    return accuracy


def linear_probe_accuracy(
    model: FFNetwork,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 100
) -> Tuple[float, float]:
    """Test features using linear probe (for label-free strategies)."""
    device = x_train.device

    with torch.no_grad():
        train_features = model(x_train)[-1]
        test_features = model(x_test)[-1]

    # Train linear classifier
    classifier = nn.Linear(train_features.shape[1], 10).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = classifier(train_features)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # Test
    with torch.no_grad():
        train_acc = (classifier(train_features).argmax(1) == y_train).float().mean().item()
        test_acc = (classifier(test_features).argmax(1) == y_test).float().mean().item()

    return train_acc, test_acc


def run_strategy_experiment(
    strategy_name: str,
    create_negative_fn,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs_per_layer: int = 1000,
    uses_labels: bool = True,
    quiet: bool = False
) -> Dict:
    """Run experiment for a single strategy."""
    device = x_train.device

    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")

    # Create model
    model = FFNetwork([784, 500, 500], threshold=2.0).to(device)

    start_time = time.time()

    # Layer-by-layer greedy training
    h_pos = overlay_y_on_x(x_train, y_train) if uses_labels else x_train
    h_neg = create_negative_fn(x_train, y_train) if uses_labels else create_negative_fn(x_train)

    for layer_idx, layer in enumerate(model.layers):
        print(f"Training layer {layer_idx}...")
        train_layer_greedy(layer, h_pos, h_neg, epochs=epochs_per_layer, quiet=quiet)

        with torch.no_grad():
            h_pos = layer(h_pos)
            h_neg = layer(h_neg)

    train_time = time.time() - start_time

    # Test
    if uses_labels:
        train_acc = get_accuracy(model, x_train, y_train)
        test_acc = get_accuracy(model, x_test, y_test)
        result = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_time': train_time,
            'uses_labels': True,
        }
    else:
        # For label-free strategies, use linear probe
        lp_train, lp_test = linear_probe_accuracy(model, x_train, y_train, x_test, y_test)
        result = {
            'train_acc': 0.0,
            'test_acc': 0.0,
            'linear_probe_train_acc': lp_train,
            'linear_probe_test_acc': lp_test,
            'train_time': train_time,
            'uses_labels': False,
        }

    print(f"Results: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Full Strategy Comparison')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs per layer')
    parser.add_argument('--output', type=str, default='results/strategy_comparison_full.json')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs per layer: {args.epochs}")

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Full batch
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    results = {}

    # Strategy 1: Label Embedding (Hinton's original)
    results['label_embedding'] = run_strategy_experiment(
        'label_embedding',
        create_negative_label_embedding,
        x_train, y_train, x_test, y_test,
        epochs_per_layer=args.epochs,
        uses_labels=True,
        quiet=args.quiet
    )

    # Strategy 2: Image Mixing (label-free)
    results['image_mixing'] = run_strategy_experiment(
        'image_mixing',
        lambda x: create_negative_image_mixing(x, torch.zeros(len(x), device=x.device)),
        x_train, y_train, x_test, y_test,
        epochs_per_layer=args.epochs,
        uses_labels=False,
        quiet=args.quiet
    )

    # Strategy 3: Random Noise (label-free)
    results['random_noise'] = run_strategy_experiment(
        'random_noise',
        lambda x: create_negative_random_noise(x),
        x_train, y_train, x_test, y_test,
        epochs_per_layer=args.epochs,
        uses_labels=False,
        quiet=args.quiet
    )

    # Add metadata
    results['metadata'] = {
        'epochs_per_layer': args.epochs,
        'batch_size': 50000,
        'architecture': [784, 500, 500],
        'threshold': 2.0,
        'learning_rate': 0.03,
        'device': str(device),
        'goodness': 'mean (CORRECT)',
        'label_scale': 'x.max() (CORRECT)',
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, result in results.items():
        if name == 'metadata':
            continue
        if result.get('uses_labels'):
            print(f"{name}: Test Acc = {result['test_acc']*100:.2f}%")
        else:
            print(f"{name}: Linear Probe = {result['linear_probe_test_acc']*100:.2f}%")


if __name__ == '__main__':
    main()
