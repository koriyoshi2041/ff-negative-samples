#!/usr/bin/env python3
"""
Image Mixing Transfer Learning Experiment
==========================================

Key finding from correct_neg_strategy_comparison.json:
- image_mixing Linear Probe accuracy: 77.2% on MNIST
- This is remarkably high for an unsupervised (label-free) strategy!

Research Question:
Do features learned via image_mixing transfer better than label_embedding?
Hypothesis: Image mixing learns more generic visual features because it doesn't
rely on label structure, potentially making it more transferable.

Experiment Design:
1. Train FF with image_mixing (500 epochs/layer, full batch) on MNIST
2. Train FF with label_embedding (same settings) on MNIST
3. Extract features from both models
4. Test Linear Probe on MNIST (source domain)
5. Transfer to Fashion-MNIST, test Linear Probe (target domain)
6. Compare transfer performance

Author: Clawd (for Parafee)
Date: 2026-02-09
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ============================================================
# Device Setup
# ============================================================

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Data Loading
# ============================================================

def get_mnist_data(device: torch.device, data_dir: str = './data'):
    """Load full MNIST dataset to device."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    return (x_train.to(device), y_train.to(device)), (x_test.to(device), y_test.to(device))


def get_fashion_mnist_data(device: torch.device, data_dir: str = './data'):
    """Load full Fashion-MNIST dataset to device."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    return (x_train.to(device), y_train.to(device)), (x_test.to(device), y_test.to(device))


# ============================================================
# Negative Sample Strategies
# ============================================================

def overlay_label(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Label Embedding Strategy (Hinton's original).
    Replace first 10 pixels with one-hot-encoded label.
    CRITICAL: Use x.max() as the label value!
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


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


# ============================================================
# Forward-Forward Implementation
# ============================================================

class FFLayer(nn.Module):
    """Forward-Forward Layer with correct implementation."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with L2 normalization."""
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """MEAN of squared activations (not sum!)."""
        return h.pow(2).mean(dim=1)

    def train_layer(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                    num_epochs: int = 500, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
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

            if verbose and hasattr(iterator, 'set_postfix'):
                pos_acc = (g_pos > self.threshold).float().mean().item()
                neg_acc = (g_neg < self.threshold).float().mean().item()
                iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pos': f'{pos_acc:.1%}',
                    'neg': f'{neg_acc:.1%}'
                })

        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFNetwork(nn.Module):
    """Forward-Forward Network with greedy training."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dims = dims
        for d in range(len(dims) - 1):
            self.layers.append(FFLayer(dims[d], dims[d + 1], threshold, lr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_all_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get activations from all layers."""
        activations = []
        h = x
        for layer in self.layers:
            h = layer(h)
            activations.append(h)
        return activations

    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 500, verbose: bool = True):
        """Greedy layer-by-layer training."""
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\nTraining layer {i}...')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg, epochs_per_layer, verbose)

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels (for label_embedding strategy)."""
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

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute accuracy using goodness-based prediction."""
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()


# ============================================================
# Linear Probe
# ============================================================

class LinearProbe(nn.Module):
    """Linear classifier for feature assessment."""

    def __init__(self, input_dim: int, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_probe(
    features_train: torch.Tensor,
    y_train: torch.Tensor,
    features_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = False
) -> Dict:
    """
    Train a linear probe on features.
    Returns train and test accuracy.
    """
    device = features_train.device
    feature_dim = features_train.size(1)
    num_classes = int(y_train.max().item()) + 1

    probe = LinearProbe(feature_dim, num_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'test_acc': []}
    iterator = tqdm(range(epochs), desc="Probe") if verbose else range(epochs)

    for _ in iterator:
        probe.train()
        optimizer.zero_grad()
        logits = probe(features_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        # Assess
        probe.train(False)
        with torch.no_grad():
            train_acc = (probe(features_train).argmax(dim=1) == y_train).float().mean().item()
            test_acc = (probe(features_test).argmax(dim=1) == y_test).float().mean().item()

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    return {
        'train_acc': history['train_acc'][-1],
        'test_acc': history['test_acc'][-1],
        'best_test_acc': max(history['test_acc']),
        'history': history
    }


def extract_features(model: FFNetwork, x: torch.Tensor,
                     use_label_embedding: bool = False) -> torch.Tensor:
    """
    Extract features from FF model.

    For label_embedding: use label=0 for consistent features
    For image_mixing: use raw input (no label embedding needed)
    """
    model.train(False)
    with torch.no_grad():
        if use_label_embedding:
            # For label_embedding trained models, need to embed a label
            batch_size = x.shape[0]
            x_embedded = overlay_label(x, torch.zeros(batch_size, dtype=torch.long, device=x.device))
            activations = model.get_all_activations(x_embedded)
        else:
            # For image_mixing, use raw input
            activations = model.get_all_activations(x)

        # Concatenate all layer activations
        return torch.cat(activations, dim=1)


# ============================================================
# Training Functions
# ============================================================

def train_image_mixing_ff(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    epochs_per_layer: int = 500,
    verbose: bool = True
) -> Tuple[FFNetwork, Dict]:
    """Train FF with image_mixing strategy."""
    print("\n" + "="*60)
    print("Training: IMAGE MIXING Strategy")
    print("="*60)

    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Positive: original images with label embedded
    x_pos = overlay_label(x, y)

    # Negative: mixed/chimera images (no labels)
    x_neg = image_mixing_strategy(x)

    start_time = time.time()
    model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    train_time = time.time() - start_time

    return model, {'train_time': train_time}


def train_label_embedding_ff(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    epochs_per_layer: int = 500,
    verbose: bool = True
) -> Tuple[FFNetwork, Dict]:
    """Train FF with label_embedding strategy (Hinton's original)."""
    print("\n" + "="*60)
    print("Training: LABEL EMBEDDING Strategy (Hinton's original)")
    print("="*60)

    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Positive: correct label embedded
    x_pos = overlay_label(x, y)

    # Negative: shuffled (wrong) labels embedded
    perm = torch.randperm(x.size(0), device=device)
    x_neg = overlay_label(x, y[perm])

    start_time = time.time()
    model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    train_time = time.time() - start_time

    return model, {'train_time': train_time}


# ============================================================
# Main Experiment
# ============================================================

def run_image_mixing_transfer_experiment(
    epochs_per_layer: int = 500,
    probe_epochs: int = 100,
    seed: int = 1234,
    verbose: bool = True
) -> Dict:
    """
    Compare image_mixing vs label_embedding transfer learning.

    Key hypothesis: image_mixing learns more generic features
    because it doesn't encode task-specific label information.
    """
    print("="*70)
    print("IMAGE MIXING TRANSFER LEARNING EXPERIMENT")
    print("="*70)
    print(f"\nExperiment Settings:")
    print(f"  Epochs per layer: {epochs_per_layer}")
    print(f"  Linear probe epochs: {probe_epochs}")
    print(f"  Seed: {seed}")
    print(f"  Architecture: [784, 500, 500]")
    print(f"  Threshold: 2.0, LR: 0.03")
    print(f"  Goodness: MEAN (correct implementation)")

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"  Device: {device}")

    results = {
        'experiment': 'image_mixing_transfer',
        'config': {
            'epochs_per_layer': epochs_per_layer,
            'probe_epochs': probe_epochs,
            'seed': seed,
            'device': str(device),
            'architecture': [784, 500, 500],
            'threshold': 2.0,
            'learning_rate': 0.03,
            'goodness': 'mean'
        }
    }

    # Load data
    print("\n" + "-"*60)
    print("Loading datasets...")
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)

    (mnist_train, mnist_train_y), (mnist_test, mnist_test_y) = get_mnist_data(device, str(data_dir))
    (fmnist_train, fmnist_train_y), (fmnist_test, fmnist_test_y) = get_fashion_mnist_data(device, str(data_dir))

    print(f"  MNIST: {len(mnist_train)} train, {len(mnist_test)} test")
    print(f"  Fashion-MNIST: {len(fmnist_train)} train, {len(fmnist_test)} test")

    # ============================================================
    # Train both models
    # ============================================================

    # Train Image Mixing FF
    torch.manual_seed(seed)
    model_mixing, stats_mixing = train_image_mixing_ff(
        mnist_train, mnist_train_y, device, epochs_per_layer, verbose
    )

    # Train Label Embedding FF
    torch.manual_seed(seed)
    model_label, stats_label = train_label_embedding_ff(
        mnist_train, mnist_train_y, device, epochs_per_layer, verbose
    )

    # ============================================================
    # Test on MNIST (source domain)
    # ============================================================
    print("\n" + "="*60)
    print("Testing on MNIST (Source Domain)")
    print("="*60)

    # Image Mixing features on MNIST
    print("\nImage Mixing - MNIST Linear Probe...")
    feat_mixing_mnist_train = extract_features(model_mixing, mnist_train, use_label_embedding=False)
    feat_mixing_mnist_test = extract_features(model_mixing, mnist_test, use_label_embedding=False)

    probe_mixing_mnist = train_linear_probe(
        feat_mixing_mnist_train, mnist_train_y,
        feat_mixing_mnist_test, mnist_test_y,
        epochs=probe_epochs, verbose=verbose
    )

    # Label Embedding - goodness-based accuracy
    print("\nLabel Embedding - MNIST Goodness-based Accuracy...")
    label_mnist_train_acc = model_label.get_accuracy(mnist_train, mnist_train_y)
    label_mnist_test_acc = model_label.get_accuracy(mnist_test, mnist_test_y)

    # Label Embedding - linear probe for fair comparison
    print("\nLabel Embedding - MNIST Linear Probe...")
    feat_label_mnist_train = extract_features(model_label, mnist_train, use_label_embedding=True)
    feat_label_mnist_test = extract_features(model_label, mnist_test, use_label_embedding=True)

    probe_label_mnist = train_linear_probe(
        feat_label_mnist_train, mnist_train_y,
        feat_label_mnist_test, mnist_test_y,
        epochs=probe_epochs, verbose=verbose
    )

    results['mnist_source'] = {
        'image_mixing': {
            'linear_probe_train_acc': probe_mixing_mnist['train_acc'],
            'linear_probe_test_acc': probe_mixing_mnist['test_acc'],
            'linear_probe_best_acc': probe_mixing_mnist['best_test_acc'],
            'train_time': stats_mixing['train_time']
        },
        'label_embedding': {
            'goodness_train_acc': label_mnist_train_acc,
            'goodness_test_acc': label_mnist_test_acc,
            'linear_probe_train_acc': probe_label_mnist['train_acc'],
            'linear_probe_test_acc': probe_label_mnist['test_acc'],
            'linear_probe_best_acc': probe_label_mnist['best_test_acc'],
            'train_time': stats_label['train_time']
        }
    }

    print(f"\nMNIST Source Domain Results:")
    print(f"  Image Mixing  - Linear Probe: {probe_mixing_mnist['test_acc']*100:.2f}%")
    print(f"  Label Embed   - Goodness:     {label_mnist_test_acc*100:.2f}%")
    print(f"  Label Embed   - Linear Probe: {probe_label_mnist['test_acc']*100:.2f}%")

    # ============================================================
    # Transfer to Fashion-MNIST (target domain)
    # ============================================================
    print("\n" + "="*60)
    print("Transfer to Fashion-MNIST (Target Domain)")
    print("="*60)

    # Image Mixing features on Fashion-MNIST
    print("\nImage Mixing - Fashion-MNIST Linear Probe...")
    feat_mixing_fmnist_train = extract_features(model_mixing, fmnist_train, use_label_embedding=False)
    feat_mixing_fmnist_test = extract_features(model_mixing, fmnist_test, use_label_embedding=False)

    probe_mixing_fmnist = train_linear_probe(
        feat_mixing_fmnist_train, fmnist_train_y,
        feat_mixing_fmnist_test, fmnist_test_y,
        epochs=probe_epochs, verbose=verbose
    )

    # Label Embedding features on Fashion-MNIST
    print("\nLabel Embedding - Fashion-MNIST Linear Probe...")
    feat_label_fmnist_train = extract_features(model_label, fmnist_train, use_label_embedding=True)
    feat_label_fmnist_test = extract_features(model_label, fmnist_test, use_label_embedding=True)

    probe_label_fmnist = train_linear_probe(
        feat_label_fmnist_train, fmnist_train_y,
        feat_label_fmnist_test, fmnist_test_y,
        epochs=probe_epochs, verbose=verbose
    )

    results['fashion_mnist_transfer'] = {
        'image_mixing': {
            'linear_probe_train_acc': probe_mixing_fmnist['train_acc'],
            'linear_probe_test_acc': probe_mixing_fmnist['test_acc'],
            'linear_probe_best_acc': probe_mixing_fmnist['best_test_acc']
        },
        'label_embedding': {
            'linear_probe_train_acc': probe_label_fmnist['train_acc'],
            'linear_probe_test_acc': probe_label_fmnist['test_acc'],
            'linear_probe_best_acc': probe_label_fmnist['best_test_acc']
        }
    }

    print(f"\nFashion-MNIST Transfer Results:")
    print(f"  Image Mixing  - Linear Probe: {probe_mixing_fmnist['test_acc']*100:.2f}%")
    print(f"  Label Embed   - Linear Probe: {probe_label_fmnist['test_acc']*100:.2f}%")

    # ============================================================
    # Random Baseline
    # ============================================================
    print("\n" + "-"*60)
    print("Random Baseline (untrained network)")

    torch.manual_seed(seed)
    model_random = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    feat_random_fmnist_train = extract_features(model_random, fmnist_train, use_label_embedding=False)
    feat_random_fmnist_test = extract_features(model_random, fmnist_test, use_label_embedding=False)

    probe_random_fmnist = train_linear_probe(
        feat_random_fmnist_train, fmnist_train_y,
        feat_random_fmnist_test, fmnist_test_y,
        epochs=probe_epochs, verbose=False
    )

    results['random_baseline'] = {
        'fashion_mnist_test_acc': probe_random_fmnist['test_acc']
    }

    print(f"  Random Init   - Linear Probe: {probe_random_fmnist['test_acc']*100:.2f}%")

    # ============================================================
    # Analysis
    # ============================================================
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    mixing_transfer_gain = probe_mixing_fmnist['test_acc'] - probe_random_fmnist['test_acc']
    label_transfer_gain = probe_label_fmnist['test_acc'] - probe_random_fmnist['test_acc']
    mixing_vs_label = probe_mixing_fmnist['test_acc'] - probe_label_fmnist['test_acc']

    print(f"\nTransfer Gain over Random Baseline:")
    print(f"  Image Mixing:  {mixing_transfer_gain*100:+.2f}%")
    print(f"  Label Embed:   {label_transfer_gain*100:+.2f}%")

    print(f"\nImage Mixing vs Label Embedding (Fashion-MNIST):")
    print(f"  Difference:    {mixing_vs_label*100:+.2f}%")

    # Retention ratio: how much performance is retained during transfer
    mixing_retention = probe_mixing_fmnist['test_acc'] / probe_mixing_mnist['test_acc'] if probe_mixing_mnist['test_acc'] > 0 else 0
    label_retention = probe_label_fmnist['test_acc'] / probe_label_mnist['test_acc'] if probe_label_mnist['test_acc'] > 0 else 0

    print(f"\nPerformance Retention (Target/Source):")
    print(f"  Image Mixing:  {mixing_retention*100:.1f}%")
    print(f"  Label Embed:   {label_retention*100:.1f}%")

    results['analysis'] = {
        'mixing_transfer_gain': mixing_transfer_gain,
        'label_transfer_gain': label_transfer_gain,
        'mixing_vs_label_gap': mixing_vs_label,
        'mixing_retention_ratio': mixing_retention,
        'label_retention_ratio': label_retention,
        'mixing_wins_transfer': mixing_vs_label > 0,
        'key_finding': (
            "Image mixing learns MORE transferable features" if mixing_vs_label > 0.01
            else "Image mixing learns LESS transferable features" if mixing_vs_label < -0.01
            else "Both strategies have SIMILAR transfer performance"
        )
    }

    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print(f"\n{'Strategy':<18} {'MNIST (Source)':<18} {'F-MNIST (Target)':<18} {'Retention':<12}")
    print("-"*70)
    print(f"{'Image Mixing':<18} {probe_mixing_mnist['test_acc']*100:>6.2f}%{'':<10} {probe_mixing_fmnist['test_acc']*100:>6.2f}%{'':<10} {mixing_retention*100:>6.1f}%")
    print(f"{'Label Embedding':<18} {probe_label_mnist['test_acc']*100:>6.2f}%{'':<10} {probe_label_fmnist['test_acc']*100:>6.2f}%{'':<10} {label_retention*100:>6.1f}%")
    print(f"{'Random Baseline':<18} {'-':<18} {probe_random_fmnist['test_acc']*100:>6.2f}%{'':<10} {'-':<12}")

    print(f"\nKey Finding: {results['analysis']['key_finding']}")

    results['timestamp'] = datetime.now().isoformat()

    return results


def save_results(results: Dict, output_path: str):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Image Mixing Transfer Learning Experiment")
    parser.add_argument('--epochs', type=int, default=500,
                        help='FF epochs per layer (default: 500)')
    parser.add_argument('--probe-epochs', type=int, default=100,
                        help='Linear probe training epochs (default: 100)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    args = parser.parse_args()

    results = run_image_mixing_transfer_experiment(
        epochs_per_layer=args.epochs,
        probe_epochs=args.probe_epochs,
        seed=args.seed,
        verbose=not args.quiet
    )

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'image_mixing_transfer.json'
    save_results(results, str(output_path))

    print("\nExperiment complete!")
    return results


if __name__ == "__main__":
    main()
