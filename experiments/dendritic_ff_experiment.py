#!/usr/bin/env python3
"""
Dendritic Forward-Forward Experiment
=====================================

Experiment Design:
------------------
Based on Wright et al. (Science 2025) neuroscience findings about distinct
learning rules in different dendritic compartments.

Hypothesis:
-----------
The apical compartment learns contextual representations that are more
general and transferable, while basal compartment learns task-specific
features. This should lead to better transfer learning performance.

Experiments:
1. Train Standard FF and Dendritic FF on MNIST (500 epochs/layer)
2. Transfer to Fashion-MNIST (freeze features, train linear head)
3. Compare transfer performance

Expected Results:
- Dendritic FF should show comparable or better source task accuracy
- Dendritic FF should show BETTER transfer learning (more general representations)

Author: Clawd (for Parafee)
Date: 2026-02-09
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dendritic_ff import (
    DendriticFFNetwork,
    StandardFFNetwork,
    overlay_y_on_x,
    get_device
)


# ============================================================
# Data Loading
# ============================================================

def get_mnist_data(device: torch.device):
    """Load full MNIST dataset to device."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    return (x_train.to(device), y_train.to(device)), (x_test.to(device), y_test.to(device))


def get_fashion_mnist_data(device: torch.device):
    """Load full Fashion-MNIST dataset to device."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    return (x_train.to(device), y_train.to(device)), (x_test.to(device), y_test.to(device))


# ============================================================
# Transfer Learning
# ============================================================

class LinearHead(nn.Module):
    """Linear classification head for transfer learning."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_transfer_head(
    features_train: torch.Tensor,
    y_train: torch.Tensor,
    features_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.01,
    verbose: bool = True
) -> Dict:
    """Train linear head on frozen features."""
    feature_dim = features_train.shape[1]
    num_classes = int(y_train.max().item()) + 1

    head = LinearHead(feature_dim, num_classes).to(features_train.device)
    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'test_acc': [], 'loss': []}

    iterator = tqdm(range(epochs), desc="Transfer training") if verbose else range(epochs)

    for epoch in iterator:
        head.train()
        indices = torch.randperm(len(features_train))
        epoch_losses = []

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = features_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            outputs = head(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Compute accuracy
        head.train(False)
        with torch.no_grad():
            train_preds = head(features_train).argmax(dim=1)
            test_preds = head(features_test).argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()
            test_acc = (test_preds == y_test).float().mean().item()

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['loss'].append(np.mean(epoch_losses))

        if verbose:
            iterator.set_postfix({
                'loss': f'{np.mean(epoch_losses):.4f}',
                'train': f'{train_acc*100:.1f}%',
                'test': f'{test_acc*100:.1f}%'
            })

    return {
        'final_accuracy': history['test_acc'][-1],
        'best_accuracy': max(history['test_acc']),
        'history': history
    }


# ============================================================
# Main Experiment
# ============================================================

def run_dendritic_ff_experiment(
    epochs_per_layer: int = 500,
    transfer_epochs: int = 100,
    alpha: float = 0.3,
    coactivity_radius: int = 5,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run the Dendritic FF experiment.

    Compares:
    1. Standard FF (baseline)
    2. Dendritic FF (with basal/apical compartments)

    On:
    - Source task: MNIST classification
    - Transfer task: Fashion-MNIST classification
    """
    print("="*70)
    print("DENDRITIC FORWARD-FORWARD EXPERIMENT")
    print("Based on Wright et al. (Science 2025) neuroscience findings")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  Epochs per layer: {epochs_per_layer}")
    print(f"  Transfer epochs: {transfer_epochs}")
    print(f"  Alpha (apical weight): {alpha}")
    print(f"  Co-activity radius: {coactivity_radius}")
    print(f"  Seed: {seed}")

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"  Device: {device}")

    results = {
        'experiment': 'dendritic_ff',
        'config': {
            'epochs_per_layer': epochs_per_layer,
            'transfer_epochs': transfer_epochs,
            'alpha': alpha,
            'coactivity_radius': coactivity_radius,
            'seed': seed,
            'device': str(device),
            'architecture': [784, 500, 500],
            'threshold': 2.0,
            'lr': 0.03
        }
    }

    # Load data
    print("\n" + "-"*60)
    print("Loading datasets...")
    (mnist_train, mnist_train_y), (mnist_test, mnist_test_y) = get_mnist_data(device)
    (fmnist_train, fmnist_train_y), (fmnist_test, fmnist_test_y) = get_fashion_mnist_data(device)
    print(f"  MNIST: {len(mnist_train)} train, {len(mnist_test)} test")
    print(f"  Fashion-MNIST: {len(fmnist_train)} train, {len(fmnist_test)} test")

    # ============================================================
    # PHASE 1: Train Standard FF
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 1: Training Standard Forward-Forward")
    print("="*60)

    torch.manual_seed(seed)
    standard_ff = StandardFFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Create positive/negative samples
    x_pos_std = overlay_y_on_x(mnist_train, mnist_train_y)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg_std = overlay_y_on_x(mnist_train, mnist_train_y[rnd])

    # Train
    std_start = time.time()
    standard_ff.train_greedy(x_pos_std, x_neg_std, epochs_per_layer, verbose)
    std_train_time = time.time() - std_start

    # Compute accuracy
    std_train_acc = standard_ff.get_accuracy(mnist_train, mnist_train_y)
    std_test_acc = standard_ff.get_accuracy(mnist_test, mnist_test_y)

    print(f"\nStandard FF Source Results:")
    print(f"  Train accuracy: {std_train_acc*100:.2f}%")
    print(f"  Test accuracy:  {std_test_acc*100:.2f}%")
    print(f"  Training time:  {std_train_time:.1f}s")

    results['standard_ff'] = {
        'source_train_acc': std_train_acc,
        'source_test_acc': std_test_acc,
        'train_time': std_train_time
    }

    # ============================================================
    # PHASE 2: Train Dendritic FF
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 2: Training Dendritic Forward-Forward")
    print("="*60)

    torch.manual_seed(seed)
    dendritic_ff = DendriticFFNetwork(
        dims=[784, 500, 500],
        context_dims=10,  # Label embedding dimension
        threshold=2.0,
        alpha=alpha,
        coactivity_radius=coactivity_radius,
        context_mode='label',
        lr=0.03
    ).to(device)

    # For dendritic FF, we don't embed label in input (it's in context)
    # But for fair comparison, we still use the same input format
    x_pos_dend = overlay_y_on_x(mnist_train, mnist_train_y)
    x_neg_dend = overlay_y_on_x(mnist_train, mnist_train_y[rnd])

    # Train
    dend_start = time.time()
    dendritic_ff.train_greedy(
        x_pos_dend, x_neg_dend,
        mnist_train_y, mnist_train_y[rnd],  # Pass labels for context
        epochs_per_layer, verbose
    )
    dend_train_time = time.time() - dend_start

    # Compute accuracy
    dend_train_acc = dendritic_ff.get_accuracy(mnist_train, mnist_train_y)
    dend_test_acc = dendritic_ff.get_accuracy(mnist_test, mnist_test_y)

    print(f"\nDendritic FF Source Results:")
    print(f"  Train accuracy: {dend_train_acc*100:.2f}%")
    print(f"  Test accuracy:  {dend_test_acc*100:.2f}%")
    print(f"  Training time:  {dend_train_time:.1f}s")

    # Get layer statistics
    layer_stats = dendritic_ff.get_layer_statistics(mnist_test, mnist_test_y)
    print(f"\nLayer Statistics:")
    for i in range(len(dendritic_ff.layers)):
        print(f"  Layer {i}: basal={layer_stats['basal_goodness'][i]:.3f}, "
              f"apical={layer_stats['apical_goodness'][i]:.3f}, "
              f"total={layer_stats['total_goodness'][i]:.3f}")

    results['dendritic_ff'] = {
        'source_train_acc': dend_train_acc,
        'source_test_acc': dend_test_acc,
        'train_time': dend_train_time,
        'layer_stats': layer_stats
    }

    # ============================================================
    # PHASE 3: Transfer Learning
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 3: Transfer Learning to Fashion-MNIST")
    print("="*60)

    # Extract features
    print("\nExtracting features...")
    std_features_train = standard_ff.get_features(fmnist_train)
    std_features_test = standard_ff.get_features(fmnist_test)
    print(f"  Standard FF features: {std_features_train.shape}")

    dend_features_train = dendritic_ff.get_features(fmnist_train)
    dend_features_test = dendritic_ff.get_features(fmnist_test)
    print(f"  Dendritic FF features: {dend_features_train.shape}")

    # Random baseline
    torch.manual_seed(seed + 1000)  # Different seed for random init
    random_ff = StandardFFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    random_features_train = random_ff.get_features(fmnist_train)
    random_features_test = random_ff.get_features(fmnist_test)
    print(f"  Random init features: {random_features_train.shape}")

    # Train transfer heads
    print("\n" + "-"*60)
    print("Training Standard FF transfer head...")
    std_transfer = train_transfer_head(
        std_features_train, fmnist_train_y,
        std_features_test, fmnist_test_y,
        epochs=transfer_epochs, verbose=verbose
    )
    results['standard_ff']['transfer_accuracy'] = std_transfer['final_accuracy']
    results['standard_ff']['transfer_best'] = std_transfer['best_accuracy']

    print("\n" + "-"*60)
    print("Training Dendritic FF transfer head...")
    dend_transfer = train_transfer_head(
        dend_features_train, fmnist_train_y,
        dend_features_test, fmnist_test_y,
        epochs=transfer_epochs, verbose=verbose
    )
    results['dendritic_ff']['transfer_accuracy'] = dend_transfer['final_accuracy']
    results['dendritic_ff']['transfer_best'] = dend_transfer['best_accuracy']

    print("\n" + "-"*60)
    print("Training Random init transfer head...")
    random_transfer = train_transfer_head(
        random_features_train, fmnist_train_y,
        random_features_test, fmnist_test_y,
        epochs=transfer_epochs, verbose=verbose
    )
    results['random'] = {
        'transfer_accuracy': random_transfer['final_accuracy'],
        'transfer_best': random_transfer['best_accuracy']
    }

    # ============================================================
    # PHASE 4: Analysis
    # ============================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Method':<25} {'Source Acc':>12} {'Transfer Acc':>14} {'Transfer Best':>14}")
    print("-" * 67)
    print(f"{'Standard FF':<25} {std_test_acc*100:>11.2f}% "
          f"{std_transfer['final_accuracy']*100:>13.2f}% "
          f"{std_transfer['best_accuracy']*100:>13.2f}%")
    print(f"{'Dendritic FF':<25} {dend_test_acc*100:>11.2f}% "
          f"{dend_transfer['final_accuracy']*100:>13.2f}% "
          f"{dend_transfer['best_accuracy']*100:>13.2f}%")
    print(f"{'Random Init':<25} {'N/A':>12} "
          f"{random_transfer['final_accuracy']*100:>13.2f}% "
          f"{random_transfer['best_accuracy']*100:>13.2f}%")

    # Compute analysis metrics
    std_gain = std_transfer['final_accuracy'] - random_transfer['final_accuracy']
    dend_gain = dend_transfer['final_accuracy'] - random_transfer['final_accuracy']
    dend_vs_std = dend_transfer['final_accuracy'] - std_transfer['final_accuracy']
    source_gap = abs(std_test_acc - dend_test_acc)

    print("\n" + "-"*60)
    print("ANALYSIS")
    print("-"*60)

    print(f"\nTransfer gain over random init:")
    print(f"  Standard FF:  {std_gain*100:+.2f}%")
    print(f"  Dendritic FF: {dend_gain*100:+.2f}%")

    print(f"\nDendritic vs Standard FF:")
    print(f"  Source accuracy gap: {(dend_test_acc - std_test_acc)*100:+.2f}%")
    print(f"  Transfer accuracy gap: {dend_vs_std*100:+.2f}%")

    # Hypothesis test
    print("\n" + "-"*60)
    print("HYPOTHESIS VERIFICATION")
    print("-"*60)

    if dend_vs_std > 0:
        print("\n[CONFIRMED] Dendritic FF shows BETTER transfer learning!")
        print(f"  Advantage: {dend_vs_std*100:.2f}% higher accuracy")
        print("  Interpretation: Apical compartment learns more general representations")
    elif dend_vs_std > -0.02:  # Within 2% is comparable
        print("\n[COMPARABLE] Dendritic FF shows similar transfer performance")
        print(f"  Gap: {abs(dend_vs_std)*100:.2f}%")
    else:
        print("\n[NOT CONFIRMED] Standard FF shows better transfer")
        print(f"  Gap: {abs(dend_vs_std)*100:.2f}%")

    # Fairness check
    if source_gap < 0.05:
        print("\nFairness: Source accuracies are comparable (gap < 5%)")
    else:
        print(f"\nWarning: Source accuracy gap is {source_gap*100:.2f}%")

    results['analysis'] = {
        'std_transfer_gain': std_gain,
        'dend_transfer_gain': dend_gain,
        'dend_vs_std_transfer': dend_vs_std,
        'source_accuracy_gap': source_gap,
        'hypothesis_confirmed': dend_vs_std > 0,
        'fair_comparison': source_gap < 0.05
    }

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
    parser = argparse.ArgumentParser(description="Dendritic FF Experiment")
    parser.add_argument('--epochs', type=int, default=500,
                        help='FF epochs per layer (default: 500)')
    parser.add_argument('--transfer-epochs', type=int, default=100,
                        help='Transfer head training epochs')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Apical goodness weight (default: 0.3)')
    parser.add_argument('--radius', type=int, default=5,
                        help='Co-activity radius (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (200 epochs/layer)')
    parser.add_argument('--output', type=str, default='./results/dendritic_ff_results.json',
                        help='Output path')
    args = parser.parse_args()

    # Set epochs based on mode
    epochs = 200 if args.quick else args.epochs

    # Run experiment
    results = run_dendritic_ff_experiment(
        epochs_per_layer=epochs,
        transfer_epochs=args.transfer_epochs,
        alpha=args.alpha,
        coactivity_radius=args.radius,
        seed=args.seed
    )

    # Save results
    save_results(results, args.output)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
