#!/usr/bin/env python3
"""
Three-Factor Forward-Forward Experiment
========================================

Tests the neuroscience-inspired Three-Factor Learning extension of FF.

Three-Factor Learning Rule:
    dw = f(pre) * f(post) * M(t)

Where M(t) is a global modulatory signal (like dopamine in the brain).

Experiment Design:
1. Train on MNIST (500 epochs/layer)
2. Transfer to Fashion-MNIST
3. Compare three modulation signal types:
   - None (baseline standard FF)
   - Top-down feedback (from output layer)
   - Reward prediction error (TD-like)
   - Layer agreement (inter-layer correlation)

Hypothesis: Global modulation signals help establish inter-layer connections,
improving transfer learning performance.

Author: Clawd (for Parafee)
Date: 2026-02-09
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.three_factor_ff import (
    ThreeFactorFFNetwork,
    ModulationType,
    create_three_factor_network,
    overlay_y_on_x,
    get_device
)


# ============================================================
# Data Loading
# ============================================================

def get_dataset(dataset_name: str, device: torch.device) -> Tuple:
    """Load dataset with full batch."""
    if dataset_name == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        dataset_class = datasets.MNIST
    elif dataset_name == 'fashion_mnist':
        mean, std = (0.2860,), (0.3530,)
        dataset_class = datasets.FashionMNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = dataset_class('./data', train=True, download=True, transform=transform)
    test_dataset = dataset_class('./data', train=False, download=True, transform=transform)

    # Full batch loading
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    return (x_train.to(device), y_train.to(device)), (x_test.to(device), y_test.to(device))


# ============================================================
# Linear Head for Transfer Learning
# ============================================================

class LinearHead(nn.Module):
    """Linear classification head for transfer learning."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_head(features_train: torch.Tensor, y_train: torch.Tensor,
                      features_test: torch.Tensor, y_test: torch.Tensor,
                      epochs: int = 50, batch_size: int = 256,
                      lr: float = 0.01) -> Dict:
    """Train linear head on frozen features."""
    device = features_train.device
    feature_dim = features_train.shape[1]
    num_classes = int(y_train.max().item()) + 1

    head = LinearHead(feature_dim, num_classes).to(device)
    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'test_acc': []}

    for epoch in range(epochs):
        head.train()
        indices = torch.randperm(len(features_train))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = features_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            outputs = head(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluate
        head.train(False)
        with torch.no_grad():
            train_preds = head(features_train).argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()

            test_preds = head(features_test).argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    return {
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'best_test_acc': max(history['test_acc']),
        'history': history
    }


# ============================================================
# Main Experiment
# ============================================================

def train_and_evaluate_model(modulation_type: str,
                             x_pos: torch.Tensor, x_neg: torch.Tensor,
                             x_train_raw: torch.Tensor,
                             y_train: torch.Tensor,
                             x_test: torch.Tensor, y_test: torch.Tensor,
                             epochs_per_layer: int,
                             seed: int,
                             device: torch.device,
                             verbose: bool = True) -> Dict:
    """Train a model with specific modulation type and evaluate."""

    torch.manual_seed(seed)

    # Create model
    model = create_three_factor_network(
        modulation_type=modulation_type,
        dims=[784, 500, 500],
        modulation_strength=0.5,
        threshold=2.0,
        lr=0.03
    ).to(device)

    # Train
    start_time = time.time()
    model.train_greedy(x_pos, x_neg, y_train, epochs_per_layer, verbose=verbose)
    train_time = time.time() - start_time

    # Evaluate (use raw data, get_accuracy handles label embedding internally)
    train_acc = model.get_accuracy(x_train_raw, y_train)
    test_acc = model.get_accuracy(x_test, y_test)

    return {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_time': train_time
    }


def run_experiment(pretrain_epochs: int = 500,
                   transfer_epochs: int = 50,
                   batch_size: int = 50000,
                   seed: int = 42,
                   verbose: bool = True) -> Dict[str, Any]:
    """
    Run the complete Three-Factor FF experiment.

    Phases:
    1. Train on MNIST with each modulation type
    2. Evaluate on MNIST (source accuracy)
    3. Transfer to Fashion-MNIST (transfer accuracy)
    4. Compare all modulation types

    Args:
        pretrain_epochs: Epochs per layer for pretraining
        transfer_epochs: Epochs for transfer learning
        batch_size: Training batch size (default 50000)
        seed: Random seed
        verbose: Print training progress
    """

    print("="*70)
    print("THREE-FACTOR FORWARD-FORWARD EXPERIMENT")
    print("="*70)
    print("\nNeuroscience-inspired modulation signals:")
    print("  - None: Standard FF baseline")
    print("  - Top-down: M(t) = softmax(final_layer)[correct_class]")
    print("  - Reward prediction: M(t) = actual_reward - expected_reward")
    print("  - Layer agreement: M(t) = correlation(current, next_layer)")

    # Setup
    torch.manual_seed(seed)
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Pretrain epochs per layer: {pretrain_epochs}")
    print(f"Transfer epochs: {transfer_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Seed: {seed}")

    results = {
        'experiment': 'Three-Factor Forward-Forward',
        'config': {
            'pretrain_epochs': pretrain_epochs,
            'transfer_epochs': transfer_epochs,
            'batch_size': batch_size,
            'seed': seed,
            'device': str(device),
            'architecture': [784, 500, 500],
            'modulation_strength': 0.5,
            'threshold': 2.0,
            'lr': 0.03
        },
        'timestamp': datetime.now().isoformat(),
        'modulation_types': {}
    }

    # Load datasets
    print("\n" + "-"*60)
    print("Loading datasets...")
    (mnist_train, mnist_y_train), (mnist_test, mnist_y_test) = get_dataset('mnist', device)
    (fmnist_train, fmnist_y_train), (fmnist_test, fmnist_y_test) = get_dataset('fashion_mnist', device)

    # Apply batch_size limit if needed
    if batch_size < len(mnist_train):
        print(f"  Using {batch_size} samples (out of {len(mnist_train)})")
        indices = torch.randperm(len(mnist_train))[:batch_size]
        mnist_train = mnist_train[indices]
        mnist_y_train = mnist_y_train[indices]

    print(f"  MNIST: {len(mnist_train)} train, {len(mnist_test)} test")
    print(f"  Fashion-MNIST: {len(fmnist_train)} train, {len(fmnist_test)} test")

    # Prepare positive and negative samples
    x_pos = overlay_y_on_x(mnist_train, mnist_y_train)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg = overlay_y_on_x(mnist_train, mnist_y_train[rnd])

    # Modulation types to test
    modulation_types = ["none", "top_down", "reward_prediction", "layer_agreement"]

    models = {}

    # ================================================================
    # Train models with each modulation type
    # ================================================================

    for mod_type in modulation_types:
        print("\n" + "="*60)
        print(f"Training: {mod_type.upper()} modulation")
        print("="*60)

        result = train_and_evaluate_model(
            modulation_type=mod_type,
            x_pos=x_pos,
            x_neg=x_neg,
            x_train_raw=mnist_train,
            y_train=mnist_y_train,
            x_test=mnist_test,
            y_test=mnist_y_test,
            epochs_per_layer=pretrain_epochs,
            seed=seed,
            device=device,
            verbose=verbose
        )

        models[mod_type] = result['model']

        print(f"\n  MNIST Results:")
        print(f"    Train Accuracy: {result['train_acc']*100:.2f}%")
        print(f"    Test Accuracy:  {result['test_acc']*100:.2f}%")
        print(f"    Training Time:  {result['train_time']:.1f}s")

        results['modulation_types'][mod_type] = {
            'source': {
                'train_acc': result['train_acc'],
                'test_acc': result['test_acc'],
                'train_time': result['train_time']
            }
        }

    # ================================================================
    # Transfer Learning to Fashion-MNIST
    # ================================================================

    print("\n" + "="*70)
    print("TRANSFER LEARNING: MNIST -> Fashion-MNIST")
    print("="*70)

    for mod_type in modulation_types:
        print(f"\n  Transfer for {mod_type}...")

        model = models[mod_type]

        # Extract features using all layers
        features_train = model.get_features(fmnist_train, label=0)
        features_test = model.get_features(fmnist_test, label=0)

        # Train linear head
        transfer_result = train_linear_head(
            features_train, fmnist_y_train,
            features_test, fmnist_y_test,
            epochs=transfer_epochs
        )

        print(f"    All Layers - Transfer Accuracy: {transfer_result['best_test_acc']*100:.2f}%")

        # Also try layer 0 only
        features_train_l0 = model.get_features(fmnist_train, up_to_layer=0, label=0)
        features_test_l0 = model.get_features(fmnist_test, up_to_layer=0, label=0)

        transfer_result_l0 = train_linear_head(
            features_train_l0, fmnist_y_train,
            features_test_l0, fmnist_y_test,
            epochs=transfer_epochs
        )

        print(f"    Layer 0 Only - Transfer Accuracy: {transfer_result_l0['best_test_acc']*100:.2f}%")

        results['modulation_types'][mod_type]['transfer'] = {
            'all_layers': {
                'best_test_acc': transfer_result['best_test_acc'],
                'final_test_acc': transfer_result['final_test_acc']
            },
            'layer_0_only': {
                'best_test_acc': transfer_result_l0['best_test_acc'],
                'final_test_acc': transfer_result_l0['final_test_acc']
            }
        }

    # ================================================================
    # Random Baseline
    # ================================================================

    print("\n" + "="*60)
    print("Random Baseline (untrained network)")
    print("="*60)

    torch.manual_seed(seed)
    random_model = create_three_factor_network(
        modulation_type="none",
        dims=[784, 500, 500]
    ).to(device)

    features_train_random = random_model.get_features(fmnist_train, label=0)
    features_test_random = random_model.get_features(fmnist_test, label=0)

    transfer_random = train_linear_head(
        features_train_random, fmnist_y_train,
        features_test_random, fmnist_y_test,
        epochs=transfer_epochs
    )

    features_train_random_l0 = random_model.get_features(fmnist_train, up_to_layer=0, label=0)
    features_test_random_l0 = random_model.get_features(fmnist_test, up_to_layer=0, label=0)

    transfer_random_l0 = train_linear_head(
        features_train_random_l0, fmnist_y_train,
        features_test_random_l0, fmnist_y_test,
        epochs=transfer_epochs
    )

    print(f"  All Layers: {transfer_random['best_test_acc']*100:.2f}%")
    print(f"  Layer 0:    {transfer_random_l0['best_test_acc']*100:.2f}%")

    results['random_baseline'] = {
        'all_layers': transfer_random['best_test_acc'],
        'layer_0_only': transfer_random_l0['best_test_acc']
    }

    # ================================================================
    # Summary and Analysis
    # ================================================================

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Modulation Type':<25} {'MNIST Test':>12} {'Transfer (All)':>15} {'Transfer (L0)':>14}")
    print("-"*70)

    baseline_transfer = None
    best_transfer = 0
    best_mod_type = None

    for mod_type in modulation_types:
        res = results['modulation_types'][mod_type]
        source_acc = res['source']['test_acc']
        transfer_all = res['transfer']['all_layers']['best_test_acc']
        transfer_l0 = res['transfer']['layer_0_only']['best_test_acc']

        print(f"{mod_type:<25} {source_acc*100:>11.2f}% {transfer_all*100:>14.2f}% {transfer_l0*100:>13.2f}%")

        if mod_type == "none":
            baseline_transfer = transfer_all
        if transfer_all > best_transfer:
            best_transfer = transfer_all
            best_mod_type = mod_type

    print("-"*70)
    print(f"{'Random Init (baseline)':<25} {'N/A':>12} {transfer_random['best_test_acc']*100:>14.2f}% {transfer_random_l0['best_test_acc']*100:>13.2f}%")

    # Analysis
    print("\n" + "-"*60)
    print("ANALYSIS")
    print("-"*60)

    improvements = {}
    for mod_type in modulation_types:
        if mod_type != "none":
            transfer_acc = results['modulation_types'][mod_type]['transfer']['all_layers']['best_test_acc']
            improvement = (transfer_acc - baseline_transfer) * 100
            improvements[mod_type] = improvement
            sign = "+" if improvement > 0 else ""
            print(f"  {mod_type} vs baseline: {sign}{improvement:.2f}%")

    results['analysis'] = {
        'baseline_transfer': baseline_transfer,
        'best_modulation_type': best_mod_type,
        'best_transfer_accuracy': best_transfer,
        'improvements_over_baseline': improvements
    }

    print(f"\n  Best modulation type: {best_mod_type}")
    print(f"  Best transfer accuracy: {best_transfer*100:.2f}%")

    # Test hypothesis
    print("\n" + "-"*60)
    print("HYPOTHESIS TEST")
    print("-"*60)
    print("H: Global modulation signals improve transfer learning")

    any_improvement = any(imp > 0 for imp in improvements.values())
    significant_improvement = any(imp > 1.0 for imp in improvements.values())

    if significant_improvement:
        print("RESULT: SUPPORTED - At least one modulation type significantly improves transfer")
        results['hypothesis_supported'] = True
    elif any_improvement:
        print("RESULT: PARTIALLY SUPPORTED - Small improvements observed")
        results['hypothesis_supported'] = 'partial'
    else:
        print("RESULT: NOT SUPPORTED - No modulation type improves transfer")
        results['hypothesis_supported'] = False

    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (float, int, str, bool)):
            return obj
        elif obj is None:
            return None
        elif hasattr(obj, 'item'):  # tensor
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return str(obj)

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Three-Factor Forward-Forward Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modulation Types:
  - none:              Standard FF (baseline)
  - top_down:          M(t) = softmax(output)[correct_class]
  - reward_prediction: M(t) = actual_reward - expected_reward
  - layer_agreement:   M(t) = correlation(current, next_layer)

Example:
  python three_factor_experiment.py --pretrain-epochs 500
        """
    )
    parser.add_argument('--pretrain-epochs', type=int, default=500,
                        help='Epochs per layer for pretraining (default: 500)')
    parser.add_argument('--transfer-epochs', type=int, default=50,
                        help='Epochs for transfer learning (default: 50)')
    parser.add_argument('--batch-size', type=int, default=50000,
                        help='Training batch size (default: 50000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (50 pretrain epochs)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    args = parser.parse_args()

    pretrain_epochs = 50 if args.quick else args.pretrain_epochs
    verbose = not args.quiet

    results = run_experiment(
        pretrain_epochs=pretrain_epochs,
        transfer_epochs=args.transfer_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=verbose
    )

    output_path = str(Path(__file__).parent.parent / 'results' / 'three_factor_ff_results.json')
    save_results(results, output_path)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
