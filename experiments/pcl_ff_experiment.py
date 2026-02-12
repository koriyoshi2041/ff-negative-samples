#!/usr/bin/env python3
"""
PCL-FF (Predictive Coding Light Forward-Forward) Experiment
============================================================

This experiment tests the hypothesis that PCL-FF produces more abstract,
generalizable representations that transfer better to new tasks.

Experiment Design:
1. Train both Standard FF and PCL-FF on MNIST (500 epochs/layer)
2. Analyze learned representations:
   - Sparsity: PCL-FF should be sparser (redundancy removed)
   - Information content: PCL-FF should retain essential structure
   - Separation: How well representations cluster by class
3. Transfer to Fashion-MNIST:
   - Freeze feature layers, train linear classifier
   - Compare transfer performance
4. Compare with Standard FF

Key Metrics:
- MNIST accuracy (source task)
- Fashion-MNIST accuracy (transfer task)
- Representation sparsity
- Reconstruction quality
- Feature discriminability (linear probe accuracy)

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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pcl_ff import (
    PCLFFNetwork, StandardFFNetwork, overlay_y_on_x, get_device
)


# ============================================================
# Data Loading
# ============================================================

def get_mnist_data(device: torch.device) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                                    Tuple[torch.Tensor, torch.Tensor]]:
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


def get_fashion_mnist_data(device: torch.device) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                                           Tuple[torch.Tensor, torch.Tensor]]:
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
# Linear Probe for Transfer Learning
# ============================================================

class LinearProbe(nn.Module):
    """Linear classification head for measuring feature quality."""

    def __init__(self, in_features: int, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_probe(model: nn.Module,
                       x_train: torch.Tensor, y_train: torch.Tensor,
                       x_test: torch.Tensor, y_test: torch.Tensor,
                       feature_dim: int,
                       epochs: int = 100,
                       batch_size: int = 256,
                       lr: float = 0.01,
                       verbose: bool = True) -> Dict:
    """
    Train a linear probe on frozen features.

    This measures how linearly separable the learned features are.
    Better features = higher linear probe accuracy.
    """
    device = x_train.device

    # Extract features (frozen)
    with torch.no_grad():
        train_features = model.get_features(x_train)
        test_features = model.get_features(x_test)

    # Create linear probe
    probe = LinearProbe(feature_dim, num_classes=10).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'test_acc': [], 'loss': []}

    for epoch in range(epochs):
        probe.train()
        indices = torch.randperm(len(train_features))
        epoch_losses = []

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = train_features[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            outputs = probe(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Measure performance
        probe.eval()
        with torch.no_grad():
            train_preds = probe(train_features).argmax(dim=1)
            test_preds = probe(test_features).argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()
            test_acc = (test_preds == y_test).float().mean().item()

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['loss'].append(np.mean(epoch_losses))

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"  Probe Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

    return {
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'best_test_acc': max(history['test_acc']),
        'history': history
    }


# ============================================================
# Representation Analysis
# ============================================================

def analyze_representations(model: nn.Module,
                            x: torch.Tensor, y: torch.Tensor,
                            model_type: str = 'pcl_ff') -> Dict:
    """
    Comprehensive analysis of learned representations.

    Metrics:
    - Sparsity: Fraction of near-zero activations
    - Activation statistics: Mean, std, max
    - Dead neurons: Fraction of always-inactive neurons
    - Inter-class distance: How separated are different classes
    - Intra-class variance: How tight are same-class clusters
    """
    device = x.device

    with torch.no_grad():
        features = model.get_features(x)

        # Basic statistics
        sparsity = (features.abs() < 0.1).float().mean().item()
        activation_mean = features.mean().item()
        activation_std = features.std().item()
        activation_max = features.abs().max().item()

        # Dead neurons
        max_per_neuron = features.abs().max(dim=0)[0]
        dead_neurons = (max_per_neuron < 0.01).float().mean().item()

        # Class separation analysis
        class_means = []
        class_vars = []
        for c in range(10):
            mask = y == c
            if mask.sum() > 0:
                class_features = features[mask]
                class_means.append(class_features.mean(dim=0))
                class_vars.append(class_features.var(dim=0).mean().item())

        # Inter-class distance (average pairwise distance between class centers)
        inter_class_distances = []
        for i in range(len(class_means)):
            for j in range(i + 1, len(class_means)):
                dist = (class_means[i] - class_means[j]).pow(2).sum().sqrt().item()
                inter_class_distances.append(dist)

        avg_inter_class_dist = np.mean(inter_class_distances) if inter_class_distances else 0
        avg_intra_class_var = np.mean(class_vars) if class_vars else 0

        # Separation ratio (higher is better)
        separation_ratio = avg_inter_class_dist / (avg_intra_class_var + 1e-8)

    return {
        'sparsity': sparsity,
        'activation_mean': activation_mean,
        'activation_std': activation_std,
        'activation_max': activation_max,
        'dead_neurons': dead_neurons,
        'avg_inter_class_distance': avg_inter_class_dist,
        'avg_intra_class_variance': avg_intra_class_var,
        'separation_ratio': separation_ratio
    }


def compute_feature_similarity(features1: torch.Tensor,
                               features2: torch.Tensor) -> float:
    """Compute cosine similarity between two feature sets."""
    with torch.no_grad():
        # Normalize features
        f1_norm = F.normalize(features1, p=2, dim=1)
        f2_norm = F.normalize(features2, p=2, dim=1)

        # Compute average cosine similarity
        similarity = (f1_norm * f2_norm).sum(dim=1).mean().item()

    return similarity


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(epochs_per_layer: int = 500,
                   batch_size: int = 50000,
                   seed: int = 42,
                   verbose: bool = True) -> Dict:
    """
    Run the full PCL-FF experiment.

    Steps:
    1. Train Standard FF on MNIST
    2. Train PCL-FF on MNIST
    3. Analyze representations
    4. Transfer to Fashion-MNIST
    5. Compare results
    """
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    results = {
        'config': {
            'epochs_per_layer': epochs_per_layer,
            'batch_size': batch_size,
            'seed': seed,
            'device': str(device),
            'architecture': [784, 500, 500],
            'threshold': 2.0,
            'lr': 0.03,
            'pcl_alpha': 0.5,
            'pcl_sparsity_weight': 0.1,
            'timestamp': datetime.now().isoformat()
        },
        'standard_ff': {},
        'pcl_ff': {},
        'transfer_learning': {},
        'representation_analysis': {}
    }

    print("="*70)
    print("PCL-FF (Predictive Coding Light Forward-Forward) Experiment")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs per layer: {epochs_per_layer}")
    print(f"Batch size: {batch_size}")
    print(f"Architecture: [784, 500, 500]")
    print()

    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------
    print("Loading datasets...")
    (mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = get_mnist_data(device)
    (fmnist_train_x, fmnist_train_y), (fmnist_test_x, fmnist_test_y) = get_fashion_mnist_data(device)
    print(f"  MNIST: {len(mnist_train_x)} train, {len(mnist_test_x)} test")
    print(f"  Fashion-MNIST: {len(fmnist_train_x)} train, {len(fmnist_test_x)} test")
    print()

    # Subset to batch_size for training
    if batch_size < len(mnist_train_x):
        indices = torch.randperm(len(mnist_train_x))[:batch_size]
        train_x = mnist_train_x[indices]
        train_y = mnist_train_y[indices]
        print(f"Using {batch_size} samples for training")
    else:
        train_x = mnist_train_x
        train_y = mnist_train_y

    # Create positive and negative samples for MNIST
    x_pos = overlay_y_on_x(train_x, train_y)
    rnd = torch.randperm(len(train_x))
    x_neg = overlay_y_on_x(train_x, train_y[rnd])

    # --------------------------------------------------------
    # Train Standard FF
    # --------------------------------------------------------
    print("="*70)
    print("Training Standard Forward-Forward Network")
    print("="*70)

    standard_ff = StandardFFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    start_time = time.time()
    standard_ff.train_greedy(x_pos, x_neg, epochs_per_layer=epochs_per_layer, verbose=verbose)
    standard_ff_train_time = time.time() - start_time

    # Measure Standard FF accuracy
    standard_ff_train_acc = standard_ff.get_accuracy(mnist_train_x, mnist_train_y)
    standard_ff_test_acc = standard_ff.get_accuracy(mnist_test_x, mnist_test_y)

    results['standard_ff'] = {
        'mnist_train_acc': standard_ff_train_acc,
        'mnist_test_acc': standard_ff_test_acc,
        'train_time': standard_ff_train_time
    }

    print(f"\nStandard FF Results:")
    print(f"  MNIST Train Accuracy: {standard_ff_train_acc*100:.2f}%")
    print(f"  MNIST Test Accuracy:  {standard_ff_test_acc*100:.2f}%")
    print(f"  Training Time: {standard_ff_train_time:.1f}s")

    # --------------------------------------------------------
    # Train PCL-FF
    # --------------------------------------------------------
    print("\n" + "="*70)
    print("Training PCL-FF (Predictive Coding Light Forward-Forward)")
    print("="*70)

    pcl_ff = PCLFFNetwork(
        [784, 500, 500],
        threshold=2.0,
        lr=0.03,
        alpha=0.5,
        sparsity_weight=0.1,
        surprise_scale=1.0
    ).to(device)

    start_time = time.time()
    pcl_ff.train_greedy(x_pos, x_neg, epochs_per_layer=epochs_per_layer, verbose=verbose)
    pcl_ff_train_time = time.time() - start_time

    # Measure PCL-FF accuracy
    pcl_ff_train_acc = pcl_ff.get_accuracy(mnist_train_x, mnist_train_y)
    pcl_ff_test_acc = pcl_ff.get_accuracy(mnist_test_x, mnist_test_y)

    results['pcl_ff'] = {
        'mnist_train_acc': pcl_ff_train_acc,
        'mnist_test_acc': pcl_ff_test_acc,
        'train_time': pcl_ff_train_time,
        'training_history': pcl_ff.get_training_history()
    }

    print(f"\nPCL-FF Results:")
    print(f"  MNIST Train Accuracy: {pcl_ff_train_acc*100:.2f}%")
    print(f"  MNIST Test Accuracy:  {pcl_ff_test_acc*100:.2f}%")
    print(f"  Training Time: {pcl_ff_train_time:.1f}s")

    # --------------------------------------------------------
    # Representation Analysis
    # --------------------------------------------------------
    print("\n" + "="*70)
    print("Analyzing Learned Representations")
    print("="*70)

    # Analyze Standard FF representations
    standard_ff_analysis = analyze_representations(
        standard_ff, mnist_test_x, mnist_test_y, 'standard_ff'
    )
    results['representation_analysis']['standard_ff'] = standard_ff_analysis

    print("\nStandard FF Representations:")
    print(f"  Sparsity: {standard_ff_analysis['sparsity']*100:.1f}%")
    print(f"  Activation Mean: {standard_ff_analysis['activation_mean']:.4f}")
    print(f"  Activation Std:  {standard_ff_analysis['activation_std']:.4f}")
    print(f"  Dead Neurons:    {standard_ff_analysis['dead_neurons']*100:.1f}%")
    print(f"  Inter-class Distance: {standard_ff_analysis['avg_inter_class_distance']:.4f}")
    print(f"  Intra-class Variance: {standard_ff_analysis['avg_intra_class_variance']:.4f}")
    print(f"  Separation Ratio:     {standard_ff_analysis['separation_ratio']:.4f}")

    # Analyze PCL-FF representations
    pcl_ff_analysis = analyze_representations(
        pcl_ff, mnist_test_x, mnist_test_y, 'pcl_ff'
    )
    results['representation_analysis']['pcl_ff'] = pcl_ff_analysis

    print("\nPCL-FF Representations:")
    print(f"  Sparsity: {pcl_ff_analysis['sparsity']*100:.1f}%")
    print(f"  Activation Mean: {pcl_ff_analysis['activation_mean']:.4f}")
    print(f"  Activation Std:  {pcl_ff_analysis['activation_std']:.4f}")
    print(f"  Dead Neurons:    {pcl_ff_analysis['dead_neurons']*100:.1f}%")
    print(f"  Inter-class Distance: {pcl_ff_analysis['avg_inter_class_distance']:.4f}")
    print(f"  Intra-class Variance: {pcl_ff_analysis['avg_intra_class_variance']:.4f}")
    print(f"  Separation Ratio:     {pcl_ff_analysis['separation_ratio']:.4f}")

    # Also get PCL-FF specific analysis
    if hasattr(pcl_ff, 'analyze_representations'):
        pcl_specific = pcl_ff.analyze_representations(mnist_test_x, mnist_test_y)
        results['representation_analysis']['pcl_ff_detailed'] = pcl_specific
        print("\nPCL-FF Layer-wise Analysis:")
        print(f"  Layer Sparsity: {pcl_specific['layer_sparsity']}")
        print(f"  Reconstruction Errors: {pcl_specific['reconstruction_errors']}")

    # --------------------------------------------------------
    # Transfer Learning to Fashion-MNIST
    # --------------------------------------------------------
    print("\n" + "="*70)
    print("Transfer Learning to Fashion-MNIST")
    print("="*70)

    # Linear probe on Standard FF features
    print("\nTraining linear probe on Standard FF features...")
    standard_ff_probe = train_linear_probe(
        standard_ff,
        fmnist_train_x, fmnist_train_y,
        fmnist_test_x, fmnist_test_y,
        feature_dim=500,
        epochs=100,
        verbose=verbose
    )
    results['transfer_learning']['standard_ff'] = {
        'fashion_mnist_train_acc': standard_ff_probe['final_train_acc'],
        'fashion_mnist_test_acc': standard_ff_probe['final_test_acc'],
        'best_test_acc': standard_ff_probe['best_test_acc']
    }

    print(f"\nStandard FF Transfer Results:")
    print(f"  Fashion-MNIST Test Accuracy: {standard_ff_probe['final_test_acc']*100:.2f}%")
    print(f"  Best Test Accuracy: {standard_ff_probe['best_test_acc']*100:.2f}%")

    # Linear probe on PCL-FF features
    print("\nTraining linear probe on PCL-FF features...")
    pcl_ff_probe = train_linear_probe(
        pcl_ff,
        fmnist_train_x, fmnist_train_y,
        fmnist_test_x, fmnist_test_y,
        feature_dim=500,
        epochs=100,
        verbose=verbose
    )
    results['transfer_learning']['pcl_ff'] = {
        'fashion_mnist_train_acc': pcl_ff_probe['final_train_acc'],
        'fashion_mnist_test_acc': pcl_ff_probe['final_test_acc'],
        'best_test_acc': pcl_ff_probe['best_test_acc']
    }

    print(f"\nPCL-FF Transfer Results:")
    print(f"  Fashion-MNIST Test Accuracy: {pcl_ff_probe['final_test_acc']*100:.2f}%")
    print(f"  Best Test Accuracy: {pcl_ff_probe['best_test_acc']*100:.2f}%")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    print("\n--- Source Task (MNIST) ---")
    print(f"{'Model':<15} {'Train Acc':<12} {'Test Acc':<12} {'Time':<10}")
    print("-" * 50)
    print(f"{'Standard FF':<15} {standard_ff_train_acc*100:>10.2f}% {standard_ff_test_acc*100:>10.2f}% {standard_ff_train_time:>8.1f}s")
    print(f"{'PCL-FF':<15} {pcl_ff_train_acc*100:>10.2f}% {pcl_ff_test_acc*100:>10.2f}% {pcl_ff_train_time:>8.1f}s")

    print("\n--- Transfer Task (Fashion-MNIST) ---")
    print(f"{'Model':<15} {'Train Acc':<12} {'Test Acc':<12} {'Best Acc':<12}")
    print("-" * 55)
    print(f"{'Standard FF':<15} {standard_ff_probe['final_train_acc']*100:>10.2f}% "
          f"{standard_ff_probe['final_test_acc']*100:>10.2f}% "
          f"{standard_ff_probe['best_test_acc']*100:>10.2f}%")
    print(f"{'PCL-FF':<15} {pcl_ff_probe['final_train_acc']*100:>10.2f}% "
          f"{pcl_ff_probe['final_test_acc']*100:>10.2f}% "
          f"{pcl_ff_probe['best_test_acc']*100:>10.2f}%")

    print("\n--- Representation Quality ---")
    print(f"{'Model':<15} {'Sparsity':<12} {'Sep. Ratio':<12} {'Dead Neurons':<12}")
    print("-" * 55)
    print(f"{'Standard FF':<15} {standard_ff_analysis['sparsity']*100:>10.1f}% "
          f"{standard_ff_analysis['separation_ratio']:>10.2f} "
          f"{standard_ff_analysis['dead_neurons']*100:>10.1f}%")
    print(f"{'PCL-FF':<15} {pcl_ff_analysis['sparsity']*100:>10.1f}% "
          f"{pcl_ff_analysis['separation_ratio']:>10.2f} "
          f"{pcl_ff_analysis['dead_neurons']*100:>10.1f}%")

    # Calculate improvements
    transfer_improvement = (pcl_ff_probe['final_test_acc'] - standard_ff_probe['final_test_acc']) * 100
    sparsity_improvement = (pcl_ff_analysis['sparsity'] - standard_ff_analysis['sparsity']) * 100

    print("\n--- Key Insights ---")
    print(f"Transfer Accuracy Change: {transfer_improvement:+.2f}%")
    print(f"Sparsity Change: {sparsity_improvement:+.1f}%")

    if transfer_improvement > 0:
        print("\n[HYPOTHESIS SUPPORTED] PCL-FF shows better transfer learning performance.")
    else:
        print("\n[HYPOTHESIS NOT SUPPORTED] Standard FF performs better or equal on transfer.")

    if sparsity_improvement > 0:
        print("[EXPECTED] PCL-FF produces sparser representations.")
    else:
        print("[UNEXPECTED] Standard FF produces sparser or equal representations.")

    # Store summary in results
    results['summary'] = {
        'mnist_accuracy_diff': pcl_ff_test_acc - standard_ff_test_acc,
        'transfer_accuracy_diff': pcl_ff_probe['final_test_acc'] - standard_ff_probe['final_test_acc'],
        'sparsity_diff': pcl_ff_analysis['sparsity'] - standard_ff_analysis['sparsity'],
        'hypothesis_supported': transfer_improvement > 0
    }

    return results


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    # Convert numpy types to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    results_converted = convert_to_serializable(results)

    with open(output_path, 'w') as f:
        json.dump(results_converted, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCL-FF Experiment')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Epochs per layer (default: 500)')
    parser.add_argument('--batch-size', type=int, default=50000,
                        help='Batch size for training (default: 50000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results JSON')
    args = parser.parse_args()

    # Run experiment
    results = run_experiment(
        epochs_per_layer=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=not args.quiet
    )

    # Save results
    if args.output:
        output_path = args.output
    else:
        # Default path in results directory
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, 'pcl_ff_results.json')

    save_results(results, output_path)

    print("\nExperiment complete!")
