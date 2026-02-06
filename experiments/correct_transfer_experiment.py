#!/usr/bin/env python3
"""
Correct Transfer Learning Experiment for Forward-Forward
=========================================================

PROBLEM: Previous experiment only trained FF for 5 epochs with mini-batch,
achieving only 56.75% source accuracy. This is NOT a fair comparison with BP (97.73%).

SOLUTION: Use correct FF implementation with greedy layer-by-layer training
to achieve ~93% accuracy, then compare transfer learning fairly.

Key settings:
- FF: Greedy layer-by-layer training, full batch (50000), MEAN goodness
- BP: Standard training to ~98%
- Transfer: Freeze feature layers, train only classification head
- Fair comparison: Same transfer setup for all methods

Author: Clawd (for Parafee)
Date: 2026-02-06
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
from torch.utils.data import DataLoader, TensorDataset
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

def get_mnist_data(device: torch.device):
    """Load full MNIST dataset to device."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Load full dataset
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
# Forward-Forward Implementation (Correct)
# ============================================================

def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Embed label in first 10 pixels.
    CRITICAL: Use x.max() as the label value, not 1.0!
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


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
                    num_epochs: int = 1000, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """Forward-Forward Network with correct greedy training."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(FFLayer(dims[d], dims[d + 1], threshold, lr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 1000, verbose: bool = True):
        """Greedy layer-by-layer training."""
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\nTraining layer {i}...')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg, epochs_per_layer, verbose)

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels."""
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(num_classes):
            h = overlay_y_on_x(x, torch.full((batch_size,), label, device=x.device))

            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))

            total_goodness = sum(goodness)
            goodness_per_label.append(total_goodness.unsqueeze(1))

        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute accuracy."""
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features from last layer (for transfer learning)."""
        with torch.no_grad():
            # Note: For FF, we need to embed a neutral label or average over labels
            # Here we use label=0 to get consistent features
            batch_size = x.shape[0]
            h = overlay_y_on_x(x, torch.zeros(batch_size, dtype=torch.long, device=x.device))
            for layer in self.layers:
                h = layer(h)
            return h


# ============================================================
# BP Network
# ============================================================

class BPNetwork(nn.Module):
    """Standard backprop network."""

    def __init__(self, dims: List[int], num_classes: int = 10):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        # Final classification layer
        layers.append(nn.Linear(dims[-1], num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features from second-to-last layer."""
        with torch.no_grad():
            # Go through all layers except the last linear
            h = x
            for layer in list(self.network.children())[:-1]:
                h = layer(h)
            return h

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            preds = self.forward(x).argmax(dim=1)
            return (preds == y).float().mean().item()


def train_bp_network(model: BPNetwork, x_train: torch.Tensor, y_train: torch.Tensor,
                     x_test: torch.Tensor, y_test: torch.Tensor,
                     epochs: int = 50, batch_size: int = 128, lr: float = 0.001,
                     verbose: bool = True) -> Dict:
    """Train BP network."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'test_acc': [], 'loss': []}

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(x_train))
        epoch_losses = []

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Evaluate
        model.eval()
        train_acc = model.get_accuracy(x_train, y_train)
        test_acc = model.get_accuracy(x_test, y_test)

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['loss'].append(np.mean(epoch_losses))

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {np.mean(epoch_losses):.4f} | "
                  f"Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

    return history


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


def train_transfer_head(features_train: torch.Tensor, y_train: torch.Tensor,
                        features_test: torch.Tensor, y_test: torch.Tensor,
                        epochs: int = 100, batch_size: int = 256, lr: float = 0.01,
                        verbose: bool = True) -> Dict:
    """Train linear head on frozen features."""
    feature_dim = features_train.shape[1]
    num_classes = int(y_train.max().item()) + 1

    head = LinearHead(feature_dim, num_classes).to(features_train.device)
    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'test_acc': [], 'loss': []}

    for epoch in range(epochs):
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

        # Evaluate
        head.train(False)
        with torch.no_grad():
            train_preds = head(features_train).argmax(dim=1)
            test_preds = head(features_test).argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()
            test_acc = (test_preds == y_test).float().mean().item()

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['loss'].append(np.mean(epoch_losses))

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {np.mean(epoch_losses):.4f} | "
                  f"Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

    return {
        'final_accuracy': history['test_acc'][-1],
        'best_accuracy': max(history['test_acc']),
        'history': history
    }


# ============================================================
# Main Experiment
# ============================================================

def run_correct_transfer_experiment(
    epochs_per_layer: int = 200,  # FF: 200 for quick (~67%), 500 for better (~90%), 1000 for best (~93%)
    bp_epochs: int = 50,           # BP: 50 epochs is enough for ~98%
    transfer_epochs: int = 100,    # Transfer: train head for 100 epochs
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run correct transfer learning experiment.

    This addresses the previous unfair comparison by:
    1. Training FF properly with greedy layer-by-layer training
    2. Ensuring both FF and BP achieve reasonable source accuracy
    3. Using same transfer setup for all methods
    """
    print("="*70)
    print("CORRECT TRANSFER LEARNING EXPERIMENT")
    print("="*70)
    print(f"\nSettings:")
    print(f"  FF epochs per layer: {epochs_per_layer}")
    print(f"  BP epochs: {bp_epochs}")
    print(f"  Transfer epochs: {transfer_epochs}")
    print(f"  Seed: {seed}")

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"  Device: {device}")

    results = {
        'experiment': 'correct_transfer_learning',
        'config': {
            'epochs_per_layer': epochs_per_layer,
            'bp_epochs': bp_epochs,
            'transfer_epochs': transfer_epochs,
            'seed': seed,
            'device': str(device),
            'architecture': [784, 500, 500],
            'threshold': 2.0,
            'lr_ff': 0.03,
            'lr_bp': 0.001,
            'lr_transfer': 0.01
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
    # 1. Train FF (Correct Implementation)
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 1: Training Forward-Forward (Correct Implementation)")
    print("="*60)

    torch.manual_seed(seed)
    ff_model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Create positive/negative samples
    x_pos = overlay_y_on_x(mnist_train, mnist_train_y)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg = overlay_y_on_x(mnist_train, mnist_train_y[rnd])

    # Train greedy layer-by-layer
    ff_start = time.time()
    ff_model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    ff_train_time = time.time() - ff_start

    # Evaluate FF
    ff_train_acc = ff_model.get_accuracy(mnist_train, mnist_train_y)
    ff_test_acc = ff_model.get_accuracy(mnist_test, mnist_test_y)

    print(f"\nFF Source Results:")
    print(f"  Train accuracy: {ff_train_acc*100:.2f}%")
    print(f"  Test accuracy:  {ff_test_acc*100:.2f}%")
    print(f"  Training time:  {ff_train_time:.1f}s")

    results['ff'] = {
        'source_train_acc': ff_train_acc,
        'source_test_acc': ff_test_acc,
        'train_time': ff_train_time
    }

    # ============================================================
    # 2. Train BP
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 2: Training Backpropagation")
    print("="*60)

    torch.manual_seed(seed)
    bp_model = BPNetwork([784, 500, 500], num_classes=10).to(device)

    bp_start = time.time()
    bp_history = train_bp_network(bp_model, mnist_train, mnist_train_y,
                                   mnist_test, mnist_test_y,
                                   epochs=bp_epochs, verbose=verbose)
    bp_train_time = time.time() - bp_start

    bp_train_acc = bp_history['train_acc'][-1]
    bp_test_acc = bp_history['test_acc'][-1]

    print(f"\nBP Source Results:")
    print(f"  Train accuracy: {bp_train_acc*100:.2f}%")
    print(f"  Test accuracy:  {bp_test_acc*100:.2f}%")
    print(f"  Training time:  {bp_train_time:.1f}s")

    results['bp'] = {
        'source_train_acc': bp_train_acc,
        'source_test_acc': bp_test_acc,
        'train_time': bp_train_time
    }

    # ============================================================
    # 3. Extract Features for Transfer
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 3: Transfer Learning to Fashion-MNIST")
    print("="*60)

    print("\nExtracting features...")

    # FF features
    ff_features_train = ff_model.get_features(fmnist_train)
    ff_features_test = ff_model.get_features(fmnist_test)
    print(f"  FF features: {ff_features_train.shape}")

    # BP features
    bp_features_train = bp_model.get_features(fmnist_train)
    bp_features_test = bp_model.get_features(fmnist_test)
    print(f"  BP features: {bp_features_train.shape}")

    # Random features (baseline)
    torch.manual_seed(seed)
    random_model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    random_features_train = random_model.get_features(fmnist_train)
    random_features_test = random_model.get_features(fmnist_test)
    print(f"  Random features: {random_features_train.shape}")

    # ============================================================
    # 4. Train Transfer Heads
    # ============================================================
    print("\n" + "-"*60)
    print("Training FF transfer head...")
    ff_transfer = train_transfer_head(
        ff_features_train, fmnist_train_y,
        ff_features_test, fmnist_test_y,
        epochs=transfer_epochs, verbose=verbose
    )
    results['ff']['transfer_accuracy'] = ff_transfer['final_accuracy']
    results['ff']['transfer_best'] = ff_transfer['best_accuracy']

    print("\n" + "-"*60)
    print("Training BP transfer head...")
    bp_transfer = train_transfer_head(
        bp_features_train, fmnist_train_y,
        bp_features_test, fmnist_test_y,
        epochs=transfer_epochs, verbose=verbose
    )
    results['bp']['transfer_accuracy'] = bp_transfer['final_accuracy']
    results['bp']['transfer_best'] = bp_transfer['best_accuracy']

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
    # 5. Summary
    # ============================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Method':<20} {'Source Acc':>12} {'Transfer Acc':>14} {'Transfer Best':>14}")
    print("-" * 62)
    print(f"{'FF (Correct))':<20} {ff_test_acc*100:>11.2f}% {ff_transfer['final_accuracy']*100:>13.2f}% {ff_transfer['best_accuracy']*100:>13.2f}%")
    print(f"{'BP':<20} {bp_test_acc*100:>11.2f}% {bp_transfer['final_accuracy']*100:>13.2f}% {bp_transfer['best_accuracy']*100:>13.2f}%")
    print(f"{'Random Init':<20} {'N/A':>12} {random_transfer['final_accuracy']*100:>13.2f}% {random_transfer['best_accuracy']*100:>13.2f}%")

    # Analysis
    print("\n" + "-"*60)
    print("ANALYSIS")
    print("-"*60)

    ff_gain = ff_transfer['final_accuracy'] - random_transfer['final_accuracy']
    bp_gain = bp_transfer['final_accuracy'] - random_transfer['final_accuracy']

    print(f"\nTransfer gain over random init:")
    print(f"  FF: {ff_gain*100:+.2f}%")
    print(f"  BP: {bp_gain*100:+.2f}%")

    ff_vs_bp = ff_transfer['final_accuracy'] - bp_transfer['final_accuracy']
    print(f"\nFF vs BP transfer gap: {ff_vs_bp*100:.2f}%")

    # Fairness check
    print("\n" + "-"*60)
    print("FAIRNESS CHECK")
    print("-"*60)
    source_gap = abs(ff_test_acc - bp_test_acc)
    print(f"Source accuracy gap: {source_gap*100:.2f}%")
    if source_gap < 0.10:
        print("Source accuracies are comparable (gap < 10%)")
    else:
        print("WARNING: Source accuracies differ significantly!")
        print("Consider increasing FF epochs_per_layer for fairer comparison")

    results['analysis'] = {
        'ff_transfer_gain': ff_gain,
        'bp_transfer_gain': bp_gain,
        'ff_vs_bp_gap': ff_vs_bp,
        'source_accuracy_gap': source_gap,
        'fair_comparison': source_gap < 0.10
    }

    results['timestamp'] = datetime.now().isoformat()

    return results


def save_results(results: Dict, output_path: str):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy/torch types
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
    parser = argparse.ArgumentParser(description="Correct Transfer Learning Experiment")
    parser.add_argument('--ff-epochs', type=int, default=200,
                        help='FF epochs per layer (200=~67%%, 500=~90%%, 1000=~93%%)')
    parser.add_argument('--bp-epochs', type=int, default=50,
                        help='BP training epochs')
    parser.add_argument('--transfer-epochs', type=int, default=100,
                        help='Transfer head training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (FF: 100 epochs/layer)')
    parser.add_argument('--full', action='store_true',
                        help='Full experiment (FF: 500 epochs/layer)')
    parser.add_argument('--best', action='store_true',
                        help='Best quality (FF: 1000 epochs/layer)')
    args = parser.parse_args()

    # Set epochs based on mode
    ff_epochs = args.ff_epochs
    if args.quick:
        ff_epochs = 100
    elif args.full:
        ff_epochs = 500
    elif args.best:
        ff_epochs = 1000

    # Run experiment
    results = run_correct_transfer_experiment(
        epochs_per_layer=ff_epochs,
        bp_epochs=args.bp_epochs,
        transfer_epochs=args.transfer_epochs,
        seed=args.seed
    )

    # Save results
    output_path = './results/correct_transfer_results.json'
    save_results(results, output_path)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
