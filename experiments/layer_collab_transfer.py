#!/usr/bin/env python3
"""
Layer Collaboration Transfer Learning Experiment
================================================

Tests the impact of Layer Collaboration on transfer learning.

Based on findings:
- layer_collab_comprehensive.json: gamma=0.7, mode=all is best (91.56% on MNIST)
- Previous transfer analysis: Layer 0 only transfer works best

Experiment Design:
A. Standard FF -> Fashion-MNIST (baseline)
B. Layer Collab FF (gamma=0.7, mode=all) -> Fashion-MNIST
C. Standard FF (Layer 0 only) -> Fashion-MNIST
D. Layer Collab FF (Layer 0 only) -> Fashion-MNIST

Pretraining: 500 epochs/layer (to reach ~90% on MNIST)
Transfer: 50 epochs (frozen features + linear head)

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

def get_dataset(dataset_name: str, device: torch.device) -> Tuple:
    """Load dataset."""
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


def overlay_y_on_x(x: torch.Tensor, y) -> torch.Tensor:
    """Embed label in first 10 pixels using x.max()."""
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


# ============================================================
# Forward-Forward Layer
# ============================================================

class FFLayer(nn.Module):
    """Forward-Forward Layer with collaboration support."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with layer normalization."""
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Compute goodness = MEAN of squared activations."""
        return h.pow(2).mean(dim=1)

    def ff_loss(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor,
                gamma_pos: torch.Tensor = None, gamma_neg: torch.Tensor = None,
                gamma_scale: float = 0.0) -> torch.Tensor:
        """FF loss with optional layer collaboration."""
        if gamma_pos is None:
            gamma_pos = torch.zeros_like(pos_goodness)
        if gamma_neg is None:
            gamma_neg = torch.zeros_like(neg_goodness)

        # Scaled collaborative logits
        pos_logit = pos_goodness + gamma_scale * gamma_pos - self.threshold
        neg_logit = neg_goodness + gamma_scale * gamma_neg - self.threshold

        # Loss
        loss = torch.log(1 + torch.exp(torch.cat([
            -pos_logit,
            neg_logit
        ]))).mean()

        return loss


# ============================================================
# Forward-Forward Network
# ============================================================

class FFNetwork(nn.Module):
    """Forward-Forward Network with Layer Collaboration support."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.dims = dims
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(FFLayer(dims[d], dims[d + 1], threshold, lr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def compute_all_goodness(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute goodness for all layers (detached)."""
        goodness_list = []
        h = x
        for layer in self.layers:
            h = layer(h)
            g = layer.goodness(h)
            goodness_list.append(g.detach())
            h = h.detach()
        return goodness_list

    def compute_gamma(self, goodness_list: List[torch.Tensor],
                      current_layer: int, mode: str = 'all') -> torch.Tensor:
        """Compute gamma for layer collaboration."""
        gamma = torch.zeros_like(goodness_list[0])

        for i, g in enumerate(goodness_list):
            if mode == 'all' and i != current_layer:
                gamma = gamma + g
            elif mode == 'prev' and i < current_layer:
                gamma = gamma + g
            elif mode == 'next' and i > current_layer:
                gamma = gamma + g

        return gamma

    def train_standard(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                       epochs_per_layer: int = 500, verbose: bool = True):
        """Standard FF training (greedy, layer-by-layer)."""
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'    Training layer {i} (Standard FF)...')

            for epoch in range(epochs_per_layer):
                out_pos = layer(h_pos)
                out_neg = layer(h_neg)

                g_pos = layer.goodness(out_pos)
                g_neg = layer.goodness(out_neg)

                loss = layer.ff_loss(g_pos, g_neg)

                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()

                if verbose and (epoch + 1) % 100 == 0:
                    print(f"      Epoch {epoch+1}: loss={loss.item():.4f}")

            h_pos = layer(h_pos).detach()
            h_neg = layer(h_neg).detach()

    def train_collaborative(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                            epochs_per_layer: int = 500,
                            gamma_mode: str = 'all',
                            gamma_scale: float = 0.7,
                            verbose: bool = True):
        """Layer Collaboration FF training."""
        for layer_idx, layer in enumerate(self.layers):
            if verbose:
                print(f'    Training layer {layer_idx} (Collab mode={gamma_mode}, scale={gamma_scale})...')

            for epoch in range(epochs_per_layer):
                # Compute all goodness (detached)
                pos_goodness_all = self.compute_all_goodness(x_pos)
                neg_goodness_all = self.compute_all_goodness(x_neg)

                # Compute gamma
                gamma_pos = self.compute_gamma(pos_goodness_all, layer_idx, gamma_mode)
                gamma_neg = self.compute_gamma(neg_goodness_all, layer_idx, gamma_mode)

                # Forward to this layer's input
                h_pos = x_pos
                h_neg = x_neg
                for i in range(layer_idx):
                    h_pos = self.layers[i](h_pos).detach()
                    h_neg = self.layers[i](h_neg).detach()

                # Forward through current layer
                out_pos = layer(h_pos)
                out_neg = layer(h_neg)

                g_pos = layer.goodness(out_pos)
                g_neg = layer.goodness(out_neg)

                loss = layer.ff_loss(g_pos, g_neg, gamma_pos, gamma_neg, gamma_scale)

                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()

                if verbose and (epoch + 1) % 100 == 0:
                    print(f"      Epoch {epoch+1}: loss={loss.item():.4f}")

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels."""
        goodness_per_label = []

        for label in range(num_classes):
            h = overlay_y_on_x(x, torch.full((x.shape[0],), label, device=x.device))
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))
            goodness_per_label.append(sum(goodness).unsqueeze(1))

        return torch.cat(goodness_per_label, dim=1).argmax(dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute accuracy."""
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()

    def get_features(self, x: torch.Tensor, up_to_layer: int = None,
                     label: int = 0) -> torch.Tensor:
        """Get features with label embedding."""
        with torch.no_grad():
            h = overlay_y_on_x(x, torch.full((x.shape[0],), label, dtype=torch.long, device=x.device))

            layers_to_use = self.layers if up_to_layer is None else self.layers[:up_to_layer + 1]

            for layer in layers_to_use:
                h = layer(h)

            return h


# ============================================================
# Linear Head for Transfer Learning
# ============================================================

class LinearHead(nn.Module):
    """Linear classification head."""

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

def run_experiment(pretrain_epochs: int = 500,
                   transfer_epochs: int = 50,
                   seed: int = 42) -> Dict[str, Any]:
    """Run the Layer Collaboration transfer learning experiment."""

    print("="*70)
    print("LAYER COLLABORATION TRANSFER LEARNING EXPERIMENT")
    print("="*70)

    # Setup
    torch.manual_seed(seed)
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Pretrain epochs per layer: {pretrain_epochs}")
    print(f"Transfer epochs: {transfer_epochs}")
    print(f"Seed: {seed}")

    results = {
        'experiment': 'Layer Collaboration Transfer Learning',
        'config': {
            'pretrain_epochs': pretrain_epochs,
            'transfer_epochs': transfer_epochs,
            'seed': seed,
            'device': str(device),
            'architecture': [784, 500, 500],
            'best_collab_config': {
                'gamma_mode': 'all',
                'gamma_scale': 0.7
            }
        },
        'timestamp': datetime.now().isoformat()
    }

    # Load datasets
    print("\n" + "-"*60)
    print("Loading datasets...")
    (mnist_train, mnist_y_train), (mnist_test, mnist_y_test) = get_dataset('mnist', device)
    (fmnist_train, fmnist_y_train), (fmnist_test, fmnist_y_test) = get_dataset('fashion_mnist', device)

    print(f"  MNIST: {len(mnist_train)} train, {len(mnist_test)} test")
    print(f"  Fashion-MNIST: {len(fmnist_train)} train, {len(fmnist_test)} test")

    # Prepare pos/neg samples for MNIST
    x_pos = overlay_y_on_x(mnist_train, mnist_y_train)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg = overlay_y_on_x(mnist_train, mnist_y_train[rnd])

    # ================================================================
    # A. Standard FF -> Fashion-MNIST
    # ================================================================
    print("\n" + "="*60)
    print("A. Standard FF -> Fashion-MNIST")
    print("="*60)

    torch.manual_seed(seed)
    ff_standard = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    print("\n  Pretraining on MNIST...")
    start = time.time()
    ff_standard.train_standard(x_pos, x_neg, pretrain_epochs, verbose=True)
    pretrain_time_standard = time.time() - start

    source_acc_standard = ff_standard.get_accuracy(mnist_test, mnist_y_test)
    print(f"\n  MNIST Source Accuracy: {source_acc_standard*100:.2f}%")

    print("\n  Extracting features for Fashion-MNIST...")
    features_train_A = ff_standard.get_features(fmnist_train, label=0)
    features_test_A = ff_standard.get_features(fmnist_test, label=0)

    print("  Training linear head...")
    transfer_A = train_linear_head(
        features_train_A, fmnist_y_train,
        features_test_A, fmnist_y_test,
        epochs=transfer_epochs
    )

    results['A_standard_ff'] = {
        'source_accuracy': source_acc_standard,
        'transfer_accuracy': transfer_A['best_test_acc'],
        'pretrain_time': pretrain_time_standard,
        'layers_transferred': 'all'
    }

    print(f"\n  Transfer Accuracy: {transfer_A['best_test_acc']*100:.2f}%")

    # ================================================================
    # B. Layer Collab FF (gamma=0.7, mode=all) -> Fashion-MNIST
    # ================================================================
    print("\n" + "="*60)
    print("B. Layer Collab FF (gamma=0.7, mode=all) -> Fashion-MNIST")
    print("="*60)

    torch.manual_seed(seed)
    ff_collab = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    print("\n  Pretraining on MNIST with Layer Collaboration...")
    start = time.time()
    ff_collab.train_collaborative(x_pos, x_neg, pretrain_epochs,
                                   gamma_mode='all', gamma_scale=0.7, verbose=True)
    pretrain_time_collab = time.time() - start

    source_acc_collab = ff_collab.get_accuracy(mnist_test, mnist_y_test)
    print(f"\n  MNIST Source Accuracy: {source_acc_collab*100:.2f}%")

    print("\n  Extracting features for Fashion-MNIST...")
    features_train_B = ff_collab.get_features(fmnist_train, label=0)
    features_test_B = ff_collab.get_features(fmnist_test, label=0)

    print("  Training linear head...")
    transfer_B = train_linear_head(
        features_train_B, fmnist_y_train,
        features_test_B, fmnist_y_test,
        epochs=transfer_epochs
    )

    results['B_layer_collab_ff'] = {
        'source_accuracy': source_acc_collab,
        'transfer_accuracy': transfer_B['best_test_acc'],
        'pretrain_time': pretrain_time_collab,
        'gamma_mode': 'all',
        'gamma_scale': 0.7,
        'layers_transferred': 'all'
    }

    print(f"\n  Transfer Accuracy: {transfer_B['best_test_acc']*100:.2f}%")

    # ================================================================
    # C. Standard FF (Layer 0 only) -> Fashion-MNIST
    # ================================================================
    print("\n" + "="*60)
    print("C. Standard FF (Layer 0 only) -> Fashion-MNIST")
    print("="*60)

    print("\n  Using features from layer 0 only (from model A)...")
    features_train_C = ff_standard.get_features(fmnist_train, up_to_layer=0, label=0)
    features_test_C = ff_standard.get_features(fmnist_test, up_to_layer=0, label=0)

    print("  Training linear head...")
    transfer_C = train_linear_head(
        features_train_C, fmnist_y_train,
        features_test_C, fmnist_y_test,
        epochs=transfer_epochs
    )

    results['C_standard_ff_layer0'] = {
        'source_accuracy': source_acc_standard,
        'transfer_accuracy': transfer_C['best_test_acc'],
        'layers_transferred': 'layer_0_only'
    }

    print(f"\n  Transfer Accuracy: {transfer_C['best_test_acc']*100:.2f}%")

    # ================================================================
    # D. Layer Collab FF (Layer 0 only) -> Fashion-MNIST
    # ================================================================
    print("\n" + "="*60)
    print("D. Layer Collab FF (Layer 0 only) -> Fashion-MNIST")
    print("="*60)

    print("\n  Using features from layer 0 only (from model B)...")
    features_train_D = ff_collab.get_features(fmnist_train, up_to_layer=0, label=0)
    features_test_D = ff_collab.get_features(fmnist_test, up_to_layer=0, label=0)

    print("  Training linear head...")
    transfer_D = train_linear_head(
        features_train_D, fmnist_y_train,
        features_test_D, fmnist_y_test,
        epochs=transfer_epochs
    )

    results['D_layer_collab_ff_layer0'] = {
        'source_accuracy': source_acc_collab,
        'transfer_accuracy': transfer_D['best_test_acc'],
        'gamma_mode': 'all',
        'gamma_scale': 0.7,
        'layers_transferred': 'layer_0_only'
    }

    print(f"\n  Transfer Accuracy: {transfer_D['best_test_acc']*100:.2f}%")

    # ================================================================
    # Random Baseline
    # ================================================================
    print("\n" + "="*60)
    print("Random Baseline (untrained network)")
    print("="*60)

    torch.manual_seed(seed)
    ff_random = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # All layers
    features_train_random = ff_random.get_features(fmnist_train, label=0)
    features_test_random = ff_random.get_features(fmnist_test, label=0)

    transfer_random = train_linear_head(
        features_train_random, fmnist_y_train,
        features_test_random, fmnist_y_test,
        epochs=transfer_epochs
    )

    # Layer 0 only
    features_train_random_l0 = ff_random.get_features(fmnist_train, up_to_layer=0, label=0)
    features_test_random_l0 = ff_random.get_features(fmnist_test, up_to_layer=0, label=0)

    transfer_random_l0 = train_linear_head(
        features_train_random_l0, fmnist_y_train,
        features_test_random_l0, fmnist_y_test,
        epochs=transfer_epochs
    )

    results['random_baseline'] = {
        'all_layers': transfer_random['best_test_acc'],
        'layer_0_only': transfer_random_l0['best_test_acc']
    }

    print(f"\n  Random (All Layers): {transfer_random['best_test_acc']*100:.2f}%")
    print(f"  Random (Layer 0):    {transfer_random_l0['best_test_acc']*100:.2f}%")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Configuration':<45} {'Source':>10} {'Transfer':>10}")
    print("-"*65)

    print(f"{'A. Standard FF (All Layers)':<45} {source_acc_standard*100:>9.2f}% {transfer_A['best_test_acc']*100:>9.2f}%")
    print(f"{'B. Layer Collab FF (All Layers)':<45} {source_acc_collab*100:>9.2f}% {transfer_B['best_test_acc']*100:>9.2f}%")
    print(f"{'C. Standard FF (Layer 0 Only)':<45} {source_acc_standard*100:>9.2f}% {transfer_C['best_test_acc']*100:>9.2f}%")
    print(f"{'D. Layer Collab FF (Layer 0 Only)':<45} {source_acc_collab*100:>9.2f}% {transfer_D['best_test_acc']*100:>9.2f}%")
    print("-"*65)
    print(f"{'Random Init (All Layers)':<45} {'N/A':>10} {transfer_random['best_test_acc']*100:>9.2f}%")
    print(f"{'Random Init (Layer 0 Only)':<45} {'N/A':>10} {transfer_random_l0['best_test_acc']*100:>9.2f}%")

    # Find best configuration
    all_transfer_accs = {
        'A': transfer_A['best_test_acc'],
        'B': transfer_B['best_test_acc'],
        'C': transfer_C['best_test_acc'],
        'D': transfer_D['best_test_acc']
    }
    best_config = max(all_transfer_accs, key=all_transfer_accs.get)

    # Analysis
    results['analysis'] = {
        'layer_collab_improvement_all_layers': transfer_B['best_test_acc'] - transfer_A['best_test_acc'],
        'layer_collab_improvement_layer0': transfer_D['best_test_acc'] - transfer_C['best_test_acc'],
        'layer0_vs_all_standard': transfer_C['best_test_acc'] - transfer_A['best_test_acc'],
        'layer0_vs_all_collab': transfer_D['best_test_acc'] - transfer_B['best_test_acc'],
        'best_configuration': best_config
    }

    print("\n" + "-"*60)
    print("KEY FINDINGS:")
    print("-"*60)

    collab_improvement_all = results['analysis']['layer_collab_improvement_all_layers'] * 100
    collab_improvement_l0 = results['analysis']['layer_collab_improvement_layer0'] * 100

    if collab_improvement_all > 0:
        print(f"1. Layer Collab IMPROVES transfer (all layers) by {collab_improvement_all:.2f}%")
    else:
        print(f"1. Layer Collab does NOT improve transfer (all layers): {collab_improvement_all:.2f}%")

    if collab_improvement_l0 > 0:
        print(f"2. Layer Collab IMPROVES transfer (layer 0) by {collab_improvement_l0:.2f}%")
    else:
        print(f"2. Layer Collab does NOT improve transfer (layer 0): {collab_improvement_l0:.2f}%")

    l0_improvement_standard = results['analysis']['layer0_vs_all_standard'] * 100
    l0_improvement_collab = results['analysis']['layer0_vs_all_collab'] * 100

    print(f"3. Layer 0 only vs All layers (Standard): {'+' if l0_improvement_standard > 0 else ''}{l0_improvement_standard:.2f}%")
    print(f"4. Layer 0 only vs All layers (Collab):   {'+' if l0_improvement_collab > 0 else ''}{l0_improvement_collab:.2f}%")

    print(f"5. Best configuration: {results['analysis']['best_configuration']}")

    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (float, int)):
            return obj
        elif hasattr(obj, 'item'):  # tensor
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Layer Collaboration Transfer Learning Experiment')
    parser.add_argument('--pretrain-epochs', type=int, default=500,
                        help='Epochs per layer for pretraining (default: 500)')
    parser.add_argument('--transfer-epochs', type=int, default=50,
                        help='Epochs for transfer learning (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (100 pretrain epochs)')
    args = parser.parse_args()

    pretrain_epochs = 100 if args.quick else args.pretrain_epochs

    results = run_experiment(
        pretrain_epochs=pretrain_epochs,
        transfer_epochs=args.transfer_epochs,
        seed=args.seed
    )

    output_path = str(Path(__file__).parent.parent / 'results' / 'layer_collab_transfer.json')
    save_results(results, output_path)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
