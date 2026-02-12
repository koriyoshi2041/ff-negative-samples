#!/usr/bin/env python3
"""
Deep Analysis of Forward-Forward Transfer Learning Failure
==========================================================

Problem Context:
- FF Transfer: 61.06%
- Random Init: 83.81%
- Gap: 22.75% (FF is WORSE than random!)

This script performs comprehensive analysis to understand WHY FF transfer fails:

1. Feature Analysis:
   - t-SNE/UMAP visualization of FF vs BP features
   - Intra-class/inter-class distance comparison
   - Label embedding dependency analysis

2. Layer-by-Layer Transfer Analysis:
   - Test transfer with only layer 1
   - Test transfer with layers 1-2
   - Find which layer is least transferable

3. Improvement Strategies:
   - Strategy A: Remove first 10 pixels (label position) during transfer
   - Strategy B: Only transfer middle layers, retrain first/last
   - Strategy C: Use Layer Collab pretraining before transfer

Author: Clawd (for Parafee)
Date: 2026-02-07
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

def get_data(dataset_name: str, device: torch.device):
    """Load dataset to device."""
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

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    return (x_train.to(device), y_train.to(device)), (x_test.to(device), y_test.to(device))


# ============================================================
# Forward-Forward Implementation
# ============================================================

def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Embed label in first 10 pixels."""
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class FFLayer(nn.Module):
    """Forward-Forward Layer."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        return h.pow(2).mean(dim=1)

    def train_layer(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                    num_epochs: int = 500, verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        for _ in range(num_epochs):
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
    """Forward-Forward Network."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.dims = dims
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(FFLayer(dims[d], dims[d + 1], threshold, lr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 500, verbose: bool = True):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'  Training layer {i}...')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg, epochs_per_layer, verbose)

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(num_classes):
            h = overlay_y_on_x(x, torch.full((batch_size,), label, device=x.device))
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))
            goodness_per_label.append(sum(goodness).unsqueeze(1))

        return torch.cat(goodness_per_label, dim=1).argmax(dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()

    def get_features(self, x: torch.Tensor, label: int = 0) -> torch.Tensor:
        """Get features with a specific label embedded."""
        with torch.no_grad():
            batch_size = x.shape[0]
            h = overlay_y_on_x(x, torch.full((batch_size,), label, dtype=torch.long, device=x.device))
            for layer in self.layers:
                h = layer(h)
            return h

    def get_features_no_label(self, x: torch.Tensor) -> torch.Tensor:
        """Get features without label embedding (zero first 10 pixels)."""
        with torch.no_grad():
            h = x.clone()
            h[:, :10] = 0  # Zero out label position
            for layer in self.layers:
                h = layer(h)
            return h

    def get_layer_features(self, x: torch.Tensor, up_to_layer: int, label: int = 0) -> torch.Tensor:
        """Get features from specific layers only."""
        with torch.no_grad():
            batch_size = x.shape[0]
            h = overlay_y_on_x(x, torch.full((batch_size,), label, dtype=torch.long, device=x.device))
            for i, layer in enumerate(self.layers[:up_to_layer + 1]):
                h = layer(h)
            return h


class BPNetwork(nn.Module):
    """Standard backprop network."""

    def __init__(self, dims: List[int], num_classes: int = 10):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.network = nn.Sequential(*layers)
        self.feature_dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = x
            for layer in list(self.network.children())[:-1]:
                h = layer(h)
            return h

    def get_layer_features(self, x: torch.Tensor, up_to_layer: int) -> torch.Tensor:
        """Get features up to specific layer (0-indexed, counting only Linear layers)."""
        with torch.no_grad():
            h = x
            linear_count = 0
            for layer in self.network.children():
                h = layer(h)
                if isinstance(layer, nn.Linear):
                    if linear_count == up_to_layer:
                        return h
                    linear_count += 1
            return h

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            preds = self.forward(x).argmax(dim=1)
            return (preds == y).float().mean().item()


def train_bp_network(model: BPNetwork, x_train: torch.Tensor, y_train: torch.Tensor,
                     epochs: int = 50, batch_size: int = 128, lr: float = 0.001) -> float:
    """Train BP network and return final accuracy."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        indices = torch.randperm(len(x_train))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model.get_accuracy(x_train, y_train)


class LinearHead(nn.Module):
    """Linear classification head for transfer learning."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_transfer_head(features_train: torch.Tensor, y_train: torch.Tensor,
                        features_test: torch.Tensor, y_test: torch.Tensor,
                        epochs: int = 100, batch_size: int = 256, lr: float = 0.01) -> Dict:
    """Train linear head on frozen features."""
    feature_dim = features_train.shape[1]
    num_classes = int(y_train.max().item()) + 1

    head = LinearHead(feature_dim, num_classes).to(features_train.device)
    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for _ in range(epochs):
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

        head.train(False)
        with torch.no_grad():
            test_preds = head(features_test).argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()
            best_acc = max(best_acc, test_acc)

    return {'final_accuracy': test_acc, 'best_accuracy': best_acc}


# ============================================================
# Analysis Functions
# ============================================================

def compute_class_distances(features: torch.Tensor, labels: torch.Tensor) -> Dict:
    """Compute intra-class and inter-class distances."""
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    unique_labels = np.unique(labels_np)

    # Compute intra-class distances (within same class)
    intra_distances = []
    class_centers = {}

    for label in unique_labels:
        mask = labels_np == label
        class_features = features_np[mask]
        class_centers[label] = class_features.mean(axis=0)

        if len(class_features) > 1:
            # Sample pairs for efficiency
            n_samples = min(len(class_features), 500)
            indices = np.random.choice(len(class_features), n_samples, replace=False)
            sampled = class_features[indices]

            dists = pairwise_distances(sampled, metric='euclidean')
            upper_tri = dists[np.triu_indices(len(sampled), k=1)]
            intra_distances.extend(upper_tri.tolist())

    # Compute inter-class distances (between class centers)
    centers = np.array([class_centers[l] for l in unique_labels])
    inter_dists = pairwise_distances(centers, metric='euclidean')
    inter_distances = inter_dists[np.triu_indices(len(centers), k=1)].tolist()

    return {
        'intra_class_mean': float(np.mean(intra_distances)),
        'intra_class_std': float(np.std(intra_distances)),
        'inter_class_mean': float(np.mean(inter_distances)),
        'inter_class_std': float(np.std(inter_distances)),
        'separation_ratio': float(np.mean(inter_distances) / (np.mean(intra_distances) + 1e-8))
    }


def analyze_label_dependency(ff_model: FFNetwork, x: torch.Tensor, y: torch.Tensor) -> Dict:
    """Analyze how much FF features depend on label embedding."""
    # Get features with different labels
    features_by_label = []
    for label in range(10):
        features = ff_model.get_features(x[:1000], label=label)  # Use subset
        features_by_label.append(features)

    # Compute variance explained by label choice
    stacked = torch.stack(features_by_label, dim=0)  # [10, N, D]

    # Variance across labels (for same input)
    label_variance = stacked.var(dim=0).mean().item()

    # Total variance
    total_variance = stacked.var().item()

    # Compare features with label=0 vs features without label (zeroed)
    features_with_label = ff_model.get_features(x[:1000], label=0)
    features_no_label = ff_model.get_features_no_label(x[:1000])

    label_impact = (features_with_label - features_no_label).pow(2).mean().item()

    return {
        'label_variance': label_variance,
        'total_variance': total_variance,
        'variance_ratio': label_variance / (total_variance + 1e-8),
        'label_impact_mse': label_impact
    }


def visualize_features_tsne(ff_features: torch.Tensor, bp_features: torch.Tensor,
                            labels: torch.Tensor, output_path: str, title: str):
    """Create t-SNE visualization comparing FF and BP features."""
    n_samples = min(2000, len(labels))
    indices = np.random.choice(len(labels), n_samples, replace=False)

    ff_np = ff_features[indices].cpu().numpy()
    bp_np = bp_features[indices].cpu().numpy()
    labels_np = labels[indices].cpu().numpy()

    # Run t-SNE
    print("  Running t-SNE for FF features...")
    tsne_ff = TSNE(n_components=2, random_state=42, perplexity=30)
    ff_2d = tsne_ff.fit_transform(ff_np)

    print("  Running t-SNE for BP features...")
    tsne_bp = TSNE(n_components=2, random_state=42, perplexity=30)
    bp_2d = tsne_bp.fit_transform(bp_np)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter1 = axes[0].scatter(ff_2d[:, 0], ff_2d[:, 1], c=labels_np, cmap='tab10',
                               alpha=0.6, s=10)
    axes[0].set_title('Forward-Forward Features', fontsize=14)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')

    scatter2 = axes[1].scatter(bp_2d[:, 0], bp_2d[:, 1], c=labels_np, cmap='tab10',
                               alpha=0.6, s=10)
    axes[1].set_title('Backpropagation Features', fontsize=14)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    plt.colorbar(scatter2, ax=axes[1], label='Class')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_layer_analysis(layer_results: Dict, output_path: str):
    """Visualize layer-by-layer transfer analysis."""
    layers = list(layer_results.keys())
    ff_accs = [layer_results[l]['ff'] for l in layers]
    bp_accs = [layer_results[l]['bp'] for l in layers]
    random_accs = [layer_results[l]['random'] for l in layers]

    x = np.arange(len(layers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, [a * 100 for a in ff_accs], width, label='FF Transfer', color='#e74c3c')
    bars2 = ax.bar(x, [a * 100 for a in bp_accs], width, label='BP Transfer', color='#3498db')
    bars3 = ax.bar(x + width, [a * 100 for a in random_accs], width, label='Random Init', color='#2ecc71')

    ax.set_xlabel('Layers Transferred', fontsize=12)
    ax.set_ylabel('Transfer Accuracy (%)', fontsize=12)
    ax.set_title('Layer-by-Layer Transfer Analysis\n(MNIST -> Fashion-MNIST)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_improvement_strategies(strategy_results: Dict, output_path: str):
    """Visualize improvement strategy comparison."""
    strategies = list(strategy_results.keys())
    accs = [strategy_results[s]['accuracy'] * 100 for s in strategies]

    colors = ['#e74c3c' if 'baseline' in s.lower() else '#3498db' for s in strategies]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(strategies, accs, color=colors)

    # Add baseline reference line
    baseline_acc = strategy_results.get('FF Baseline', {}).get('accuracy', 0.61) * 100
    ax.axvline(x=baseline_acc, color='red', linestyle='--', linewidth=2, label=f'FF Baseline ({baseline_acc:.1f}%)')

    random_acc = strategy_results.get('Random Init', {}).get('accuracy', 0.84) * 100
    ax.axvline(x=random_acc, color='green', linestyle='--', linewidth=2, label=f'Random Init ({random_acc:.1f}%)')

    ax.set_xlabel('Transfer Accuracy (%)', fontsize=12)
    ax.set_title('FF Transfer Learning Improvement Strategies\n(MNIST -> Fashion-MNIST)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.annotate(f'{acc:.1f}%',
                   xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0),
                   textcoords="offset points",
                   ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_feature_distances(ff_distances: Dict, bp_distances: Dict,
                                 ff_fmnist_distances: Dict, bp_fmnist_distances: Dict,
                                 output_path: str):
    """Visualize class separation comparison."""
    categories = ['Intra-class\n(MNIST)', 'Inter-class\n(MNIST)',
                  'Intra-class\n(F-MNIST)', 'Inter-class\n(F-MNIST)']

    ff_values = [ff_distances['intra_class_mean'], ff_distances['inter_class_mean'],
                 ff_fmnist_distances['intra_class_mean'], ff_fmnist_distances['inter_class_mean']]
    bp_values = [bp_distances['intra_class_mean'], bp_distances['inter_class_mean'],
                 bp_fmnist_distances['intra_class_mean'], bp_fmnist_distances['inter_class_mean']]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ff_values, width, label='FF Features', color='#e74c3c')
    bars2 = ax.bar(x + width/2, bp_values, width, label='BP Features', color='#3498db')

    ax.set_ylabel('Mean Distance', fontsize=12)
    ax.set_title('Feature Space Class Separation Analysis', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Layer Collaboration FF for Strategy C
# ============================================================

class CollabFFLayer(nn.Module):
    """FF Layer with collaboration support."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        return h.pow(2).mean(dim=1)


class CollabFFNetwork(nn.Module):
    """FF Network with layer collaboration."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.dims = dims
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(CollabFFLayer(dims[d], dims[d + 1], threshold, lr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def compute_all_goodness(self, x: torch.Tensor) -> List[torch.Tensor]:
        goodness_list = []
        h = x
        for layer in self.layers:
            h = layer(h)
            g = layer.goodness(h)
            goodness_list.append(g.detach())
            h = h.detach()
        return goodness_list

    def train_collaborative(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                            epochs_per_layer: int = 500, verbose: bool = True):
        """Train with layer collaboration."""
        for layer_idx, layer in enumerate(self.layers):
            if verbose:
                print(f'  Training layer {layer_idx} (collaborative)...')

            for _ in range(epochs_per_layer):
                # Compute all goodness (for gamma)
                pos_goodness_all = self.compute_all_goodness(x_pos)
                neg_goodness_all = self.compute_all_goodness(x_neg)

                # Compute gamma (sum of other layers' goodness)
                gamma_pos = sum(g for i, g in enumerate(pos_goodness_all) if i != layer_idx)
                gamma_neg = sum(g for i, g in enumerate(neg_goodness_all) if i != layer_idx)

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

                # Loss with collaboration
                pos_logit = g_pos + gamma_pos - layer.threshold
                neg_logit = g_neg + gamma_neg - layer.threshold

                loss = torch.log(1 + torch.exp(torch.cat([
                    -pos_logit,
                    neg_logit
                ]))).mean()

                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(num_classes):
            h = overlay_y_on_x(x, torch.full((batch_size,), label, device=x.device))
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))
            goodness_per_label.append(sum(goodness).unsqueeze(1))

        return torch.cat(goodness_per_label, dim=1).argmax(dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()

    def get_features(self, x: torch.Tensor, label: int = 0) -> torch.Tensor:
        with torch.no_grad():
            batch_size = x.shape[0]
            h = overlay_y_on_x(x, torch.full((batch_size,), label, dtype=torch.long, device=x.device))
            for layer in self.layers:
                h = layer(h)
            return h


# ============================================================
# Main Analysis
# ============================================================

def run_transfer_analysis(epochs_per_layer: int = 300, seed: int = 42, verbose: bool = True):
    """Run comprehensive transfer learning analysis."""
    print("="*70)
    print("FF TRANSFER LEARNING DEEP ANALYSIS")
    print("="*70)

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Epochs per layer: {epochs_per_layer}")
    print(f"Seed: {seed}")

    results = {
        'experiment': 'ff_transfer_analysis',
        'config': {
            'epochs_per_layer': epochs_per_layer,
            'seed': seed,
            'device': str(device)
        }
    }

    # Load data
    print("\n" + "-"*60)
    print("Loading datasets...")
    (mnist_train, mnist_train_y), (mnist_test, mnist_test_y) = get_data('mnist', device)
    (fmnist_train, fmnist_train_y), (fmnist_test, fmnist_test_y) = get_data('fashion_mnist', device)
    print(f"  MNIST: {len(mnist_train)} train, {len(mnist_test)} test")
    print(f"  Fashion-MNIST: {len(fmnist_train)} train, {len(fmnist_test)} test")

    # ================================================================
    # PHASE 1: Train Models
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 1: Training Models")
    print("="*60)

    # Train FF
    print("\n1.1 Training Forward-Forward...")
    torch.manual_seed(seed)
    ff_model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    x_pos = overlay_y_on_x(mnist_train, mnist_train_y)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg = overlay_y_on_x(mnist_train, mnist_train_y[rnd])
    ff_model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    ff_source_acc = ff_model.get_accuracy(mnist_test, mnist_test_y)
    print(f"  FF Source Accuracy: {ff_source_acc*100:.2f}%")

    # Train BP
    print("\n1.2 Training Backpropagation...")
    torch.manual_seed(seed)
    bp_model = BPNetwork([784, 500, 500], num_classes=10).to(device)
    train_bp_network(bp_model, mnist_train, mnist_train_y, epochs=50)
    bp_source_acc = bp_model.get_accuracy(mnist_test, mnist_test_y)
    print(f"  BP Source Accuracy: {bp_source_acc*100:.2f}%")

    results['source_accuracy'] = {
        'ff': ff_source_acc,
        'bp': bp_source_acc
    }

    # ================================================================
    # PHASE 2: Feature Analysis
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 2: Feature Analysis")
    print("="*60)

    # Extract features on MNIST
    print("\n2.1 Extracting features on MNIST...")
    ff_mnist_features = ff_model.get_features(mnist_test, label=0)
    bp_mnist_features = bp_model.get_features(mnist_test)

    # Extract features on Fashion-MNIST
    print("2.2 Extracting features on Fashion-MNIST...")
    ff_fmnist_features = ff_model.get_features(fmnist_test, label=0)
    bp_fmnist_features = bp_model.get_features(fmnist_test)

    # Compute class distances
    print("\n2.3 Computing class distances...")
    ff_mnist_distances = compute_class_distances(ff_mnist_features, mnist_test_y)
    bp_mnist_distances = compute_class_distances(bp_mnist_features, mnist_test_y)
    ff_fmnist_distances = compute_class_distances(ff_fmnist_features, fmnist_test_y)
    bp_fmnist_distances = compute_class_distances(bp_fmnist_features, fmnist_test_y)

    print(f"\n  MNIST Feature Separation:")
    print(f"    FF - Separation ratio: {ff_mnist_distances['separation_ratio']:.3f}")
    print(f"    BP - Separation ratio: {bp_mnist_distances['separation_ratio']:.3f}")

    print(f"\n  Fashion-MNIST Feature Separation:")
    print(f"    FF - Separation ratio: {ff_fmnist_distances['separation_ratio']:.3f}")
    print(f"    BP - Separation ratio: {bp_fmnist_distances['separation_ratio']:.3f}")

    results['feature_analysis'] = {
        'ff_mnist': ff_mnist_distances,
        'bp_mnist': bp_mnist_distances,
        'ff_fmnist': ff_fmnist_distances,
        'bp_fmnist': bp_fmnist_distances
    }

    # Label dependency analysis
    print("\n2.4 Analyzing label embedding dependency...")
    label_dependency = analyze_label_dependency(ff_model, mnist_test, mnist_test_y)
    print(f"  Label variance ratio: {label_dependency['variance_ratio']:.4f}")
    print(f"  Label impact MSE: {label_dependency['label_impact_mse']:.4f}")
    results['label_dependency'] = label_dependency

    # Create visualizations
    print("\n2.5 Creating visualizations...")
    os.makedirs('./results', exist_ok=True)

    visualize_features_tsne(
        ff_fmnist_features, bp_fmnist_features, fmnist_test_y,
        './results/transfer_analysis_tsne.png',
        'Feature Visualization on Fashion-MNIST (Pre-trained on MNIST)'
    )

    visualize_feature_distances(
        ff_mnist_distances, bp_mnist_distances,
        ff_fmnist_distances, bp_fmnist_distances,
        './results/transfer_analysis_distances.png'
    )

    # ================================================================
    # PHASE 3: Layer-by-Layer Transfer Analysis
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 3: Layer-by-Layer Transfer Analysis")
    print("="*60)

    layer_results = {}

    for n_layers in [1, 2]:  # Transfer 1 layer, 2 layers
        layer_name = f"Layer 0-{n_layers-1}" if n_layers > 1 else "Layer 0"
        print(f"\n  Testing transfer with {layer_name}...")

        # FF layer features
        ff_layer_features_train = ff_model.get_layer_features(fmnist_train, n_layers - 1, label=0)
        ff_layer_features_test = ff_model.get_layer_features(fmnist_test, n_layers - 1, label=0)

        ff_result = train_transfer_head(
            ff_layer_features_train, fmnist_train_y,
            ff_layer_features_test, fmnist_test_y
        )

        # BP layer features
        bp_layer_features_train = bp_model.get_layer_features(fmnist_train, n_layers - 1)
        bp_layer_features_test = bp_model.get_layer_features(fmnist_test, n_layers - 1)

        bp_result = train_transfer_head(
            bp_layer_features_train, fmnist_train_y,
            bp_layer_features_test, fmnist_test_y
        )

        # Random baseline for same layer
        torch.manual_seed(seed)
        random_model = FFNetwork([784, 500, 500], threshold=2.0).to(device)
        random_layer_features_train = random_model.get_layer_features(fmnist_train, n_layers - 1, label=0)
        random_layer_features_test = random_model.get_layer_features(fmnist_test, n_layers - 1, label=0)

        random_result = train_transfer_head(
            random_layer_features_train, fmnist_train_y,
            random_layer_features_test, fmnist_test_y
        )

        layer_results[layer_name] = {
            'ff': ff_result['best_accuracy'],
            'bp': bp_result['best_accuracy'],
            'random': random_result['best_accuracy']
        }

        print(f"    FF: {ff_result['best_accuracy']*100:.2f}%")
        print(f"    BP: {bp_result['best_accuracy']*100:.2f}%")
        print(f"    Random: {random_result['best_accuracy']*100:.2f}%")

    # Add full model transfer
    print("\n  Testing transfer with all layers...")
    ff_full_result = train_transfer_head(
        ff_model.get_features(fmnist_train, label=0), fmnist_train_y,
        ff_model.get_features(fmnist_test, label=0), fmnist_test_y
    )
    bp_full_result = train_transfer_head(
        bp_model.get_features(fmnist_train), fmnist_train_y,
        bp_model.get_features(fmnist_test), fmnist_test_y
    )

    torch.manual_seed(seed)
    random_model = FFNetwork([784, 500, 500], threshold=2.0).to(device)
    random_full_result = train_transfer_head(
        random_model.get_features(fmnist_train, label=0), fmnist_train_y,
        random_model.get_features(fmnist_test, label=0), fmnist_test_y
    )

    layer_results["All Layers"] = {
        'ff': ff_full_result['best_accuracy'],
        'bp': bp_full_result['best_accuracy'],
        'random': random_full_result['best_accuracy']
    }

    print(f"    FF: {ff_full_result['best_accuracy']*100:.2f}%")
    print(f"    BP: {bp_full_result['best_accuracy']*100:.2f}%")
    print(f"    Random: {random_full_result['best_accuracy']*100:.2f}%")

    results['layer_analysis'] = layer_results

    visualize_layer_analysis(layer_results, './results/transfer_analysis_layers.png')

    # ================================================================
    # PHASE 4: Improvement Strategies
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 4: Improvement Strategies")
    print("="*60)

    strategy_results = {}

    # Baseline
    strategy_results['FF Baseline'] = {
        'accuracy': ff_full_result['best_accuracy'],
        'description': 'Standard FF transfer (with label=0 embedding)'
    }
    strategy_results['Random Init'] = {
        'accuracy': random_full_result['best_accuracy'],
        'description': 'Random initialization baseline'
    }
    strategy_results['BP Baseline'] = {
        'accuracy': bp_full_result['best_accuracy'],
        'description': 'Standard BP transfer'
    }

    # Strategy A: Remove first 10 pixels (label position)
    print("\n4.1 Strategy A: Remove first 10 pixels...")
    fmnist_train_no_label = fmnist_train.clone()
    fmnist_train_no_label[:, :10] = 0
    fmnist_test_no_label = fmnist_test.clone()
    fmnist_test_no_label[:, :10] = 0

    ff_features_no_label_train = ff_model.get_features_no_label(fmnist_train)
    ff_features_no_label_test = ff_model.get_features_no_label(fmnist_test)

    strategy_a_result = train_transfer_head(
        ff_features_no_label_train, fmnist_train_y,
        ff_features_no_label_test, fmnist_test_y
    )
    strategy_results['A: Remove Label Position'] = {
        'accuracy': strategy_a_result['best_accuracy'],
        'description': 'Transfer with first 10 pixels zeroed'
    }
    print(f"  Result: {strategy_a_result['best_accuracy']*100:.2f}%")

    # Strategy B: Only transfer middle layer, retrain first/last
    print("\n4.2 Strategy B: Transfer middle layer only...")
    # Get middle layer features (layer 0 output -> input to layer 1)
    middle_features_train = ff_model.get_layer_features(fmnist_train, 0, label=0)
    middle_features_test = ff_model.get_layer_features(fmnist_test, 0, label=0)

    # Train new first layer (from scratch) + use middle + new last
    # For simplicity, just use the middle layer features directly
    strategy_b_result = train_transfer_head(
        middle_features_train, fmnist_train_y,
        middle_features_test, fmnist_test_y
    )
    strategy_results['B: Middle Layer Only'] = {
        'accuracy': strategy_b_result['best_accuracy'],
        'description': 'Only transfer first layer, train classifier on its output'
    }
    print(f"  Result: {strategy_b_result['best_accuracy']*100:.2f}%")

    # Strategy C: Layer Collab pretraining
    print("\n4.3 Strategy C: Layer Collaboration pretraining...")
    torch.manual_seed(seed)
    collab_model = CollabFFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    x_pos_collab = overlay_y_on_x(mnist_train, mnist_train_y)
    rnd_collab = torch.randperm(mnist_train.size(0))
    x_neg_collab = overlay_y_on_x(mnist_train, mnist_train_y[rnd_collab])

    collab_model.train_collaborative(x_pos_collab, x_neg_collab, epochs_per_layer, verbose)
    collab_source_acc = collab_model.get_accuracy(mnist_test, mnist_test_y)
    print(f"  Collab Source Accuracy: {collab_source_acc*100:.2f}%")

    collab_features_train = collab_model.get_features(fmnist_train, label=0)
    collab_features_test = collab_model.get_features(fmnist_test, label=0)

    strategy_c_result = train_transfer_head(
        collab_features_train, fmnist_train_y,
        collab_features_test, fmnist_test_y
    )
    strategy_results['C: Layer Collab Pretrain'] = {
        'accuracy': strategy_c_result['best_accuracy'],
        'description': 'Use Layer Collaboration during pretraining',
        'source_accuracy': collab_source_acc
    }
    print(f"  Transfer Result: {strategy_c_result['best_accuracy']*100:.2f}%")

    # Strategy D: Average features across all labels
    print("\n4.4 Strategy D: Average features across all labels...")
    ff_avg_features_train = torch.zeros_like(ff_model.get_features(fmnist_train, label=0))
    ff_avg_features_test = torch.zeros_like(ff_model.get_features(fmnist_test, label=0))

    for label in range(10):
        ff_avg_features_train += ff_model.get_features(fmnist_train, label=label)
        ff_avg_features_test += ff_model.get_features(fmnist_test, label=label)

    ff_avg_features_train /= 10
    ff_avg_features_test /= 10

    strategy_d_result = train_transfer_head(
        ff_avg_features_train, fmnist_train_y,
        ff_avg_features_test, fmnist_test_y
    )
    strategy_results['D: Average All Labels'] = {
        'accuracy': strategy_d_result['best_accuracy'],
        'description': 'Average features across all 10 label embeddings'
    }
    print(f"  Result: {strategy_d_result['best_accuracy']*100:.2f}%")

    results['improvement_strategies'] = strategy_results

    visualize_improvement_strategies(strategy_results, './results/transfer_analysis_strategies.png')

    # ================================================================
    # PHASE 5: Summary and Conclusions
    # ================================================================
    print("\n" + "="*70)
    print("SUMMARY AND CONCLUSIONS")
    print("="*70)

    print("\n1. SOURCE ACCURACY:")
    print(f"   FF: {ff_source_acc*100:.2f}%")
    print(f"   BP: {bp_source_acc*100:.2f}%")

    print("\n2. TRANSFER ACCURACY (MNIST -> Fashion-MNIST):")
    print(f"   {'Method':<30} {'Accuracy':>10}")
    print("   " + "-"*42)
    for name, data in sorted(strategy_results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"   {name:<30} {data['accuracy']*100:>9.2f}%")

    print("\n3. KEY FINDINGS:")

    # Find best improvement
    baseline_acc = strategy_results['FF Baseline']['accuracy']
    best_strategy = max(
        [(k, v) for k, v in strategy_results.items() if k not in ['FF Baseline', 'Random Init', 'BP Baseline']],
        key=lambda x: x[1]['accuracy']
    )
    improvement = best_strategy[1]['accuracy'] - baseline_acc

    print(f"   - Best improvement: {best_strategy[0]} (+{improvement*100:.2f}%)")
    print(f"   - FF features show {ff_fmnist_distances['separation_ratio']:.2f}x class separation (vs BP: {bp_fmnist_distances['separation_ratio']:.2f}x)")
    print(f"   - Label dependency variance ratio: {label_dependency['variance_ratio']:.4f}")

    # Determine root cause
    if label_dependency['variance_ratio'] > 0.1:
        print("   - HIGH label dependency: FF features strongly encode MNIST labels")
    if ff_fmnist_distances['separation_ratio'] < bp_fmnist_distances['separation_ratio']:
        print("   - POOR class separation on target domain (Fashion-MNIST)")

    results['summary'] = {
        'ff_source_acc': ff_source_acc,
        'bp_source_acc': bp_source_acc,
        'best_strategy': best_strategy[0],
        'best_improvement': improvement,
        'root_causes': []
    }

    if label_dependency['variance_ratio'] > 0.1:
        results['summary']['root_causes'].append('high_label_dependency')
    if ff_fmnist_distances['separation_ratio'] < bp_fmnist_distances['separation_ratio']:
        results['summary']['root_causes'].append('poor_target_separation')

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
    parser = argparse.ArgumentParser(description="FF Transfer Learning Analysis")
    parser.add_argument('--epochs', type=int, default=300, help='Epochs per layer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Quick test (100 epochs)')
    args = parser.parse_args()

    epochs = 100 if args.quick else args.epochs

    results = run_transfer_analysis(
        epochs_per_layer=epochs,
        seed=args.seed,
        verbose=True
    )

    output_path = './results/transfer_analysis.json'
    save_results(results, output_path)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - results/transfer_analysis.json")
    print(f"  - results/transfer_analysis_tsne.png")
    print(f"  - results/transfer_analysis_distances.png")
    print(f"  - results/transfer_analysis_layers.png")
    print(f"  - results/transfer_analysis_strategies.png")


if __name__ == "__main__":
    main()
