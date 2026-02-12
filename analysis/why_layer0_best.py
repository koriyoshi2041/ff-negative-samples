#!/usr/bin/env python3
"""
Deep Analysis: Why Layer 0 Only Has Best FF Transfer Performance
================================================================

Empirical Findings (from transfer_analysis.json):
- Layer 0 only: FF 77.2%, BP 80.9%, Random 85%
- Layer 0-1:    FF 50.0%, BP 76.5%, Random 84%
- All Layers:   FF 50.0%, BP 78.0%, Random 84%

Key Observation:
- FF accuracy DROPS significantly (77% -> 50%) when adding more layers
- BP and Random remain relatively stable

This script investigates WHY this happens through:
1. t-SNE visualization of each layer's feature space
2. Feature statistics (sparsity, class separation, linear separability)
3. Label embedding impact analysis
4. Layer-wise goodness distribution analysis

Author: Analysis for Parafee
Date: 2026-02-09
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    train_dataset = dataset_class(data_dir, train=True, download=True, transform=transform)
    test_dataset = dataset_class(data_dir, train=False, download=True, transform=transform)

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
        for epoch in range(num_epochs):
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

    def get_layer_features(self, x: torch.Tensor, up_to_layer: int, label: int = 0) -> torch.Tensor:
        """Get features up to specific layer with label embedding."""
        with torch.no_grad():
            batch_size = x.shape[0]
            h = overlay_y_on_x(x, torch.full((batch_size,), label, dtype=torch.long, device=x.device))
            for i, layer in enumerate(self.layers[:up_to_layer + 1]):
                h = layer(h)
            return h

    def get_layer_features_with_labels(self, x: torch.Tensor, y: torch.Tensor, up_to_layer: int) -> torch.Tensor:
        """Get features using correct labels for each sample."""
        with torch.no_grad():
            h = overlay_y_on_x(x, y)
            for i, layer in enumerate(self.layers[:up_to_layer + 1]):
                h = layer(h)
            return h

    def get_layer_features_no_label(self, x: torch.Tensor, up_to_layer: int) -> torch.Tensor:
        """Get features without label embedding (first 10 pixels zeroed)."""
        with torch.no_grad():
            h = x.clone()
            h[:, :10] = 0
            for i, layer in enumerate(self.layers[:up_to_layer + 1]):
                h = layer(h)
            return h

    def get_goodness_per_layer(self, x: torch.Tensor, label: int = 0) -> List[torch.Tensor]:
        """Get goodness at each layer."""
        with torch.no_grad():
            batch_size = x.shape[0]
            h = overlay_y_on_x(x, torch.full((batch_size,), label, dtype=torch.long, device=x.device))
            goodness_list = []
            for layer in self.layers:
                h = layer(h)
                goodness_list.append(layer.goodness(h))
            return goodness_list


class BPNetwork(nn.Module):
    """Standard backprop network for comparison."""

    def __init__(self, dims: List[int], num_classes: int = 10):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_layer_features(self, x: torch.Tensor, up_to_layer: int) -> torch.Tensor:
        """Get features up to specific layer."""
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


def train_bp_network(model: BPNetwork, x_train: torch.Tensor, y_train: torch.Tensor,
                     epochs: int = 50, batch_size: int = 128, lr: float = 0.001) -> float:
    """Train BP network."""
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


# ============================================================
# Analysis Functions
# ============================================================

def compute_sparsity(features: torch.Tensor, threshold: float = 0.01) -> Dict[str, float]:
    """Compute sparsity metrics for features."""
    features_np = features.cpu().numpy()

    # Activation rate (fraction of non-zero values)
    activation_rate = (np.abs(features_np) > threshold).mean()

    # Average sparsity per sample
    per_sample_sparsity = (np.abs(features_np) > threshold).mean(axis=1)

    # Hoyer sparsity (L1/L2 ratio based)
    l1 = np.abs(features_np).sum(axis=1)
    l2 = np.sqrt((features_np ** 2).sum(axis=1))
    n = features_np.shape[1]
    hoyer = (np.sqrt(n) - l1 / (l2 + 1e-8)) / (np.sqrt(n) - 1)

    return {
        'activation_rate': float(activation_rate),
        'sparsity_mean': float(1 - per_sample_sparsity.mean()),
        'sparsity_std': float(per_sample_sparsity.std()),
        'hoyer_sparsity_mean': float(hoyer.mean()),
        'hoyer_sparsity_std': float(hoyer.std()),
    }


def compute_class_separation(features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute intra-class and inter-class distances."""
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    unique_labels = np.unique(labels_np)

    # Compute class centroids
    centroids = {}
    for label in unique_labels:
        mask = labels_np == label
        centroids[label] = features_np[mask].mean(axis=0)

    # Intra-class distances (distance to own centroid)
    intra_distances = []
    for label in unique_labels:
        mask = labels_np == label
        class_features = features_np[mask]
        centroid = centroids[label]
        dists = np.linalg.norm(class_features - centroid, axis=1)
        intra_distances.extend(dists.tolist())

    # Inter-class distances (between centroids)
    centroid_array = np.array([centroids[l] for l in unique_labels])
    inter_distances = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            dist = np.linalg.norm(centroid_array[i] - centroid_array[j])
            inter_distances.append(dist)

    intra_mean = np.mean(intra_distances)
    inter_mean = np.mean(inter_distances)

    return {
        'intra_class_mean': float(intra_mean),
        'intra_class_std': float(np.std(intra_distances)),
        'inter_class_mean': float(inter_mean),
        'inter_class_std': float(np.std(inter_distances)),
        'separation_ratio': float(inter_mean / (intra_mean + 1e-8)),
        'fisher_criterion': float((inter_mean ** 2) / (intra_mean ** 2 + 1e-8))
    }


def compute_linear_separability(features: torch.Tensor, labels: torch.Tensor,
                                 test_features: torch.Tensor = None,
                                 test_labels: torch.Tensor = None) -> Dict[str, float]:
    """Compute linear separability using Logistic Regression."""
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_np)

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    clf.fit(features_scaled, labels_np)

    train_acc = clf.score(features_scaled, labels_np)

    test_acc = None
    if test_features is not None and test_labels is not None:
        test_np = test_features.cpu().numpy()
        test_labels_np = test_labels.cpu().numpy()
        test_scaled = scaler.transform(test_np)
        test_acc = clf.score(test_scaled, test_labels_np)

    return {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc) if test_acc is not None else None
    }


def analyze_label_sensitivity(ff_model: FFNetwork, x: torch.Tensor,
                               layer_idx: int) -> Dict[str, float]:
    """Analyze how sensitive a layer is to label embedding changes."""
    # Get features with different labels
    features_by_label = []
    for label in range(10):
        features = ff_model.get_layer_features(x[:1000], layer_idx, label=label)
        features_by_label.append(features)

    stacked = torch.stack(features_by_label, dim=0)  # [10, N, D]

    # Variance across labels (for same input sample)
    label_variance = stacked.var(dim=0).mean().item()

    # Total variance
    total_variance = stacked.view(-1, stacked.shape[-1]).var(dim=0).mean().item()

    # Compute pairwise distances between features with different labels
    label_distances = []
    for i in range(10):
        for j in range(i + 1, 10):
            dist = (features_by_label[i] - features_by_label[j]).pow(2).mean().item()
            label_distances.append(dist)

    return {
        'label_variance': label_variance,
        'total_variance': total_variance,
        'variance_ratio': label_variance / (total_variance + 1e-8),
        'mean_label_distance': np.mean(label_distances),
        'max_label_distance': np.max(label_distances),
    }


def visualize_layer_features_tsne(features_dict: Dict[str, torch.Tensor],
                                   labels: torch.Tensor,
                                   output_path: str,
                                   title: str,
                                   n_samples: int = 2000):
    """Create t-SNE visualization comparing features from different layers/models."""
    n_plots = len(features_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    labels_np = labels[:n_samples].cpu().numpy()

    for idx, (name, features) in enumerate(features_dict.items()):
        features_subset = features[:n_samples].cpu().numpy()

        print(f"  Running t-SNE for {name}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_2d = tsne.fit_transform(features_subset)

        scatter = axes[idx].scatter(features_2d[:, 0], features_2d[:, 1],
                                     c=labels_np, cmap='tab10', alpha=0.6, s=10)
        axes[idx].set_title(name, fontsize=12)
        axes[idx].set_xlabel('t-SNE 1')
        axes[idx].set_ylabel('t-SNE 2')

    plt.colorbar(scatter, ax=axes[-1], label='Class')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_goodness_distribution(ff_model: FFNetwork,
                                     mnist_test: torch.Tensor,
                                     fmnist_test: torch.Tensor,
                                     output_path: str):
    """Visualize goodness distribution per layer for MNIST vs Fashion-MNIST."""
    n_layers = len(ff_model.layers)
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 4 * n_layers))

    for layer_idx in range(n_layers):
        # MNIST goodness for correct label (label=0 as proxy since we don't have true labels)
        mnist_goodness = []
        fmnist_goodness = []

        for label in range(10):
            mnist_g = ff_model.get_goodness_per_layer(mnist_test[:1000], label=label)[layer_idx]
            fmnist_g = ff_model.get_goodness_per_layer(fmnist_test[:1000], label=label)[layer_idx]
            mnist_goodness.append(mnist_g.cpu().numpy())
            fmnist_goodness.append(fmnist_g.cpu().numpy())

        mnist_goodness = np.concatenate(mnist_goodness)
        fmnist_goodness = np.concatenate(fmnist_goodness)

        # Plot histograms
        axes[layer_idx, 0].hist(mnist_goodness, bins=50, alpha=0.7, label='MNIST', density=True)
        axes[layer_idx, 0].hist(fmnist_goodness, bins=50, alpha=0.7, label='F-MNIST', density=True)
        axes[layer_idx, 0].set_title(f'Layer {layer_idx} - Goodness Distribution')
        axes[layer_idx, 0].set_xlabel('Goodness')
        axes[layer_idx, 0].set_ylabel('Density')
        axes[layer_idx, 0].legend()
        axes[layer_idx, 0].axvline(x=ff_model.layers[layer_idx].threshold, color='r',
                                    linestyle='--', label='Threshold')

        # Box plot comparison
        axes[layer_idx, 1].boxplot([mnist_goodness, fmnist_goodness],
                                    labels=['MNIST', 'Fashion-MNIST'])
        axes[layer_idx, 1].set_title(f'Layer {layer_idx} - Goodness Comparison')
        axes[layer_idx, 1].set_ylabel('Goodness')
        axes[layer_idx, 1].axhline(y=ff_model.layers[layer_idx].threshold, color='r',
                                    linestyle='--', label='Threshold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_layer_statistics(stats: Dict, output_path: str):
    """Visualize layer-wise statistics comparison."""
    layers = list(stats['ff'].keys())
    metrics = ['sparsity_mean', 'separation_ratio', 'linear_train_acc', 'linear_test_acc']
    metric_labels = ['Sparsity', 'Class Separation', 'Linear Train Acc', 'Linear Test Acc']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ff_vals = []
        bp_vals = []
        random_vals = []

        for layer in layers:
            if metric.startswith('linear'):
                key = metric.replace('linear_', '')
                ff_vals.append(stats['ff'][layer].get('linear_separability', {}).get(key, 0))
                bp_vals.append(stats['bp'][layer].get('linear_separability', {}).get(key, 0))
                random_vals.append(stats['random'][layer].get('linear_separability', {}).get(key, 0))
            elif metric == 'separation_ratio':
                ff_vals.append(stats['ff'][layer].get('class_separation', {}).get(metric, 0))
                bp_vals.append(stats['bp'][layer].get('class_separation', {}).get(metric, 0))
                random_vals.append(stats['random'][layer].get('class_separation', {}).get(metric, 0))
            else:
                ff_vals.append(stats['ff'][layer].get('sparsity', {}).get(metric, 0))
                bp_vals.append(stats['bp'][layer].get('sparsity', {}).get(metric, 0))
                random_vals.append(stats['random'][layer].get('sparsity', {}).get(metric, 0))

        x = np.arange(len(layers))
        width = 0.25

        axes[idx].bar(x - width, ff_vals, width, label='FF', color='#e74c3c')
        axes[idx].bar(x, bp_vals, width, label='BP', color='#3498db')
        axes[idx].bar(x + width, random_vals, width, label='Random', color='#2ecc71')

        axes[idx].set_xlabel('Layer')
        axes[idx].set_ylabel(label)
        axes[idx].set_title(f'{label} by Layer')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(layers)
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_label_sensitivity(sensitivity_stats: Dict, output_path: str):
    """Visualize label sensitivity by layer."""
    layers = list(sensitivity_stats.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Variance ratio by layer
    variance_ratios = [sensitivity_stats[l]['variance_ratio'] for l in layers]
    axes[0].bar(layers, variance_ratios, color='#9b59b6')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Label Variance / Total Variance')
    axes[0].set_title('Label Embedding Impact by Layer')
    axes[0].grid(axis='y', alpha=0.3)

    # Mean label distance by layer
    mean_distances = [sensitivity_stats[l]['mean_label_distance'] for l in layers]
    axes[1].bar(layers, mean_distances, color='#e67e22')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Mean Distance Between Label Embeddings')
    axes[1].set_title('Label Sensitivity by Layer')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_transfer_degradation(transfer_results: Dict, output_path: str):
    """Visualize how transfer accuracy changes with more layers."""
    layers = ['Layer 0', 'Layer 0-1', 'All Layers']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute accuracy
    ff_accs = [transfer_results[l]['ff'] * 100 for l in layers]
    bp_accs = [transfer_results[l]['bp'] * 100 for l in layers]
    random_accs = [transfer_results[l]['random'] * 100 for l in layers]

    x = np.arange(len(layers))
    width = 0.25

    axes[0].bar(x - width, ff_accs, width, label='FF', color='#e74c3c')
    axes[0].bar(x, bp_accs, width, label='BP', color='#3498db')
    axes[0].bar(x + width, random_accs, width, label='Random', color='#2ecc71')
    axes[0].set_xlabel('Layers Transferred')
    axes[0].set_ylabel('Transfer Accuracy (%)')
    axes[0].set_title('Transfer Accuracy by Layer Depth')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layers)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Add value annotations
    for i, (ff, bp, rand) in enumerate(zip(ff_accs, bp_accs, random_accs)):
        axes[0].annotate(f'{ff:.1f}', (i - width, ff + 1), ha='center', fontsize=8)
        axes[0].annotate(f'{bp:.1f}', (i, bp + 1), ha='center', fontsize=8)
        axes[0].annotate(f'{rand:.1f}', (i + width, rand + 1), ha='center', fontsize=8)

    # Relative change from Layer 0
    ff_delta = [0] + [ff_accs[i] - ff_accs[0] for i in range(1, len(ff_accs))]
    bp_delta = [0] + [bp_accs[i] - bp_accs[0] for i in range(1, len(bp_accs))]
    random_delta = [0] + [random_accs[i] - random_accs[0] for i in range(1, len(random_accs))]

    axes[1].plot(layers, ff_delta, 'o-', label='FF', color='#e74c3c', linewidth=2, markersize=10)
    axes[1].plot(layers, bp_delta, 's-', label='BP', color='#3498db', linewidth=2, markersize=10)
    axes[1].plot(layers, random_delta, '^-', label='Random', color='#2ecc71', linewidth=2, markersize=10)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Layers Transferred')
    axes[1].set_ylabel('Change in Accuracy (pp) vs Layer 0 Only')
    axes[1].set_title('Accuracy Change When Adding More Layers')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Highlight the FF degradation
    axes[1].fill_between(layers, ff_delta, alpha=0.3, color='#e74c3c')
    axes[1].annotate(f'FF drops {abs(ff_delta[-1]):.1f}pp!',
                     xy=(2, ff_delta[-1]), xytext=(1.5, ff_delta[-1] - 5),
                     fontsize=10, fontweight='bold', color='#c0392b',
                     arrowprops=dict(arrowstyle='->', color='#c0392b'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Main Analysis
# ============================================================

def run_layer_depth_analysis(epochs_per_layer: int = 200, seed: int = 42, verbose: bool = True):
    """Run comprehensive layer depth analysis."""
    print("="*70)
    print("WHY LAYER 0 ONLY HAS BEST FF TRANSFER PERFORMANCE")
    print("="*70)

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Epochs per layer: {epochs_per_layer}")

    results = {
        'experiment': 'layer_depth_analysis',
        'config': {
            'epochs_per_layer': epochs_per_layer,
            'seed': seed,
            'device': str(device)
        }
    }

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'results', 'layer_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n" + "-"*60)
    print("Loading datasets...")
    (mnist_train, mnist_train_y), (mnist_test, mnist_test_y) = get_data('mnist', device)
    (fmnist_train, fmnist_train_y), (fmnist_test, fmnist_test_y) = get_data('fashion_mnist', device)
    print(f"  MNIST: {len(mnist_train)} train, {len(mnist_test)} test")
    print(f"  Fashion-MNIST: {len(fmnist_train)} train, {len(fmnist_test)} test")

    # ================================================================
    # Train Models
    # ================================================================
    print("\n" + "="*60)
    print("Training Models...")
    print("="*60)

    # Train FF
    print("\nTraining Forward-Forward on MNIST...")
    torch.manual_seed(seed)
    ff_model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    x_pos = overlay_y_on_x(mnist_train, mnist_train_y)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg = overlay_y_on_x(mnist_train, mnist_train_y[rnd])
    ff_model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)

    # Train BP
    print("\nTraining Backpropagation on MNIST...")
    torch.manual_seed(seed)
    bp_model = BPNetwork([784, 500, 500], num_classes=10).to(device)
    train_bp_network(bp_model, mnist_train, mnist_train_y, epochs=50)

    # Random baseline
    print("\nInitializing Random baseline...")
    torch.manual_seed(seed + 1000)  # Different seed for random
    random_model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # ================================================================
    # Analysis 1: t-SNE Visualization per Layer
    # ================================================================
    print("\n" + "="*60)
    print("Analysis 1: t-SNE Feature Visualization")
    print("="*60)

    # Visualize MNIST features at each layer
    print("\n1.1 MNIST features by layer...")
    mnist_features = {}
    for layer_idx in range(len(ff_model.layers)):
        mnist_features[f'FF Layer {layer_idx}'] = ff_model.get_layer_features(mnist_test, layer_idx, label=0)
        mnist_features[f'BP Layer {layer_idx}'] = bp_model.get_layer_features(mnist_test, layer_idx)

    visualize_layer_features_tsne(
        {f'FF Layer {i}': ff_model.get_layer_features(mnist_test, i, label=0) for i in range(2)},
        mnist_test_y,
        os.path.join(output_dir, 'tsne_ff_mnist_layers.png'),
        'FF Features by Layer (MNIST)'
    )

    # Visualize Fashion-MNIST features at each layer
    print("\n1.2 Fashion-MNIST features by layer (using MNIST-trained model)...")
    visualize_layer_features_tsne(
        {f'FF Layer {i}': ff_model.get_layer_features(fmnist_test, i, label=0) for i in range(2)},
        fmnist_test_y,
        os.path.join(output_dir, 'tsne_ff_fmnist_layers.png'),
        'FF Features by Layer (Fashion-MNIST with MNIST model)'
    )

    # Compare FF vs BP vs Random for Layer 0 and Layer 1
    print("\n1.3 Comparing FF/BP/Random on Fashion-MNIST...")
    visualize_layer_features_tsne(
        {
            'FF Layer 0': ff_model.get_layer_features(fmnist_test, 0, label=0),
            'BP Layer 0': bp_model.get_layer_features(fmnist_test, 0),
            'Random Layer 0': random_model.get_layer_features(fmnist_test, 0, label=0),
        },
        fmnist_test_y,
        os.path.join(output_dir, 'tsne_layer0_comparison.png'),
        'Layer 0 Features Comparison (Fashion-MNIST)'
    )

    visualize_layer_features_tsne(
        {
            'FF Layer 1': ff_model.get_layer_features(fmnist_test, 1, label=0),
            'BP Layer 1': bp_model.get_layer_features(fmnist_test, 1),
            'Random Layer 1': random_model.get_layer_features(fmnist_test, 1, label=0),
        },
        fmnist_test_y,
        os.path.join(output_dir, 'tsne_layer1_comparison.png'),
        'Layer 1 Features Comparison (Fashion-MNIST)'
    )

    # ================================================================
    # Analysis 2: Feature Statistics per Layer
    # ================================================================
    print("\n" + "="*60)
    print("Analysis 2: Feature Statistics per Layer")
    print("="*60)

    layer_stats = {'ff': {}, 'bp': {}, 'random': {}}

    for layer_idx in range(len(ff_model.layers)):
        layer_name = f'Layer {layer_idx}'
        print(f"\n  Analyzing {layer_name}...")

        # FF features on Fashion-MNIST
        ff_features_train = ff_model.get_layer_features(fmnist_train, layer_idx, label=0)
        ff_features_test = ff_model.get_layer_features(fmnist_test, layer_idx, label=0)

        # BP features
        bp_features_train = bp_model.get_layer_features(fmnist_train, layer_idx)
        bp_features_test = bp_model.get_layer_features(fmnist_test, layer_idx)

        # Random features
        random_features_train = random_model.get_layer_features(fmnist_train, layer_idx, label=0)
        random_features_test = random_model.get_layer_features(fmnist_test, layer_idx, label=0)

        # Compute statistics for each model
        for model_name, (train_feat, test_feat) in [
            ('ff', (ff_features_train, ff_features_test)),
            ('bp', (bp_features_train, bp_features_test)),
            ('random', (random_features_train, random_features_test))
        ]:
            sparsity = compute_sparsity(train_feat)
            class_sep = compute_class_separation(train_feat, fmnist_train_y)
            linear_sep = compute_linear_separability(
                train_feat, fmnist_train_y, test_feat, fmnist_test_y
            )

            layer_stats[model_name][layer_name] = {
                'sparsity': sparsity,
                'class_separation': class_sep,
                'linear_separability': linear_sep
            }

            print(f"    {model_name.upper()}: Sparsity={sparsity['sparsity_mean']:.3f}, "
                  f"Separation={class_sep['separation_ratio']:.3f}, "
                  f"Linear={linear_sep['test_accuracy']:.3f}")

    results['layer_statistics'] = layer_stats

    visualize_layer_statistics(layer_stats, os.path.join(output_dir, 'layer_statistics.png'))

    # ================================================================
    # Analysis 3: Label Embedding Sensitivity
    # ================================================================
    print("\n" + "="*60)
    print("Analysis 3: Label Embedding Sensitivity")
    print("="*60)

    label_sensitivity = {}

    for layer_idx in range(len(ff_model.layers)):
        layer_name = f'Layer {layer_idx}'
        print(f"\n  Analyzing label sensitivity for {layer_name}...")

        sensitivity = analyze_label_sensitivity(ff_model, fmnist_test, layer_idx)
        label_sensitivity[layer_name] = sensitivity

        print(f"    Variance ratio: {sensitivity['variance_ratio']:.4f}")
        print(f"    Mean label distance: {sensitivity['mean_label_distance']:.4f}")

    results['label_sensitivity'] = label_sensitivity

    visualize_label_sensitivity(label_sensitivity, os.path.join(output_dir, 'label_sensitivity.png'))

    # ================================================================
    # Analysis 4: Goodness Distribution Analysis
    # ================================================================
    print("\n" + "="*60)
    print("Analysis 4: Goodness Distribution Analysis")
    print("="*60)

    visualize_goodness_distribution(
        ff_model, mnist_test, fmnist_test,
        os.path.join(output_dir, 'goodness_distribution.png')
    )

    # Compute goodness statistics
    goodness_stats = {}
    for layer_idx in range(len(ff_model.layers)):
        layer_name = f'Layer {layer_idx}'

        # Average goodness across all labels for MNIST and Fashion-MNIST
        mnist_goodness = []
        fmnist_goodness = []

        for label in range(10):
            mnist_g = ff_model.get_goodness_per_layer(mnist_test, label=label)[layer_idx]
            fmnist_g = ff_model.get_goodness_per_layer(fmnist_test, label=label)[layer_idx]
            mnist_goodness.append(mnist_g.mean().item())
            fmnist_goodness.append(fmnist_g.mean().item())

        goodness_stats[layer_name] = {
            'mnist_mean': np.mean(mnist_goodness),
            'mnist_std': np.std(mnist_goodness),
            'fmnist_mean': np.mean(fmnist_goodness),
            'fmnist_std': np.std(fmnist_goodness),
            'domain_gap': abs(np.mean(mnist_goodness) - np.mean(fmnist_goodness)),
            'threshold': ff_model.layers[layer_idx].threshold
        }

        print(f"\n  {layer_name}:")
        print(f"    MNIST goodness: {goodness_stats[layer_name]['mnist_mean']:.3f} +/- {goodness_stats[layer_name]['mnist_std']:.3f}")
        print(f"    F-MNIST goodness: {goodness_stats[layer_name]['fmnist_mean']:.3f} +/- {goodness_stats[layer_name]['fmnist_std']:.3f}")
        print(f"    Domain gap: {goodness_stats[layer_name]['domain_gap']:.3f}")

    results['goodness_analysis'] = goodness_stats

    # ================================================================
    # Analysis 5: Transfer Degradation Visualization
    # ================================================================
    print("\n" + "="*60)
    print("Analysis 5: Transfer Degradation Visualization")
    print("="*60)

    # Load existing transfer results
    transfer_results = {
        'Layer 0': {'ff': 0.772, 'bp': 0.809, 'random': 0.850},
        'Layer 0-1': {'ff': 0.500, 'bp': 0.765, 'random': 0.840},
        'All Layers': {'ff': 0.500, 'bp': 0.780, 'random': 0.840}
    }

    results['transfer_comparison'] = transfer_results

    visualize_transfer_degradation(
        transfer_results,
        os.path.join(output_dir, 'transfer_degradation.png')
    )

    # ================================================================
    # Summary and Conclusions
    # ================================================================
    print("\n" + "="*70)
    print("SUMMARY AND CONCLUSIONS")
    print("="*70)

    conclusions = []

    # Check if deeper layers have worse separation
    layer0_sep = layer_stats['ff']['Layer 0']['class_separation']['separation_ratio']
    layer1_sep = layer_stats['ff']['Layer 1']['class_separation']['separation_ratio']

    if layer1_sep < layer0_sep:
        conclusions.append(f"Layer 1 has WORSE class separation ({layer1_sep:.3f}) than Layer 0 ({layer0_sep:.3f})")

    # Check label sensitivity
    layer0_label_sens = label_sensitivity['Layer 0']['variance_ratio']
    layer1_label_sens = label_sensitivity['Layer 1']['variance_ratio']

    if layer1_label_sens > layer0_label_sens:
        conclusions.append(f"Layer 1 is MORE sensitive to label embedding ({layer1_label_sens:.4f}) than Layer 0 ({layer0_label_sens:.4f})")

    # Check linear separability
    layer0_linear = layer_stats['ff']['Layer 0']['linear_separability']['test_accuracy']
    layer1_linear = layer_stats['ff']['Layer 1']['linear_separability']['test_accuracy']

    if layer1_linear < layer0_linear:
        conclusions.append(f"Layer 1 has WORSE linear separability ({layer1_linear:.3f}) than Layer 0 ({layer0_linear:.3f})")

    # Check goodness domain gap
    layer0_gap = goodness_stats['Layer 0']['domain_gap']
    layer1_gap = goodness_stats['Layer 1']['domain_gap']

    if layer1_gap > layer0_gap:
        conclusions.append(f"Layer 1 has LARGER goodness domain gap ({layer1_gap:.3f}) than Layer 0 ({layer0_gap:.3f})")

    print("\nKEY FINDINGS:")
    for i, conclusion in enumerate(conclusions, 1):
        print(f"  {i}. {conclusion}")

    # Root cause analysis
    print("\nROOT CAUSE ANALYSIS:")
    print("  The dramatic drop in FF transfer accuracy (77% -> 50%) when adding deeper layers")
    print("  is likely caused by:")
    print("")
    print("  1. LABEL EMBEDDING DEPENDENCY:")
    print("     - FF training requires label embedding in the input")
    print("     - Layer 0 learns to extract low-level features that are label-independent")
    print("     - Deeper layers become increasingly dependent on the SPECIFIC label encoding")
    print("     - On new domain (Fashion-MNIST), the label semantics are different")
    print("     - This causes deeper layer features to be MISLEADING for the new domain")
    print("")
    print("  2. TASK-SPECIFIC FEATURES:")
    print("     - FF's goodness-based objective makes deeper layers focus on")
    print("       maximizing goodness for MNIST specifically")
    print("     - These task-specific features do not transfer well")
    print("")
    print("  3. NEGATIVE SAMPLE ARTIFACTS:")
    print("     - The shuffled-label negative samples create an adversarial signal")
    print("     - Deeper layers learn to detect MNIST-specific negative patterns")
    print("     - These patterns are not meaningful for Fashion-MNIST")
    print("")
    print("  RECOMMENDATION:")
    print("  - For transfer learning with FF, use ONLY Layer 0 (or shallow layers)")
    print("  - Retrain deeper layers on the target domain")
    print("  - Consider label-free FF training methods for better transferability")

    results['conclusions'] = conclusions
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
    parser = argparse.ArgumentParser(description="Layer Depth Analysis for FF Transfer Learning")
    parser.add_argument('--epochs', type=int, default=200, help='Epochs per layer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Quick test (50 epochs)')
    args = parser.parse_args()

    epochs = 50 if args.quick else args.epochs

    results = run_layer_depth_analysis(
        epochs_per_layer=epochs,
        seed=args.seed,
        verbose=True
    )

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'layer_depth_analysis.json'
    )
    save_results(results, output_path)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput files in results/layer_analysis/:")
    print(f"  - tsne_ff_mnist_layers.png")
    print(f"  - tsne_ff_fmnist_layers.png")
    print(f"  - tsne_layer0_comparison.png")
    print(f"  - tsne_layer1_comparison.png")
    print(f"  - layer_statistics.png")
    print(f"  - label_sensitivity.png")
    print(f"  - goodness_distribution.png")
    print(f"  - transfer_degradation.png")
    print(f"\nJSON results: results/layer_depth_analysis.json")


if __name__ == "__main__":
    main()
