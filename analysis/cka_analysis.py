#!/usr/bin/env python3
"""
CKA (Centered Kernel Alignment) Analysis for FF vs BP Networks
==============================================================

Compares layer-wise representations between Forward-Forward and Backpropagation networks.
Based on Kornblith et al., ICML 2019 "Similarity of Neural Network Representations Revisited"
"""

import sys
import os
sys.path.insert(0, os.path.expanduser('~/Desktop/Rios/ff-experiment/corrected'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from ff_core import FFNetwork, BPNetwork, embed_label, normalize


def centering_matrix(n: int) -> torch.Tensor:
    """Create centering matrix H = I - 1/n * 11^T"""
    return torch.eye(n) - torch.ones(n, n) / n


def hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Compute Hilbert-Schmidt Independence Criterion (HSIC).
    HSIC(K, L) = trace(KHLH) / (n-1)^2
    """
    n = K.shape[0]
    H = centering_matrix(n).to(K.device)
    return torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)


def linear_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute linear kernel K = X @ X^T"""
    return X @ X.T


def cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute CKA (Centered Kernel Alignment) between two representations.
    
    Args:
        X: Activations from model 1 (n_samples, n_features1)
        Y: Activations from model 2 (n_samples, n_features2)
    
    Returns:
        CKA similarity score in [0, 1]
    """
    # Use linear kernel
    K = linear_kernel(X)
    L = linear_kernel(Y)
    
    # Compute CKA
    hsic_KL = hsic(K, L)
    hsic_KK = hsic(K, K)
    hsic_LL = hsic(L, L)
    
    return (hsic_KL / torch.sqrt(hsic_KK * hsic_LL)).item()


def minibatch_cka(X: torch.Tensor, Y: torch.Tensor, batch_size: int = 256) -> float:
    """
    Compute CKA with minibatch estimation for memory efficiency.
    """
    n = X.shape[0]
    if n <= batch_size:
        return cka(X, Y)
    
    # Sample indices
    indices = torch.randperm(n)[:batch_size]
    return cka(X[indices], Y[indices])


class FFNetworkWithHooks(nn.Module):
    """FF Network with activation extraction hooks"""
    
    def __init__(self, base_model: FFNetwork):
        super().__init__()
        self.model = base_model
        self.activations = {}
    
    def get_layer_activations(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations from all layers for positive samples"""
        batch_size = images.size(0)
        images_flat = images.view(batch_size, -1)
        
        # Create positive samples with correct labels
        x = embed_label(images_flat, labels, self.model.num_classes)
        x = normalize(x)
        
        activations = {'input': x.clone()}
        
        for i, layer in enumerate(self.model.layers):
            x = layer.get_output(x)
            activations[f'layer_{i}'] = x.clone()
            x = normalize(x)
        
        return activations


class BPNetworkWithHooks(nn.Module):
    """BP Network with activation extraction hooks"""
    
    def __init__(self, layer_sizes: List[int], lr: float = 0.001):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        self.activations = {}
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_layer_activations(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations from all layers"""
        x = images.view(images.size(0), -1)
        activations = {'input': x.clone()}
        
        layer_idx = 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations[f'layer_{layer_idx}'] = x.clone()
                layer_idx += 1
        
        return activations
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images).argmax(dim=1)


def collect_activations(model, dataloader, device: str, model_type: str = 'ff') -> Dict[str, torch.Tensor]:
    """
    Collect activations from all layers for the entire dataset.
    
    Args:
        model: FF or BP network
        dataloader: Data loader
        device: Device to use
        model_type: 'ff' or 'bp'
    
    Returns:
        Dictionary mapping layer names to activation tensors
    """
    all_activations = {}
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Collecting {model_type.upper()} activations"):
            images, labels = images.to(device), labels.to(device)
            
            if model_type == 'ff':
                batch_acts = model.get_layer_activations(images, labels)
            else:
                batch_acts = model.get_layer_activations(images)
            
            for name, act in batch_acts.items():
                if name not in all_activations:
                    all_activations[name] = []
                all_activations[name].append(act.cpu())
    
    # Concatenate all batches
    return {name: torch.cat(acts, dim=0) for name, acts in all_activations.items()}


def compute_cka_matrix(ff_activations: Dict[str, torch.Tensor],
                       bp_activations: Dict[str, torch.Tensor],
                       device: str = 'cpu') -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute CKA similarity matrix between FF and BP layers.
    
    Returns:
        cka_matrix: 2D numpy array of CKA values
        ff_layer_names: List of FF layer names
        bp_layer_names: List of BP layer names
    """
    ff_names = [k for k in ff_activations.keys() if k.startswith('layer')]
    bp_names = [k for k in bp_activations.keys() if k.startswith('layer')]
    
    cka_matrix = np.zeros((len(ff_names), len(bp_names)))
    
    for i, ff_name in enumerate(tqdm(ff_names, desc="Computing CKA matrix")):
        for j, bp_name in enumerate(bp_names):
            ff_act = ff_activations[ff_name].to(device)
            bp_act = bp_activations[bp_name].to(device)
            cka_matrix[i, j] = minibatch_cka(ff_act, bp_act, batch_size=1000)
    
    return cka_matrix, ff_names, bp_names


def compute_self_cka_matrix(activations: Dict[str, torch.Tensor],
                            device: str = 'cpu') -> Tuple[np.ndarray, List[str]]:
    """
    Compute self-CKA matrix within a single network.
    """
    layer_names = [k for k in activations.keys() if k.startswith('layer')]
    n_layers = len(layer_names)
    cka_matrix = np.zeros((n_layers, n_layers))
    
    for i, name_i in enumerate(layer_names):
        for j, name_j in enumerate(layer_names):
            act_i = activations[name_i].to(device)
            act_j = activations[name_j].to(device)
            cka_matrix[i, j] = minibatch_cka(act_i, act_j, batch_size=1000)
    
    return cka_matrix, layer_names


def plot_cka_heatmap(cka_matrix: np.ndarray,
                     row_labels: List[str],
                     col_labels: List[str],
                     title: str,
                     save_path: str):
    """Plot CKA heatmap and save to file"""
    plt.figure(figsize=(10, 8))
    
    # Format labels
    row_labels_fmt = [f"FF {l.replace('layer_', 'L')}" for l in row_labels]
    col_labels_fmt = [f"BP {l.replace('layer_', 'L')}" for l in col_labels]
    
    ax = sns.heatmap(cka_matrix, 
                     xticklabels=col_labels_fmt,
                     yticklabels=row_labels_fmt,
                     annot=True, 
                     fmt='.2f',
                     cmap='viridis',
                     vmin=0, vmax=1,
                     cbar_kws={'label': 'CKA Similarity'})
    
    plt.title(title, fontsize=14)
    plt.xlabel('BP Network Layers', fontsize=12)
    plt.ylabel('FF Network Layers', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_self_cka_comparison(ff_cka: np.ndarray, bp_cka: np.ndarray,
                             layer_names: List[str], save_path: str):
    """Plot side-by-side self-CKA matrices for FF and BP"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = [l.replace('layer_', 'L') for l in layer_names]
    
    # FF self-CKA
    sns.heatmap(ff_cka, ax=axes[0], 
                xticklabels=labels, yticklabels=labels,
                annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('FF Network Self-CKA', fontsize=12)
    
    # BP self-CKA
    sns.heatmap(bp_cka, ax=axes[1],
                xticklabels=labels, yticklabels=labels,
                annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('BP Network Self-CKA', fontsize=12)
    
    plt.suptitle('Self-CKA Comparison: FF vs BP Layer Similarity', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_diagonal_cka(cka_matrix: np.ndarray, save_path: str):
    """Plot diagonal CKA values (corresponding layers)"""
    diagonal = np.diag(cka_matrix)
    layers = [f'Layer {i}' for i in range(len(diagonal))]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(layers, diagonal, color='steelblue', edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, diagonal):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', fontsize=10)
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('CKA Similarity', fontsize=12)
    plt.title('FF vs BP: Same-Layer CKA Similarity', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def train_models(device: str, epochs_ff: int = 10, epochs_bp: int = 10,
                 hidden_sizes: List[int] = [500, 500]):
    """Train FF and BP models on MNIST"""
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Train FF Network
    print("\n" + "="*60)
    print("Training Forward-Forward Network")
    print("="*60)
    
    ff_model = FFNetwork(input_size=784, hidden_sizes=hidden_sizes, 
                         num_classes=10, threshold=2.0, lr=0.03)
    ff_model.to(device)
    
    for epoch in range(epochs_ff):
        epoch_stats = []
        for images, labels in tqdm(train_loader, desc=f"FF Epoch {epoch+1}/{epochs_ff}"):
            images, labels = images.to(device), labels.to(device)
            stats = ff_model.train_batch(images, labels)
            epoch_stats.append(stats)
        
        # Evaluate
        ff_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = ff_model.predict(images)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total * 100
        avg_loss = np.mean([s[0]['loss'] for s in epoch_stats])
        print(f"Epoch {epoch+1}: Test Acc = {acc:.2f}%, Avg Loss = {avg_loss:.4f}")
    
    # Train BP Network  
    print("\n" + "="*60)
    print("Training Backpropagation Network")
    print("="*60)
    
    bp_layer_sizes = [784] + hidden_sizes + [10]
    bp_model = BPNetworkWithHooks(bp_layer_sizes, lr=0.001)
    bp_model.to(device)
    
    for epoch in range(epochs_bp):
        epoch_loss = 0
        for images, labels in tqdm(train_loader, desc=f"BP Epoch {epoch+1}/{epochs_bp}"):
            images, labels = images.to(device), labels.to(device)
            loss = bp_model.train_step(images, labels)
            epoch_loss += loss
        
        # Evaluate
        bp_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = bp_model.predict(images)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total * 100
        print(f"Epoch {epoch+1}: Test Acc = {acc:.2f}%, Avg Loss = {epoch_loss/len(train_loader):.4f}")
    
    return ff_model, bp_model, test_loader


def main():
    """Main analysis pipeline"""
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Output paths
    results_dir = os.path.expanduser('~/Desktop/Rios/ff-research/results')
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Train models
    hidden_sizes = [500, 500, 500]  # 3 hidden layers for richer analysis
    ff_model, bp_model, test_loader = train_models(
        device, epochs_ff=15, epochs_bp=15, hidden_sizes=hidden_sizes
    )
    
    # Wrap FF model for activation extraction
    ff_wrapped = FFNetworkWithHooks(ff_model)
    
    # Use subset for CKA (memory efficient)
    subset_indices = list(range(2000))
    subset_loader = DataLoader(
        Subset(test_loader.dataset, subset_indices),
        batch_size=256, shuffle=False
    )
    
    print("\n" + "="*60)
    print("Computing CKA Analysis")
    print("="*60)
    
    # Collect activations
    ff_activations = collect_activations(ff_wrapped, subset_loader, device, 'ff')
    bp_activations = collect_activations(bp_model, subset_loader, device, 'bp')
    
    # Cross-model CKA
    print("\nComputing FF vs BP CKA matrix...")
    cka_matrix, ff_names, bp_names = compute_cka_matrix(ff_activations, bp_activations, device)
    
    # Self-CKA matrices
    print("\nComputing self-CKA matrices...")
    ff_self_cka, _ = compute_self_cka_matrix(ff_activations, device)
    bp_self_cka, _ = compute_self_cka_matrix(bp_activations, device)
    
    # Save results
    np.save(os.path.join(results_dir, 'cka_ff_bp.npy'), cka_matrix)
    np.save(os.path.join(results_dir, 'cka_ff_self.npy'), ff_self_cka)
    np.save(os.path.join(results_dir, 'cka_bp_self.npy'), bp_self_cka)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    plot_cka_heatmap(cka_matrix, ff_names, bp_names,
                     'CKA Similarity: FF vs BP Networks (MNIST)',
                     os.path.join(viz_dir, 'cka_ff_vs_bp.png'))
    
    plot_self_cka_comparison(ff_self_cka, bp_self_cka, ff_names,
                             os.path.join(viz_dir, 'cka_self_comparison.png'))
    
    plot_diagonal_cka(cka_matrix, os.path.join(viz_dir, 'cka_diagonal.png'))
    
    # Print summary
    print("\n" + "="*60)
    print("CKA Analysis Summary")
    print("="*60)
    
    print("\nFF vs BP Cross-CKA Matrix:")
    print(f"  Shape: {cka_matrix.shape}")
    print(f"  Diagonal (same-layer similarity): {np.diag(cka_matrix)}")
    print(f"  Mean diagonal CKA: {np.mean(np.diag(cka_matrix)):.3f}")
    print(f"  Min diagonal CKA: {np.min(np.diag(cka_matrix)):.3f} (Layer {np.argmin(np.diag(cka_matrix))})")
    print(f"  Max diagonal CKA: {np.max(np.diag(cka_matrix)):.3f} (Layer {np.argmax(np.diag(cka_matrix))})")
    
    print("\nFF Self-CKA (layer correlation):")
    print(f"  Mean off-diagonal: {(ff_self_cka.sum() - np.trace(ff_self_cka)) / (ff_self_cka.size - len(ff_names)):.3f}")
    
    print("\nBP Self-CKA (layer correlation):")
    print(f"  Mean off-diagonal: {(bp_self_cka.sum() - np.trace(bp_self_cka)) / (bp_self_cka.size - len(bp_names)):.3f}")
    
    # Save summary to file
    summary = {
        'cka_diagonal': np.diag(cka_matrix).tolist(),
        'mean_diagonal_cka': float(np.mean(np.diag(cka_matrix))),
        'min_diagonal_cka': float(np.min(np.diag(cka_matrix))),
        'max_diagonal_cka': float(np.max(np.diag(cka_matrix))),
        'ff_self_cka_mean_offdiag': float((ff_self_cka.sum() - np.trace(ff_self_cka)) / (ff_self_cka.size - len(ff_names))),
        'bp_self_cka_mean_offdiag': float((bp_self_cka.sum() - np.trace(bp_self_cka)) / (bp_self_cka.size - len(bp_names))),
    }
    
    import json
    with open(os.path.join(results_dir, 'cka_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    print("Done!")
    
    return cka_matrix, ff_self_cka, bp_self_cka


if __name__ == '__main__':
    main()
