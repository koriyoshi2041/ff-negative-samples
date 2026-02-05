#!/usr/bin/env python3
"""
Linear Probing Analysis for FF vs BP Networks
==============================================

Evaluates the linear separability of features at each layer.
This measures "how much task-relevant information" each layer contains.
"""

import sys
import os
sys.path.insert(0, os.path.expanduser('~/Desktop/Rios/ff-experiment/corrected'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from ff_core import FFNetwork, embed_label, normalize


class FFNetworkWithHooks(nn.Module):
    """FF Network with activation extraction"""
    
    def __init__(self, base_model: FFNetwork):
        super().__init__()
        self.model = base_model
    
    def get_layer_activations(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations from all layers"""
        batch_size = images.size(0)
        images_flat = images.view(batch_size, -1)
        
        x = embed_label(images_flat, labels, self.model.num_classes)
        x = normalize(x)
        
        activations = {'input': images_flat.clone()}
        
        for i, layer in enumerate(self.model.layers):
            x = layer.get_output(x)
            activations[f'layer_{i}'] = x.clone()
            x = normalize(x)
        
        return activations


class BPNetworkWithHooks(nn.Module):
    """BP Network with activation extraction"""
    
    def __init__(self, layer_sizes: List[int], lr: float = 0.001):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
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
        for layer in self.layers:
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


def collect_features_and_labels(model, dataloader, device: str, 
                                 model_type: str = 'ff') -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Collect features from all layers and corresponding labels.
    """
    all_activations = {}
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Collecting {model_type.upper()} features"):
            images, labels = images.to(device), labels.to(device)
            
            if model_type == 'ff':
                batch_acts = model.get_layer_activations(images, labels)
            else:
                batch_acts = model.get_layer_activations(images)
            
            for name, act in batch_acts.items():
                if name not in all_activations:
                    all_activations[name] = []
                all_activations[name].append(act.cpu().numpy())
            
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate
    features = {name: np.concatenate(acts, axis=0) for name, acts in all_activations.items()}
    labels = np.concatenate(all_labels, axis=0)
    
    return features, labels


def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       max_iter: int = 1000) -> Tuple[float, float]:
    """
    Train a linear probe (logistic regression) and return train/test accuracy.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    clf = LogisticRegression(
        max_iter=max_iter, 
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    
    return train_acc, test_acc


def run_linear_probing(train_features: Dict[str, np.ndarray], 
                       train_labels: np.ndarray,
                       test_features: Dict[str, np.ndarray],
                       test_labels: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Run linear probing on all layers.
    
    Returns:
        Dict mapping layer names to (train_acc, test_acc) tuples
    """
    results = {}
    
    layer_names = [k for k in train_features.keys() if k.startswith('layer') or k == 'input']
    
    for name in tqdm(layer_names, desc="Linear probing"):
        X_train = train_features[name]
        X_test = test_features[name]
        
        train_acc, test_acc = train_linear_probe(X_train, train_labels, X_test, test_labels)
        results[name] = (train_acc, test_acc)
        print(f"  {name}: train={train_acc:.4f}, test={test_acc:.4f}")
    
    return results


def plot_linear_probing_comparison(ff_results: Dict[str, Tuple[float, float]],
                                   bp_results: Dict[str, Tuple[float, float]],
                                   save_path: str):
    """Plot linear probing accuracy comparison"""
    # Get layer names (excluding input)
    ff_layers = sorted([k for k in ff_results.keys() if k.startswith('layer')],
                       key=lambda x: int(x.split('_')[1]))
    bp_layers = sorted([k for k in bp_results.keys() if k.startswith('layer')],
                       key=lambda x: int(x.split('_')[1]))
    
    # Ensure same number of layers
    n_layers = min(len(ff_layers), len(bp_layers))
    ff_layers = ff_layers[:n_layers]
    bp_layers = bp_layers[:n_layers]
    
    # Extract test accuracies
    ff_test_accs = [ff_results[l][1] * 100 for l in ff_layers]
    bp_test_accs = [bp_results[l][1] * 100 for l in bp_layers]
    
    # Also add input layer if available
    if 'input' in ff_results:
        ff_test_accs.insert(0, ff_results['input'][1] * 100)
        bp_test_accs.insert(0, bp_results['input'][1] * 100)
        layer_labels = ['Input'] + [f'Layer {i}' for i in range(n_layers)]
    else:
        layer_labels = [f'Layer {i}' for i in range(n_layers)]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(layer_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ff_test_accs, width, label='Forward-Forward', 
                   color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, bp_test_accs, width, label='Backpropagation',
                   color='#3498db', edgecolor='black')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Linear Probe Accuracy (%)', fontsize=12)
    ax.set_title('Linear Separability by Layer: FF vs BP (MNIST)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_layer_degradation(ff_results: Dict[str, Tuple[float, float]],
                           bp_results: Dict[str, Tuple[float, float]],
                           save_path: str):
    """Plot how much accuracy degrades from best layer"""
    ff_layers = sorted([k for k in ff_results.keys() if k.startswith('layer')],
                       key=lambda x: int(x.split('_')[1]))
    bp_layers = sorted([k for k in bp_results.keys() if k.startswith('layer')],
                       key=lambda x: int(x.split('_')[1]))
    
    ff_test_accs = [ff_results[l][1] * 100 for l in ff_layers]
    bp_test_accs = [bp_results[l][1] * 100 for l in bp_layers]
    
    # Calculate degradation from previous layer
    ff_deltas = [0] + [ff_test_accs[i] - ff_test_accs[i-1] for i in range(1, len(ff_test_accs))]
    bp_deltas = [0] + [bp_test_accs[i] - bp_test_accs[i-1] for i in range(1, len(bp_test_accs))]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute accuracy
    ax1 = axes[0]
    x = range(len(ff_layers))
    ax1.plot(x, ff_test_accs, 'o-', label='FF', color='#2ecc71', linewidth=2, markersize=8)
    ax1.plot(x, bp_test_accs, 's-', label='BP', color='#3498db', linewidth=2, markersize=8)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Linear Probe Accuracy (%)', fontsize=12)
    ax1.set_title('Absolute Accuracy by Layer', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xticks(x)
    
    # Delta (change from previous layer)
    ax2 = axes[1]
    ax2.bar([i - 0.2 for i in x], ff_deltas, width=0.4, label='FF', color='#2ecc71', edgecolor='black')
    ax2.bar([i + 0.2 for i in x], bp_deltas, width=0.4, label='BP', color='#3498db', edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Δ Accuracy (%)', fontsize=12)
    ax2.set_title('Change from Previous Layer', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xticks(x)
    
    plt.suptitle('Layer-wise Feature Quality Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_information_flow(ff_results: Dict[str, Tuple[float, float]],
                          bp_results: Dict[str, Tuple[float, float]],
                          save_path: str):
    """Visualize information flow through layers"""
    ff_layers = sorted([k for k in ff_results.keys() if k.startswith('layer')],
                       key=lambda x: int(x.split('_')[1]))
    
    ff_test_accs = [ff_results[l][1] * 100 for l in ff_layers]
    bp_test_accs = [bp_results[l][1] * 100 for l in ff_layers]
    
    # Calculate "information retention" relative to best layer
    ff_best = max(ff_test_accs)
    bp_best = max(bp_test_accs)
    
    ff_retention = [acc / ff_best * 100 for acc in ff_test_accs]
    bp_retention = [acc / bp_best * 100 for acc in bp_test_accs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(ff_layers))
    
    # Fill area between curves
    ax.fill_between(x, ff_retention, bp_retention, alpha=0.3, color='gray')
    ax.plot(x, ff_retention, 'o-', label='FF', color='#e74c3c', linewidth=2, markersize=10)
    ax.plot(x, bp_retention, 's-', label='BP', color='#9b59b6', linewidth=2, markersize=10)
    
    # Add annotations for gaps
    for i in x:
        gap = bp_retention[i] - ff_retention[i]
        if abs(gap) > 1:
            mid = (ff_retention[i] + bp_retention[i]) / 2
            ax.annotate(f'{gap:+.1f}%', xy=(i, mid), fontsize=9, ha='center')
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Information Retention (%)', fontsize=12)
    ax.set_title('Information Flow: FF vs BP\n(Relative to Best Layer Performance)', fontsize=14)
    ax.legend()
    ax.set_ylim(70, 105)
    ax.grid(alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in x])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def train_models(device: str, epochs_ff: int = 10, epochs_bp: int = 10,
                 hidden_sizes: List[int] = [500, 500]):
    """Train FF and BP models"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Train FF
    print("\n" + "="*60)
    print("Training Forward-Forward Network")
    print("="*60)
    
    ff_model = FFNetwork(input_size=784, hidden_sizes=hidden_sizes,
                         num_classes=10, threshold=2.0, lr=0.03)
    ff_model.to(device)
    
    for epoch in range(epochs_ff):
        for images, labels in tqdm(train_loader, desc=f"FF Epoch {epoch+1}/{epochs_ff}"):
            images, labels = images.to(device), labels.to(device)
            ff_model.train_batch(images, labels)
        
        # Evaluate
        ff_model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = ff_model.predict(images)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Epoch {epoch+1}: Test Acc = {correct/total*100:.2f}%")
    
    # Train BP
    print("\n" + "="*60)
    print("Training Backpropagation Network")
    print("="*60)
    
    bp_layer_sizes = [784] + hidden_sizes + [10]
    bp_model = BPNetworkWithHooks(bp_layer_sizes, lr=0.001)
    bp_model.to(device)
    
    for epoch in range(epochs_bp):
        for images, labels in tqdm(train_loader, desc=f"BP Epoch {epoch+1}/{epochs_bp}"):
            images, labels = images.to(device), labels.to(device)
            bp_model.train_step(images, labels)
        
        # Evaluate
        bp_model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = bp_model.predict(images)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Epoch {epoch+1}: Test Acc = {correct/total*100:.2f}%")
    
    return ff_model, bp_model, train_loader, test_loader


def main():
    """Main linear probing pipeline"""
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results_dir = os.path.expanduser('~/Desktop/Rios/ff-research/results')
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Train models
    hidden_sizes = [500, 500, 500]
    ff_model, bp_model, train_loader, test_loader = train_models(
        device, epochs_ff=15, epochs_bp=15, hidden_sizes=hidden_sizes
    )
    
    # Wrap FF model
    ff_wrapped = FFNetworkWithHooks(ff_model)
    
    print("\n" + "="*60)
    print("Collecting Features for Linear Probing")
    print("="*60)
    
    # Collect features
    ff_train_features, train_labels = collect_features_and_labels(
        ff_wrapped, train_loader, device, 'ff'
    )
    ff_test_features, test_labels = collect_features_and_labels(
        ff_wrapped, test_loader, device, 'ff'
    )
    
    bp_train_features, _ = collect_features_and_labels(bp_model, train_loader, device, 'bp')
    bp_test_features, _ = collect_features_and_labels(bp_model, test_loader, device, 'bp')
    
    print("\n" + "="*60)
    print("Running Linear Probing")
    print("="*60)
    
    print("\nFF Network:")
    ff_probe_results = run_linear_probing(ff_train_features, train_labels,
                                          ff_test_features, test_labels)
    
    print("\nBP Network:")
    bp_probe_results = run_linear_probing(bp_train_features, train_labels,
                                          bp_test_features, test_labels)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    plot_linear_probing_comparison(ff_probe_results, bp_probe_results,
                                   os.path.join(viz_dir, 'linear_probe_comparison.png'))
    
    plot_layer_degradation(ff_probe_results, bp_probe_results,
                           os.path.join(viz_dir, 'layer_degradation.png'))
    
    plot_information_flow(ff_probe_results, bp_probe_results,
                          os.path.join(viz_dir, 'information_flow.png'))
    
    # Print summary
    print("\n" + "="*60)
    print("Linear Probing Summary")
    print("="*60)
    
    ff_layers = sorted([k for k in ff_probe_results.keys() if k.startswith('layer')],
                       key=lambda x: int(x.split('_')[1]))
    
    print("\n| Layer | FF Test Acc | BP Test Acc | Gap |")
    print("|-------|-------------|-------------|-----|")
    for layer in ff_layers:
        ff_acc = ff_probe_results[layer][1] * 100
        bp_acc = bp_probe_results[layer][1] * 100
        gap = bp_acc - ff_acc
        print(f"| {layer} | {ff_acc:.2f}% | {bp_acc:.2f}% | {gap:+.2f}% |")
    
    # Find worst layer
    ff_test_accs = {l: ff_probe_results[l][1] for l in ff_layers}
    worst_layer = min(ff_test_accs, key=ff_test_accs.get)
    print(f"\nFF's worst layer: {worst_layer} ({ff_test_accs[worst_layer]*100:.2f}%)")
    
    # Evidence of information breakage
    ff_accs = [ff_probe_results[l][1] * 100 for l in ff_layers]
    bp_accs = [bp_probe_results[l][1] * 100 for l in ff_layers]
    
    ff_drops = [ff_accs[i] - ff_accs[i-1] for i in range(1, len(ff_accs))]
    bp_drops = [bp_accs[i] - bp_accs[i-1] for i in range(1, len(bp_accs))]
    
    print(f"\nInformation flow analysis:")
    print(f"  FF layer-to-layer changes: {[f'{d:+.2f}' for d in ff_drops]}")
    print(f"  BP layer-to-layer changes: {[f'{d:+.2f}' for d in bp_drops]}")
    
    if min(ff_drops) < min(bp_drops) - 1:
        breakage_layer = ff_drops.index(min(ff_drops)) + 1
        print(f"\n⚠️ Evidence of information breakage detected at Layer {breakage_layer}!")
        print(f"   FF drop: {min(ff_drops):.2f}% vs BP drop: {bp_drops[breakage_layer-1]:.2f}%")
    
    # Save results
    import json
    summary = {
        'ff_probe_results': {k: {'train': v[0], 'test': v[1]} for k, v in ff_probe_results.items()},
        'bp_probe_results': {k: {'train': v[0], 'test': v[1]} for k, v in bp_probe_results.items()},
        'ff_worst_layer': worst_layer,
        'ff_worst_acc': ff_test_accs[worst_layer],
    }
    
    with open(os.path.join(results_dir, 'linear_probe_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    print("Done!")
    
    return ff_probe_results, bp_probe_results


if __name__ == '__main__':
    main()
