#!/usr/bin/env python3
"""
Layer Collab Transfer Learning - Full Experiment

Objective: Test if Layer Collab's global information improves transfer learning

Best configuration from layer_collab_comprehensive.json:
- gamma_mode = "all"
- gamma_scale = 0.7
- MNIST accuracy: 91.56%

Experiment Design:
1. Pretrain Standard FF on MNIST (500 epochs/layer, full batch)
2. Pretrain Layer Collab FF on MNIST (500 epochs/layer, full batch)
3. Transfer both to Fashion-MNIST
4. Compare: source accuracy, transfer accuracy, improvement

Output: results/layer_collab_transfer_full.json
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


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_dataset(dataset_name: str, device: torch.device) -> Tuple:
    """Load dataset with full batch."""
    data_dir = Path(__file__).parent / 'data'

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

    train_dataset = dataset_class(str(data_dir), train=True, download=True, transform=transform)
    test_dataset = dataset_class(str(data_dir), train=False, download=True, transform=transform)

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
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        return h.pow(2).mean(dim=1)

    def ff_loss(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor,
                gamma_pos: torch.Tensor = None, gamma_neg: torch.Tensor = None,
                gamma_scale: float = 0.0) -> torch.Tensor:
        if gamma_pos is None:
            gamma_pos = torch.zeros_like(pos_goodness)
        if gamma_neg is None:
            gamma_neg = torch.zeros_like(neg_goodness)

        pos_logit = pos_goodness + gamma_scale * gamma_pos - self.threshold
        neg_logit = neg_goodness + gamma_scale * gamma_neg - self.threshold

        loss = torch.log(1 + torch.exp(torch.cat([
            -pos_logit,
            neg_logit
        ]))).mean()

        return loss


class FFNetwork(nn.Module):
    """Forward-Forward Network with Layer Collaboration support."""

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

    def compute_all_goodness(self, x: torch.Tensor) -> List[torch.Tensor]:
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
                pos_goodness_all = self.compute_all_goodness(x_pos)
                neg_goodness_all = self.compute_all_goodness(x_neg)

                gamma_pos = self.compute_gamma(pos_goodness_all, layer_idx, gamma_mode)
                gamma_neg = self.compute_gamma(neg_goodness_all, layer_idx, gamma_mode)

                h_pos = x_pos
                h_neg = x_neg
                for i in range(layer_idx):
                    h_pos = self.layers[i](h_pos).detach()
                    h_neg = self.layers[i](h_neg).detach()

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
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()

    def get_features(self, x: torch.Tensor, label: int = 0) -> torch.Tensor:
        with torch.no_grad():
            h = overlay_y_on_x(x, torch.full((x.shape[0],), label, dtype=torch.long, device=x.device))
            for layer in self.layers:
                h = layer(h)
            return h


class LinearHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_head(features_train: torch.Tensor, y_train: torch.Tensor,
                      features_test: torch.Tensor, y_test: torch.Tensor,
                      epochs: int = 100, batch_size: int = 256,
                      lr: float = 0.01) -> Dict:
    device = features_train.device
    feature_dim = features_train.shape[1]
    num_classes = int(y_train.max().item()) + 1

    head = LinearHead(feature_dim, num_classes).to(device)
    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
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

        head.train(False)
        with torch.no_grad():
            train_preds = head(features_train).argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()

            test_preds = head(features_test).argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        best_test_acc = max(best_test_acc, test_acc)

        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    return {
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'best_test_acc': best_test_acc,
    }


def run_experiment(seed: int = 42) -> Dict[str, Any]:
    """Run the full Layer Collaboration transfer learning experiment."""

    print("="*70)
    print("LAYER COLLABORATION TRANSFER LEARNING - FULL EXPERIMENT")
    print("="*70)

    # Configuration
    PRETRAIN_EPOCHS = 500
    TRANSFER_EPOCHS = 100
    GAMMA_MODE = 'all'
    GAMMA_SCALE = 0.7
    ARCHITECTURE = [784, 500, 500]

    torch.manual_seed(seed)
    device = get_device()

    print(f"\nDevice: {device}")
    print(f"Pretrain epochs per layer: {PRETRAIN_EPOCHS}")
    print(f"Transfer epochs: {TRANSFER_EPOCHS}")
    print(f"Best Collab config: gamma_mode={GAMMA_MODE}, gamma_scale={GAMMA_SCALE}")
    print(f"Seed: {seed}")

    results = {
        'experiment': 'Layer Collab Transfer Learning - Full',
        'question': 'Does Layer Collab global information improve transfer learning?',
        'config': {
            'pretrain_epochs': PRETRAIN_EPOCHS,
            'transfer_epochs': TRANSFER_EPOCHS,
            'seed': seed,
            'device': str(device),
            'architecture': ARCHITECTURE,
            'best_collab_config': {
                'gamma_mode': GAMMA_MODE,
                'gamma_scale': GAMMA_SCALE
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
    # 1. Standard FF Pretraining
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 1: Standard FF Pretraining on MNIST")
    print("="*60)

    torch.manual_seed(seed)
    ff_standard = FFNetwork(ARCHITECTURE, threshold=2.0, lr=0.03).to(device)

    print("\n  Training...")
    start = time.time()
    ff_standard.train_standard(x_pos, x_neg, PRETRAIN_EPOCHS, verbose=True)
    pretrain_time_standard = time.time() - start

    source_acc_standard = ff_standard.get_accuracy(mnist_test, mnist_y_test)
    print(f"\n  MNIST Test Accuracy (Standard FF): {source_acc_standard*100:.2f}%")
    print(f"  Training time: {pretrain_time_standard:.1f}s")

    results['standard_ff'] = {
        'source_accuracy': source_acc_standard,
        'pretrain_time': pretrain_time_standard
    }

    # ================================================================
    # 2. Layer Collab FF Pretraining
    # ================================================================
    print("\n" + "="*60)
    print(f"PHASE 2: Layer Collab FF Pretraining on MNIST")
    print(f"         (gamma_mode={GAMMA_MODE}, gamma_scale={GAMMA_SCALE})")
    print("="*60)

    torch.manual_seed(seed)
    ff_collab = FFNetwork(ARCHITECTURE, threshold=2.0, lr=0.03).to(device)

    print("\n  Training...")
    start = time.time()
    ff_collab.train_collaborative(x_pos, x_neg, PRETRAIN_EPOCHS,
                                   gamma_mode=GAMMA_MODE, gamma_scale=GAMMA_SCALE, verbose=True)
    pretrain_time_collab = time.time() - start

    source_acc_collab = ff_collab.get_accuracy(mnist_test, mnist_y_test)
    print(f"\n  MNIST Test Accuracy (Layer Collab FF): {source_acc_collab*100:.2f}%")
    print(f"  Training time: {pretrain_time_collab:.1f}s")

    results['layer_collab_ff'] = {
        'source_accuracy': source_acc_collab,
        'pretrain_time': pretrain_time_collab,
        'gamma_mode': GAMMA_MODE,
        'gamma_scale': GAMMA_SCALE
    }

    # ================================================================
    # 3. Transfer Learning - Standard FF
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 3: Transfer Standard FF to Fashion-MNIST")
    print("="*60)

    print("\n  Extracting features...")
    features_train_std = ff_standard.get_features(fmnist_train, label=0)
    features_test_std = ff_standard.get_features(fmnist_test, label=0)

    print("  Training linear head...")
    transfer_std = train_linear_head(
        features_train_std, fmnist_y_train,
        features_test_std, fmnist_y_test,
        epochs=TRANSFER_EPOCHS
    )

    print(f"\n  Transfer Accuracy (Standard FF): {transfer_std['best_test_acc']*100:.2f}%")

    results['standard_ff']['transfer_accuracy'] = transfer_std['best_test_acc']

    # ================================================================
    # 4. Transfer Learning - Layer Collab FF
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 4: Transfer Layer Collab FF to Fashion-MNIST")
    print("="*60)

    print("\n  Extracting features...")
    features_train_collab = ff_collab.get_features(fmnist_train, label=0)
    features_test_collab = ff_collab.get_features(fmnist_test, label=0)

    print("  Training linear head...")
    transfer_collab = train_linear_head(
        features_train_collab, fmnist_y_train,
        features_test_collab, fmnist_y_test,
        epochs=TRANSFER_EPOCHS
    )

    print(f"\n  Transfer Accuracy (Layer Collab FF): {transfer_collab['best_test_acc']*100:.2f}%")

    results['layer_collab_ff']['transfer_accuracy'] = transfer_collab['best_test_acc']

    # ================================================================
    # 5. Random Baseline
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 5: Random Baseline (no pretraining)")
    print("="*60)

    torch.manual_seed(seed)
    ff_random = FFNetwork(ARCHITECTURE, threshold=2.0, lr=0.03).to(device)

    features_train_random = ff_random.get_features(fmnist_train, label=0)
    features_test_random = ff_random.get_features(fmnist_test, label=0)

    print("  Training linear head on random features...")
    transfer_random = train_linear_head(
        features_train_random, fmnist_y_train,
        features_test_random, fmnist_y_test,
        epochs=TRANSFER_EPOCHS
    )

    print(f"\n  Transfer Accuracy (Random): {transfer_random['best_test_acc']*100:.2f}%")

    results['random_baseline'] = {
        'transfer_accuracy': transfer_random['best_test_acc']
    }

    # ================================================================
    # Analysis
    # ================================================================
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    source_improvement = source_acc_collab - source_acc_standard
    transfer_improvement = transfer_collab['best_test_acc'] - transfer_std['best_test_acc']

    std_vs_random = transfer_std['best_test_acc'] - transfer_random['best_test_acc']
    collab_vs_random = transfer_collab['best_test_acc'] - transfer_random['best_test_acc']

    results['analysis'] = {
        'source_accuracy': {
            'standard_ff': source_acc_standard,
            'layer_collab_ff': source_acc_collab,
            'improvement': source_improvement
        },
        'transfer_accuracy': {
            'standard_ff': transfer_std['best_test_acc'],
            'layer_collab_ff': transfer_collab['best_test_acc'],
            'random_baseline': transfer_random['best_test_acc'],
            'layer_collab_improvement': transfer_improvement
        },
        'vs_random_baseline': {
            'standard_ff_vs_random': std_vs_random,
            'layer_collab_ff_vs_random': collab_vs_random
        },
        'key_finding': ''
    }

    print(f"\n{'Metric':<35} {'Standard FF':>12} {'Layer Collab':>12} {'Improvement':>12}")
    print("-"*70)
    print(f"{'Source Accuracy (MNIST)':<35} {source_acc_standard*100:>11.2f}% {source_acc_collab*100:>11.2f}% {source_improvement*100:>+11.2f}%")
    print(f"{'Transfer Accuracy (Fashion-MNIST)':<35} {transfer_std['best_test_acc']*100:>11.2f}% {transfer_collab['best_test_acc']*100:>11.2f}% {transfer_improvement*100:>+11.2f}%")
    print("-"*70)
    print(f"{'Random Baseline':<35} {transfer_random['best_test_acc']*100:>11.2f}%")
    print(f"{'Standard FF vs Random':<35} {std_vs_random*100:>+11.2f}%")
    print(f"{'Layer Collab vs Random':<35} {collab_vs_random*100:>+11.2f}%")

    print("\n" + "-"*70)
    print("KEY FINDINGS:")
    print("-"*70)

    findings = []

    if source_improvement > 0.01:
        findings.append(f"1. Layer Collab IMPROVES source accuracy by {source_improvement*100:.2f}% (confirmed)")
    else:
        findings.append(f"1. Layer Collab has similar source accuracy ({source_improvement*100:+.2f}%)")

    if transfer_improvement > 0.01:
        finding = f"2. Layer Collab IMPROVES transfer accuracy by {transfer_improvement*100:.2f}%"
        findings.append(finding)
        results['analysis']['key_finding'] = "Layer Collab global information DOES improve transfer learning"
    elif transfer_improvement < -0.01:
        finding = f"2. Layer Collab DECREASES transfer accuracy by {abs(transfer_improvement)*100:.2f}%"
        findings.append(finding)
        results['analysis']['key_finding'] = "Layer Collab global information does NOT improve transfer learning"
    else:
        finding = f"2. Layer Collab has similar transfer accuracy ({transfer_improvement*100:+.2f}%)"
        findings.append(finding)
        results['analysis']['key_finding'] = "Layer Collab has no significant effect on transfer learning"

    if collab_vs_random > 0:
        findings.append(f"3. Layer Collab FF beats random baseline by {collab_vs_random*100:.2f}%")
    else:
        findings.append(f"3. Random baseline beats Layer Collab FF by {abs(collab_vs_random)*100:.2f}%")

    if std_vs_random > 0:
        findings.append(f"4. Standard FF beats random baseline by {std_vs_random*100:.2f}%")
    else:
        findings.append(f"4. Random baseline beats Standard FF by {abs(std_vs_random)*100:.2f}%")

    for f in findings:
        print(f)

    results['findings'] = findings

    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (float, int)):
            return obj
        elif hasattr(obj, 'item'):
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

    parser = argparse.ArgumentParser(description='Layer Collab Transfer Learning Full Experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    results = run_experiment(seed=args.seed)

    output_path = str(Path(__file__).parent.parent / 'results' / 'layer_collab_transfer_full.json')
    save_results(results, output_path)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nKey Finding: {results['analysis']['key_finding']}")


if __name__ == '__main__':
    main()
