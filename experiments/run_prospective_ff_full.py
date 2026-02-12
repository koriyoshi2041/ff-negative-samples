"""
Prospective FF Full Experiment

Based on: Prospective Configuration (Nature Neuroscience, 2024)

Key neuroscience insight:
1. Network first INFERS what neural activity should be after learning
2. Then modifies synaptic weights to CONSOLIDATE this activity change
3. Only needs ONE ITERATION to learn (vs BP needing many)

Full experiment parameters:
- epochs_per_layer: 500
- batch_size: 50000
- Tests: Single-iteration learning, Interference test, MNIST->Fashion transfer

Results saved to: results/prospective_ff_results.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import time
import json
import sys
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.prospective_ff import (
    ProspectiveFFNetwork,
    overlay_y_on_x,
    get_device,
    get_mnist_loaders,
    get_fashion_mnist_loaders
)


# =============================================================================
# Standard FF Network (for comparison)
# =============================================================================

class StandardFFLayer(nn.Module):
    """Standard FF layer without prospective configuration."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        return h.pow(2).mean(dim=1)


class StandardFFNetwork(nn.Module):
    """Standard FF network for comparison."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.dims = dims
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(StandardFFLayer(dims[d], dims[d + 1], threshold, lr))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        activations = []
        h = x
        for layer in self.layers:
            h = layer(h)
            activations.append(h)
        return activations

    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 500, verbose: bool = True):
        """Greedy layer-by-layer training."""
        h_pos, h_neg = x_pos, x_neg

        for l, layer in enumerate(self.layers):
            if verbose:
                print(f'\n  Training layer {l}...')

            for epoch in range(epochs_per_layer):
                out_pos = layer(h_pos)
                out_neg = layer(h_neg)

                g_pos = layer.goodness(out_pos)
                g_neg = layer.goodness(out_neg)

                loss = torch.log(1 + torch.exp(torch.cat([
                    -g_pos + layer.threshold,
                    g_neg - layer.threshold
                ]))).mean()

                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()

                if verbose and (epoch + 1) % 100 == 0:
                    print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, "
                          f"g+={g_pos.mean().item():.3f}, g-={g_neg.mean().item():.3f}")

            h_pos = layer(h_pos).detach()
            h_neg = layer(h_neg).detach()

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(num_classes):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))
            goodness_per_label.append(sum(goodness).unsqueeze(1))

        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()


# =============================================================================
# BP Network (for comparison)
# =============================================================================

class BPNetwork(nn.Module):
    """Standard BP network for baseline comparison."""

    def __init__(self, dims: List[int], lr: float = 0.001):
        super().__init__()
        self.dims = dims
        layers = []
        for d in range(len(dims) - 1):
            layers.append(nn.Linear(dims[d], dims[d + 1]))
            if d < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.opt = Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def train_epoch(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.opt.zero_grad()
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self.opt.step()
        return loss.item()

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            logits = self.forward(x)
            predictions = logits.argmax(dim=1)
            return (predictions == y).float().mean().item()


# =============================================================================
# Experiment Configuration
# =============================================================================

FULL_CONFIG = {
    'epochs_per_layer': 500,
    'batch_size': 50000,
    'dims': [784, 500, 500],
    'threshold': 2.0,
    'lr': 0.03,
    'beta': 0.5,
    'consolidation_lr': 0.01,
    'prospective_iterations': 500,  # Match with epochs_per_layer for fair comparison
    'seed': 42
}


# =============================================================================
# Experiment 1: Single-Iteration Learning
# =============================================================================

def experiment_single_iteration_learning(
    device: torch.device,
    config: Dict,
    num_iterations_list: List[int] = [1, 10, 50, 100, 250, 500]
) -> Dict:
    """
    Test: Can prospective FF learn effectively in fewer iterations?

    Compare learning curves of:
    - Standard FF (with equivalent training steps)
    - Prospective FF (with varying iterations)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Single-Iteration Learning Test")
    print("=" * 70)
    print(f"Config: epochs_per_layer={config['epochs_per_layer']}, "
          f"batch_size={config['batch_size']}")

    torch.manual_seed(config['seed'])

    # Load data
    train_loader, test_loader = get_mnist_loaders(
        train_batch_size=config['batch_size'],
        test_batch_size=10000
    )
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)

    print(f"\nData loaded: train={x.shape[0]}, test={x_te.shape[0]}")

    # Prepare samples
    y_onehot = F.one_hot(y, 10).float()
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0), device=device)
    x_neg = overlay_y_on_x(x, y[rnd])
    y_neg = F.one_hot(y[rnd], 10).float()

    results = {
        'prospective_ff': {},
        'standard_ff': {}
    }

    # Test Prospective FF with different iteration counts
    print("\n--- Prospective FF ---")
    for num_iter in num_iterations_list:
        torch.manual_seed(config['seed'])
        model = ProspectiveFFNetwork(
            dims=config['dims'],
            threshold=config['threshold'],
            lr=config['lr'],
            beta=config['beta'],
            consolidation_lr=config['consolidation_lr']
        ).to(device)

        start_time = time.time()
        history = model.train_prospective_multi_iteration(
            x_pos, x_neg, y_onehot, y_neg,
            num_iterations=num_iter,
            verbose=False
        )
        train_time = time.time() - start_time

        train_acc = model.get_accuracy(x, y)
        test_acc = model.get_accuracy(x_te, y_te)

        results['prospective_ff'][str(num_iter)] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'time': train_time,
            'final_loss': history[-1]['total_loss'] if history else 0
        }

        print(f"  {num_iter:4d} iterations: train={train_acc*100:.2f}%, "
              f"test={test_acc*100:.2f}%, time={train_time:.2f}s")

    # Test Standard FF with equivalent epochs
    print("\n--- Standard FF (for comparison) ---")
    for num_epochs in [100, 250, 500]:
        torch.manual_seed(config['seed'])
        model = StandardFFNetwork(
            config['dims'],
            threshold=config['threshold'],
            lr=config['lr']
        ).to(device)

        start_time = time.time()
        model.train_greedy(x_pos, x_neg, epochs_per_layer=num_epochs, verbose=False)
        train_time = time.time() - start_time

        train_acc = model.get_accuracy(x, y)
        test_acc = model.get_accuracy(x_te, y_te)

        results['standard_ff'][str(num_epochs)] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'time': train_time
        }

        print(f"  {num_epochs:4d} epochs/layer: train={train_acc*100:.2f}%, "
              f"test={test_acc*100:.2f}%, time={train_time:.2f}s")

    return results


# =============================================================================
# Experiment 2: Interference Test (Continual Learning)
# =============================================================================

def experiment_interference_test(
    device: torch.device,
    config: Dict
) -> Dict:
    """
    Test: Does prospective configuration reduce catastrophic forgetting?

    Protocol:
    1. Train on Task A (digits 0-4)
    2. Train on Task B (digits 5-9)
    3. Test on Task A again

    Measure: How much Task A performance drops after learning Task B
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Interference Test (Continual Learning)")
    print("=" * 70)

    torch.manual_seed(config['seed'])

    # Load data (full dataset)
    train_loader, test_loader = get_mnist_loaders(
        train_batch_size=60000,
        test_batch_size=10000
    )
    x_all, y_all = next(iter(train_loader))
    x_all, y_all = x_all.to(device), y_all.to(device)

    x_te_all, y_te_all = next(iter(test_loader))
    x_te_all, y_te_all = x_te_all.to(device), y_te_all.to(device)

    # Split into Task A (0-4) and Task B (5-9)
    mask_a = y_all < 5
    mask_b = y_all >= 5
    mask_a_te = y_te_all < 5
    mask_b_te = y_te_all >= 5

    x_a, y_a = x_all[mask_a], y_all[mask_a]
    x_b, y_b = x_all[mask_b], y_all[mask_b]
    x_a_te, y_a_te = x_te_all[mask_a_te], y_te_all[mask_a_te]
    x_b_te, y_b_te = x_te_all[mask_b_te], y_te_all[mask_b_te]

    print(f"\nTask A: digits 0-4 ({x_a.shape[0]} train, {x_a_te.shape[0]} test)")
    print(f"Task B: digits 5-9 ({x_b.shape[0]} train, {x_b_te.shape[0]} test)")

    results = {}

    # Test Standard FF
    print("\n--- Standard FF ---")
    torch.manual_seed(config['seed'])
    model_std = StandardFFNetwork(
        config['dims'],
        threshold=config['threshold'],
        lr=config['lr']
    ).to(device)

    # Prepare Task A samples
    x_a_pos = overlay_y_on_x(x_a, y_a)
    rnd_a = torch.randperm(x_a.size(0), device=device)
    x_a_neg = overlay_y_on_x(x_a, y_a[rnd_a])

    # Train on Task A
    print("  Training on Task A (0-4)...")
    model_std.train_greedy(x_a_pos, x_a_neg,
                           epochs_per_layer=config['epochs_per_layer'], verbose=False)
    acc_a_after_a_std = model_std.get_accuracy(x_a_te, y_a_te)
    print(f"    Task A accuracy after Task A: {acc_a_after_a_std*100:.2f}%")

    # Prepare Task B samples
    x_b_pos = overlay_y_on_x(x_b, y_b)
    rnd_b = torch.randperm(x_b.size(0), device=device)
    x_b_neg = overlay_y_on_x(x_b, y_b[rnd_b])

    # Train on Task B
    print("  Training on Task B (5-9)...")
    model_std.train_greedy(x_b_pos, x_b_neg,
                           epochs_per_layer=config['epochs_per_layer'], verbose=False)

    acc_a_after_b_std = model_std.get_accuracy(x_a_te, y_a_te)
    acc_b_after_b_std = model_std.get_accuracy(x_b_te, y_b_te)
    forgetting_std = acc_a_after_a_std - acc_a_after_b_std

    print(f"    Task A accuracy after Task B: {acc_a_after_b_std*100:.2f}%")
    print(f"    Task B accuracy after Task B: {acc_b_after_b_std*100:.2f}%")
    print(f"    Forgetting (Task A drop): {forgetting_std*100:.2f}%")

    results['standard_ff'] = {
        'acc_a_after_a': acc_a_after_a_std,
        'acc_a_after_b': acc_a_after_b_std,
        'acc_b_after_b': acc_b_after_b_std,
        'forgetting': forgetting_std
    }

    # Test Prospective FF
    print("\n--- Prospective FF ---")
    torch.manual_seed(config['seed'])
    model_prosp = ProspectiveFFNetwork(
        dims=config['dims'],
        threshold=config['threshold'],
        lr=config['lr'],
        beta=config['beta'],
        consolidation_lr=config['consolidation_lr']
    ).to(device)

    # Prepare Task A samples with labels
    y_a_onehot = F.one_hot(y_a, 10).float()
    y_a_neg = F.one_hot(y_a[rnd_a], 10).float()

    # Train on Task A
    print("  Training on Task A (0-4)...")
    model_prosp.train_prospective_multi_iteration(
        x_a_pos, x_a_neg, y_a_onehot, y_a_neg,
        num_iterations=config['prospective_iterations'],
        verbose=False
    )
    acc_a_after_a_prosp = model_prosp.get_accuracy(x_a_te, y_a_te)
    print(f"    Task A accuracy after Task A: {acc_a_after_a_prosp*100:.2f}%")

    # Prepare Task B samples with labels
    y_b_onehot = F.one_hot(y_b, 10).float()
    y_b_neg = F.one_hot(y_b[rnd_b], 10).float()

    # Train on Task B
    print("  Training on Task B (5-9)...")
    model_prosp.train_prospective_multi_iteration(
        x_b_pos, x_b_neg, y_b_onehot, y_b_neg,
        num_iterations=config['prospective_iterations'],
        verbose=False
    )

    acc_a_after_b_prosp = model_prosp.get_accuracy(x_a_te, y_a_te)
    acc_b_after_b_prosp = model_prosp.get_accuracy(x_b_te, y_b_te)
    forgetting_prosp = acc_a_after_a_prosp - acc_a_after_b_prosp

    print(f"    Task A accuracy after Task B: {acc_a_after_b_prosp*100:.2f}%")
    print(f"    Task B accuracy after Task B: {acc_b_after_b_prosp*100:.2f}%")
    print(f"    Forgetting (Task A drop): {forgetting_prosp*100:.2f}%")

    results['prospective_ff'] = {
        'acc_a_after_a': acc_a_after_a_prosp,
        'acc_a_after_b': acc_a_after_b_prosp,
        'acc_b_after_b': acc_b_after_b_prosp,
        'forgetting': forgetting_prosp
    }

    # Test BP
    print("\n--- Backprop (baseline) ---")
    torch.manual_seed(config['seed'])
    model_bp = BPNetwork([784, 500, 500, 10], lr=0.001).to(device)

    # Train on Task A
    print("  Training on Task A...")
    for epoch in range(100):
        model_bp.train_epoch(x_a, y_a)

    acc_a_after_a_bp = model_bp.get_accuracy(x_a_te, y_a_te)
    print(f"    Task A accuracy after Task A: {acc_a_after_a_bp*100:.2f}%")

    # Train on Task B
    print("  Training on Task B...")
    for epoch in range(100):
        model_bp.train_epoch(x_b, y_b)

    acc_a_after_b_bp = model_bp.get_accuracy(x_a_te, y_a_te)
    acc_b_after_b_bp = model_bp.get_accuracy(x_b_te, y_b_te)
    forgetting_bp = acc_a_after_a_bp - acc_a_after_b_bp

    print(f"    Task A accuracy after Task B: {acc_a_after_b_bp*100:.2f}%")
    print(f"    Task B accuracy after Task B: {acc_b_after_b_bp*100:.2f}%")
    print(f"    Forgetting (Task A drop): {forgetting_bp*100:.2f}%")

    results['backprop'] = {
        'acc_a_after_a': acc_a_after_a_bp,
        'acc_a_after_b': acc_a_after_b_bp,
        'acc_b_after_b': acc_b_after_b_bp,
        'forgetting': forgetting_bp
    }

    return results


# =============================================================================
# Experiment 3: Transfer Learning Test (MNIST -> Fashion-MNIST)
# =============================================================================

def experiment_transfer_learning(
    device: torch.device,
    config: Dict
) -> Dict:
    """
    Test: Does prospective configuration improve transfer learning?

    Protocol:
    1. Pre-train on MNIST
    2. Fine-tune on Fashion-MNIST
    3. Measure transfer efficiency
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Transfer Learning (MNIST -> Fashion-MNIST)")
    print("=" * 70)

    torch.manual_seed(config['seed'])

    # Load MNIST
    mnist_train, mnist_test = get_mnist_loaders(
        train_batch_size=config['batch_size'],
        test_batch_size=10000
    )
    x_mnist, y_mnist = next(iter(mnist_train))
    x_mnist, y_mnist = x_mnist.to(device), y_mnist.to(device)

    # Load Fashion-MNIST
    fmnist_train, fmnist_test = get_fashion_mnist_loaders(
        train_batch_size=config['batch_size'],
        test_batch_size=10000
    )
    x_fmnist, y_fmnist = next(iter(fmnist_train))
    x_fmnist, y_fmnist = x_fmnist.to(device), y_fmnist.to(device)
    x_fmnist_te, y_fmnist_te = next(iter(fmnist_test))
    x_fmnist_te, y_fmnist_te = x_fmnist_te.to(device), y_fmnist_te.to(device)

    print(f"\nMNIST: {x_mnist.shape[0]} samples")
    print(f"Fashion-MNIST: {x_fmnist.shape[0]} train, {x_fmnist_te.shape[0]} test")

    results = {}

    # ==========================================================================
    # Standard FF
    # ==========================================================================
    print("\n--- Standard FF ---")

    # Prepare Fashion-MNIST samples
    x_fmnist_pos = overlay_y_on_x(x_fmnist, y_fmnist)
    rnd = torch.randperm(x_fmnist.size(0), device=device)
    x_fmnist_neg = overlay_y_on_x(x_fmnist, y_fmnist[rnd])

    # Train from scratch on Fashion-MNIST
    torch.manual_seed(config['seed'])
    model_scratch = StandardFFNetwork(
        config['dims'],
        threshold=config['threshold'],
        lr=config['lr']
    ).to(device)

    print("  Training from scratch on Fashion-MNIST...")
    start_time = time.time()
    model_scratch.train_greedy(x_fmnist_pos, x_fmnist_neg,
                               epochs_per_layer=config['epochs_per_layer'], verbose=False)
    scratch_time = time.time() - start_time
    scratch_acc = model_scratch.get_accuracy(x_fmnist_te, y_fmnist_te)
    print(f"    From scratch accuracy: {scratch_acc*100:.2f}% (time: {scratch_time:.2f}s)")

    # Pre-train on MNIST, then fine-tune
    torch.manual_seed(config['seed'])
    model_transfer = StandardFFNetwork(
        config['dims'],
        threshold=config['threshold'],
        lr=config['lr']
    ).to(device)

    x_mnist_pos = overlay_y_on_x(x_mnist, y_mnist)
    rnd = torch.randperm(x_mnist.size(0), device=device)
    x_mnist_neg = overlay_y_on_x(x_mnist, y_mnist[rnd])

    print("  Pre-training on MNIST...")
    model_transfer.train_greedy(x_mnist_pos, x_mnist_neg,
                                epochs_per_layer=config['epochs_per_layer'], verbose=False)

    zero_shot_acc = model_transfer.get_accuracy(x_fmnist_te, y_fmnist_te)
    print(f"    Zero-shot Fashion-MNIST accuracy: {zero_shot_acc*100:.2f}%")

    print("  Fine-tuning on Fashion-MNIST...")
    finetune_epochs = config['epochs_per_layer'] // 5  # 1/5 of pre-training
    start_time = time.time()
    model_transfer.train_greedy(x_fmnist_pos, x_fmnist_neg,
                                epochs_per_layer=finetune_epochs, verbose=False)
    transfer_time = time.time() - start_time
    transfer_acc = model_transfer.get_accuracy(x_fmnist_te, y_fmnist_te)
    transfer_gain = transfer_acc - scratch_acc

    print(f"    After fine-tuning accuracy: {transfer_acc*100:.2f}%")
    print(f"    Transfer gain: {transfer_gain*100:+.2f}%")

    results['standard_ff'] = {
        'scratch_acc': scratch_acc,
        'scratch_time': scratch_time,
        'zero_shot_acc': zero_shot_acc,
        'transfer_acc': transfer_acc,
        'transfer_time': transfer_time,
        'transfer_gain': transfer_gain
    }

    # ==========================================================================
    # Prospective FF
    # ==========================================================================
    print("\n--- Prospective FF ---")

    # Prepare Fashion-MNIST samples with labels
    y_fmnist_onehot = F.one_hot(y_fmnist, 10).float()
    rnd = torch.randperm(x_fmnist.size(0), device=device)
    y_fmnist_neg = F.one_hot(y_fmnist[rnd], 10).float()
    x_fmnist_pos = overlay_y_on_x(x_fmnist, y_fmnist)
    x_fmnist_neg = overlay_y_on_x(x_fmnist, y_fmnist[rnd])

    # Train from scratch
    torch.manual_seed(config['seed'])
    model_scratch = ProspectiveFFNetwork(
        dims=config['dims'],
        threshold=config['threshold'],
        lr=config['lr'],
        beta=config['beta'],
        consolidation_lr=config['consolidation_lr']
    ).to(device)

    print("  Training from scratch on Fashion-MNIST...")
    start_time = time.time()
    model_scratch.train_prospective_multi_iteration(
        x_fmnist_pos, x_fmnist_neg, y_fmnist_onehot, y_fmnist_neg,
        num_iterations=config['prospective_iterations'],
        verbose=False
    )
    scratch_time = time.time() - start_time
    scratch_acc = model_scratch.get_accuracy(x_fmnist_te, y_fmnist_te)
    print(f"    From scratch accuracy: {scratch_acc*100:.2f}% (time: {scratch_time:.2f}s)")

    # Pre-train on MNIST
    torch.manual_seed(config['seed'])
    model_transfer = ProspectiveFFNetwork(
        dims=config['dims'],
        threshold=config['threshold'],
        lr=config['lr'],
        beta=config['beta'],
        consolidation_lr=config['consolidation_lr']
    ).to(device)

    y_mnist_onehot = F.one_hot(y_mnist, 10).float()
    rnd = torch.randperm(x_mnist.size(0), device=device)
    y_mnist_neg = F.one_hot(y_mnist[rnd], 10).float()
    x_mnist_pos = overlay_y_on_x(x_mnist, y_mnist)
    x_mnist_neg = overlay_y_on_x(x_mnist, y_mnist[rnd])

    print("  Pre-training on MNIST...")
    model_transfer.train_prospective_multi_iteration(
        x_mnist_pos, x_mnist_neg, y_mnist_onehot, y_mnist_neg,
        num_iterations=config['prospective_iterations'],
        verbose=False
    )

    zero_shot_acc = model_transfer.get_accuracy(x_fmnist_te, y_fmnist_te)
    print(f"    Zero-shot Fashion-MNIST accuracy: {zero_shot_acc*100:.2f}%")

    print("  Fine-tuning on Fashion-MNIST...")
    finetune_iters = config['prospective_iterations'] // 5
    start_time = time.time()
    model_transfer.train_prospective_multi_iteration(
        x_fmnist_pos, x_fmnist_neg, y_fmnist_onehot, y_fmnist_neg,
        num_iterations=finetune_iters,
        verbose=False
    )
    transfer_time = time.time() - start_time
    transfer_acc = model_transfer.get_accuracy(x_fmnist_te, y_fmnist_te)
    transfer_gain = transfer_acc - scratch_acc

    print(f"    After fine-tuning accuracy: {transfer_acc*100:.2f}%")
    print(f"    Transfer gain: {transfer_gain*100:+.2f}%")

    results['prospective_ff'] = {
        'scratch_acc': scratch_acc,
        'scratch_time': scratch_time,
        'zero_shot_acc': zero_shot_acc,
        'transfer_acc': transfer_acc,
        'transfer_time': transfer_time,
        'transfer_gain': transfer_gain
    }

    return results


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_all_experiments() -> Dict:
    """Run all experiments with full configuration and save results."""
    device = get_device()
    config = FULL_CONFIG

    print("=" * 70)
    print("PROSPECTIVE FF FULL EXPERIMENT")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    all_results = {
        'config': config,
        'device': str(device),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': {}
    }

    total_start = time.time()

    # Experiment 1: Single-iteration learning
    print("\n" + "=" * 70)
    exp1_start = time.time()
    results_1 = experiment_single_iteration_learning(device, config)
    exp1_time = time.time() - exp1_start
    all_results['experiments']['single_iteration'] = {
        'results': results_1,
        'duration_seconds': exp1_time
    }

    # Experiment 2: Interference test
    exp2_start = time.time()
    results_2 = experiment_interference_test(device, config)
    exp2_time = time.time() - exp2_start
    all_results['experiments']['interference'] = {
        'results': results_2,
        'duration_seconds': exp2_time
    }

    # Experiment 3: Transfer learning
    exp3_start = time.time()
    results_3 = experiment_transfer_learning(device, config)
    exp3_time = time.time() - exp3_start
    all_results['experiments']['transfer'] = {
        'results': results_3,
        'duration_seconds': exp3_time
    }

    total_time = time.time() - total_start
    all_results['total_duration_seconds'] = total_time

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print("\n1. Single-Iteration Learning:")
    print("   Prospective FF can learn effectively with fewer iterations.")
    if 'prospective_ff' in results_1:
        best_acc = 0
        best_iter = 0
        for k, v in results_1['prospective_ff'].items():
            if v['test_acc'] > best_acc:
                best_acc = v['test_acc']
                best_iter = k
        print(f"   Best Prospective FF: {best_acc*100:.2f}% at {best_iter} iterations")
    if 'standard_ff' in results_1:
        best_acc = 0
        best_epoch = 0
        for k, v in results_1['standard_ff'].items():
            if v['test_acc'] > best_acc:
                best_acc = v['test_acc']
                best_epoch = k
        print(f"   Best Standard FF: {best_acc*100:.2f}% at {best_epoch} epochs/layer")

    print("\n2. Interference (Continual Learning):")
    print("   Forgetting after learning Task B:")
    for method, res in results_2.items():
        print(f"     {method}: {res['forgetting']*100:.2f}%")

    print("\n3. Transfer Learning (MNIST -> Fashion-MNIST):")
    print("   Transfer gain over from-scratch:")
    for method, res in results_3.items():
        print(f"     {method}: {res['transfer_gain']*100:+.2f}%")

    print(f"\nTotal experiment time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    # Save results
    results_path = Path(__file__).parent.parent / 'results' / 'prospective_ff_results.json'
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return all_results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Starting Prospective FF Full Experiment...")
    print("Based on: Nature Neuroscience 2024 - Prospective Configuration")
    print()

    results = run_all_experiments()
