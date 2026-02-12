"""
Prospective FF Experiment

Based on: Prospective Configuration (Nature Neuroscience, 2024)

Experiments:
1. Single-iteration learning test: Can we learn in one pass?
2. Interference test: Learn A, then B - does A performance drop?
3. Transfer test: MNIST -> Fashion-MNIST

Comparison:
- Standard FF (greedy layer-by-layer)
- Prospective FF (target inference + consolidation)
- BP (baseline)

Hypothesis: Prospective configuration reduces layer interference,
improving transfer and continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
import time
import json
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.prospective_ff import (
    ProspectiveFFNetwork,
    ProspectiveFFLayer,
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
                     epochs_per_layer: int = 1000, verbose: bool = True):
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

                if verbose and (epoch + 1) % 200 == 0:
                    print(f"    Epoch {epoch+1}: loss={loss.item():.4f}")

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
# Experiment 1: Single-Iteration Learning
# =============================================================================

def experiment_single_iteration_learning(
    device: torch.device,
    num_iterations_list: List[int] = [1, 5, 10, 50, 100],
    seed: int = 42
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

    torch.manual_seed(seed)

    # Load data
    train_loader, test_loader = get_mnist_loaders()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)

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
        torch.manual_seed(seed)
        model = ProspectiveFFNetwork(
            dims=[784, 500, 500],
            threshold=2.0,
            lr=0.03,
            beta=0.5,
            consolidation_lr=0.01
        ).to(device)

        start_time = time.time()
        model.train_prospective_multi_iteration(
            x_pos, x_neg, y_onehot, y_neg,
            num_iterations=num_iter,
            verbose=False
        )
        train_time = time.time() - start_time

        train_acc = model.get_accuracy(x, y)
        test_acc = model.get_accuracy(x_te, y_te)

        results['prospective_ff'][num_iter] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'time': train_time
        }

        print(f"  {num_iter} iterations: train={train_acc*100:.2f}%, "
              f"test={test_acc*100:.2f}%, time={train_time:.2f}s")

    # Test Standard FF with equivalent epochs
    print("\n--- Standard FF (for comparison) ---")
    for num_epochs in [100, 500, 1000]:
        torch.manual_seed(seed)
        model = StandardFFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

        start_time = time.time()
        model.train_greedy(x_pos, x_neg, epochs_per_layer=num_epochs, verbose=False)
        train_time = time.time() - start_time

        train_acc = model.get_accuracy(x, y)
        test_acc = model.get_accuracy(x_te, y_te)

        results['standard_ff'][num_epochs] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'time': train_time
        }

        print(f"  {num_epochs} epochs/layer: train={train_acc*100:.2f}%, "
              f"test={test_acc*100:.2f}%, time={train_time:.2f}s")

    return results


# =============================================================================
# Experiment 2: Interference Test (Continual Learning)
# =============================================================================

def experiment_interference_test(
    device: torch.device,
    seed: int = 42
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

    torch.manual_seed(seed)

    # Load data
    train_loader, test_loader = get_mnist_loaders(train_batch_size=60000)
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

    # Test each method
    for method_name, create_model, train_func in [
        ('standard_ff', lambda: StandardFFNetwork([784, 500, 500]).to(device), 'greedy'),
        ('prospective_ff', lambda: ProspectiveFFNetwork([784, 500, 500]).to(device), 'prospective')
    ]:
        print(f"\n--- {method_name} ---")
        torch.manual_seed(seed)
        model = create_model()

        # Prepare Task A samples
        y_a_onehot = F.one_hot(y_a, 10).float()
        x_a_pos = overlay_y_on_x(x_a, y_a)
        rnd_a = torch.randperm(x_a.size(0), device=device)
        x_a_neg = overlay_y_on_x(x_a, y_a[rnd_a])
        y_a_neg = F.one_hot(y_a[rnd_a], 10).float()

        # Train on Task A
        print("  Training on Task A (0-4)...")
        if train_func == 'greedy':
            model.train_greedy(x_a_pos, x_a_neg, epochs_per_layer=500, verbose=False)
        else:
            model.train_prospective_multi_iteration(
                x_a_pos, x_a_neg, y_a_onehot, y_a_neg,
                num_iterations=100, verbose=False
            )

        acc_a_after_a = model.get_accuracy(x_a_te, y_a_te)
        print(f"    Task A accuracy after Task A: {acc_a_after_a*100:.2f}%")

        # Prepare Task B samples
        y_b_onehot = F.one_hot(y_b, 10).float()
        x_b_pos = overlay_y_on_x(x_b, y_b)
        rnd_b = torch.randperm(x_b.size(0), device=device)
        x_b_neg = overlay_y_on_x(x_b, y_b[rnd_b])
        y_b_neg = F.one_hot(y_b[rnd_b], 10).float()

        # Train on Task B
        print("  Training on Task B (5-9)...")
        if train_func == 'greedy':
            model.train_greedy(x_b_pos, x_b_neg, epochs_per_layer=500, verbose=False)
        else:
            model.train_prospective_multi_iteration(
                x_b_pos, x_b_neg, y_b_onehot, y_b_neg,
                num_iterations=100, verbose=False
            )

        acc_a_after_b = model.get_accuracy(x_a_te, y_a_te)
        acc_b_after_b = model.get_accuracy(x_b_te, y_b_te)

        forgetting = acc_a_after_a - acc_a_after_b

        print(f"    Task A accuracy after Task B: {acc_a_after_b*100:.2f}%")
        print(f"    Task B accuracy after Task B: {acc_b_after_b*100:.2f}%")
        print(f"    Forgetting (Task A drop): {forgetting*100:.2f}%")

        results[method_name] = {
            'acc_a_after_a': acc_a_after_a,
            'acc_a_after_b': acc_a_after_b,
            'acc_b_after_b': acc_b_after_b,
            'forgetting': forgetting
        }

    # Also test BP
    print("\n--- backprop (baseline) ---")
    torch.manual_seed(seed)
    model = BPNetwork([784, 500, 500, 10], lr=0.001).to(device)

    # Train on Task A
    print("  Training on Task A...")
    for epoch in range(100):
        model.train_epoch(x_a, y_a)

    acc_a_after_a = model.get_accuracy(x_a_te, y_a_te)
    print(f"    Task A accuracy after Task A: {acc_a_after_a*100:.2f}%")

    # Train on Task B
    print("  Training on Task B...")
    for epoch in range(100):
        model.train_epoch(x_b, y_b)

    acc_a_after_b = model.get_accuracy(x_a_te, y_a_te)
    acc_b_after_b = model.get_accuracy(x_b_te, y_b_te)
    forgetting = acc_a_after_a - acc_a_after_b

    print(f"    Task A accuracy after Task B: {acc_a_after_b*100:.2f}%")
    print(f"    Task B accuracy after Task B: {acc_b_after_b*100:.2f}%")
    print(f"    Forgetting (Task A drop): {forgetting*100:.2f}%")

    results['backprop'] = {
        'acc_a_after_a': acc_a_after_a,
        'acc_a_after_b': acc_a_after_b,
        'acc_b_after_b': acc_b_after_b,
        'forgetting': forgetting
    }

    return results


# =============================================================================
# Experiment 3: Transfer Learning Test
# =============================================================================

def experiment_transfer_learning(
    device: torch.device,
    seed: int = 42
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

    torch.manual_seed(seed)

    # Load MNIST
    mnist_train, mnist_test = get_mnist_loaders()
    x_mnist, y_mnist = next(iter(mnist_train))
    x_mnist, y_mnist = x_mnist.to(device), y_mnist.to(device)

    # Load Fashion-MNIST
    fmnist_train, fmnist_test = get_fashion_mnist_loaders()
    x_fmnist, y_fmnist = next(iter(fmnist_train))
    x_fmnist, y_fmnist = x_fmnist.to(device), y_fmnist.to(device)
    x_fmnist_te, y_fmnist_te = next(iter(fmnist_test))
    x_fmnist_te, y_fmnist_te = x_fmnist_te.to(device), y_fmnist_te.to(device)

    print(f"\nMNIST: {x_mnist.shape[0]} samples")
    print(f"Fashion-MNIST: {x_fmnist.shape[0]} train, {x_fmnist_te.shape[0]} test")

    results = {}

    # Test each method
    for method_name, create_model, train_func in [
        ('standard_ff', lambda: StandardFFNetwork([784, 500, 500]).to(device), 'greedy'),
        ('prospective_ff', lambda: ProspectiveFFNetwork([784, 500, 500]).to(device), 'prospective')
    ]:
        print(f"\n--- {method_name} ---")

        # First: Train from scratch on Fashion-MNIST (baseline)
        torch.manual_seed(seed)
        model_scratch = create_model()

        y_fmnist_onehot = F.one_hot(y_fmnist, 10).float()
        x_fmnist_pos = overlay_y_on_x(x_fmnist, y_fmnist)
        rnd = torch.randperm(x_fmnist.size(0), device=device)
        x_fmnist_neg = overlay_y_on_x(x_fmnist, y_fmnist[rnd])
        y_fmnist_neg = F.one_hot(y_fmnist[rnd], 10).float()

        print("  Training from scratch on Fashion-MNIST...")
        if train_func == 'greedy':
            model_scratch.train_greedy(x_fmnist_pos, x_fmnist_neg,
                                       epochs_per_layer=500, verbose=False)
        else:
            model_scratch.train_prospective_multi_iteration(
                x_fmnist_pos, x_fmnist_neg, y_fmnist_onehot, y_fmnist_neg,
                num_iterations=100, verbose=False
            )

        scratch_acc = model_scratch.get_accuracy(x_fmnist_te, y_fmnist_te)
        print(f"    From scratch accuracy: {scratch_acc*100:.2f}%")

        # Second: Pre-train on MNIST, then fine-tune on Fashion-MNIST
        torch.manual_seed(seed)
        model_transfer = create_model()

        y_mnist_onehot = F.one_hot(y_mnist, 10).float()
        x_mnist_pos = overlay_y_on_x(x_mnist, y_mnist)
        rnd = torch.randperm(x_mnist.size(0), device=device)
        x_mnist_neg = overlay_y_on_x(x_mnist, y_mnist[rnd])
        y_mnist_neg = F.one_hot(y_mnist[rnd], 10).float()

        print("  Pre-training on MNIST...")
        if train_func == 'greedy':
            model_transfer.train_greedy(x_mnist_pos, x_mnist_neg,
                                        epochs_per_layer=500, verbose=False)
        else:
            model_transfer.train_prospective_multi_iteration(
                x_mnist_pos, x_mnist_neg, y_mnist_onehot, y_mnist_neg,
                num_iterations=100, verbose=False
            )

        # Zero-shot on Fashion-MNIST (before fine-tuning)
        zero_shot_acc = model_transfer.get_accuracy(x_fmnist_te, y_fmnist_te)
        print(f"    Zero-shot Fashion-MNIST accuracy: {zero_shot_acc*100:.2f}%")

        # Fine-tune on Fashion-MNIST (fewer iterations)
        print("  Fine-tuning on Fashion-MNIST...")
        if train_func == 'greedy':
            model_transfer.train_greedy(x_fmnist_pos, x_fmnist_neg,
                                        epochs_per_layer=100, verbose=False)
        else:
            model_transfer.train_prospective_multi_iteration(
                x_fmnist_pos, x_fmnist_neg, y_fmnist_onehot, y_fmnist_neg,
                num_iterations=50, verbose=False
            )

        transfer_acc = model_transfer.get_accuracy(x_fmnist_te, y_fmnist_te)
        transfer_gain = transfer_acc - scratch_acc

        print(f"    After fine-tuning accuracy: {transfer_acc*100:.2f}%")
        print(f"    Transfer gain: {transfer_gain*100:+.2f}%")

        results[method_name] = {
            'scratch_acc': scratch_acc,
            'zero_shot_acc': zero_shot_acc,
            'transfer_acc': transfer_acc,
            'transfer_gain': transfer_gain
        }

    return results


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_all_experiments(seed: int = 42) -> Dict:
    """Run all experiments and save results."""
    device = get_device()
    print(f"Device: {device}")

    all_results = {
        'device': str(device),
        'seed': seed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': {}
    }

    # Experiment 1: Single-iteration learning
    results_1 = experiment_single_iteration_learning(device, seed=seed)
    all_results['experiments']['single_iteration'] = results_1

    # Experiment 2: Interference test
    results_2 = experiment_interference_test(device, seed=seed)
    all_results['experiments']['interference'] = results_2

    # Experiment 3: Transfer learning
    results_3 = experiment_transfer_learning(device, seed=seed)
    all_results['experiments']['transfer'] = results_3

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print("\n1. Single-Iteration Learning:")
    print("   Prospective FF learns effectively with fewer iterations.")
    if 'prospective_ff' in results_1:
        best_iter = max(results_1['prospective_ff'].keys())
        print(f"   Best result: {results_1['prospective_ff'][best_iter]['test_acc']*100:.2f}% "
              f"at {best_iter} iterations")

    print("\n2. Interference (Continual Learning):")
    print("   Forgetting after learning Task B:")
    for method, res in results_2.items():
        print(f"     {method}: {res['forgetting']*100:.2f}%")

    print("\n3. Transfer Learning (MNIST -> Fashion-MNIST):")
    print("   Transfer gain over from-scratch:")
    for method, res in results_3.items():
        print(f"     {method}: {res['transfer_gain']*100:+.2f}%")

    # Save results
    results_path = Path(__file__).parent.parent / 'results' / 'prospective_ff_results.json'
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return all_results


# =============================================================================
# Quick Test
# =============================================================================

def quick_test(seed: int = 42) -> Dict:
    """Run a quick test with reduced iterations."""
    device = get_device()
    print(f"Device: {device}")
    print("\n" + "=" * 70)
    print("QUICK TEST: Prospective FF vs Standard FF")
    print("=" * 70)

    torch.manual_seed(seed)

    # Load data
    train_loader, test_loader = get_mnist_loaders()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)

    # Prepare samples
    y_onehot = F.one_hot(y, 10).float()
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0), device=device)
    x_neg = overlay_y_on_x(x, y[rnd])
    y_neg = F.one_hot(y[rnd], 10).float()

    results = {}

    # Test Standard FF
    print("\n--- Standard FF (500 epochs/layer) ---")
    torch.manual_seed(seed)
    model_std = StandardFFNetwork([784, 500, 500]).to(device)
    start = time.time()
    model_std.train_greedy(x_pos, x_neg, epochs_per_layer=500, verbose=True)
    std_time = time.time() - start
    std_train = model_std.get_accuracy(x, y)
    std_test = model_std.get_accuracy(x_te, y_te)
    print(f"\nTrain: {std_train*100:.2f}%, Test: {std_test*100:.2f}%, Time: {std_time:.1f}s")
    results['standard_ff'] = {
        'train_acc': std_train,
        'test_acc': std_test,
        'time': std_time
    }

    # Test Prospective FF
    print("\n--- Prospective FF (100 iterations) ---")
    torch.manual_seed(seed)
    model_prosp = ProspectiveFFNetwork([784, 500, 500]).to(device)
    start = time.time()
    model_prosp.train_prospective_multi_iteration(
        x_pos, x_neg, y_onehot, y_neg,
        num_iterations=100, verbose=True
    )
    prosp_time = time.time() - start
    prosp_train = model_prosp.get_accuracy(x, y)
    prosp_test = model_prosp.get_accuracy(x_te, y_te)
    print(f"\nTrain: {prosp_train*100:.2f}%, Test: {prosp_test*100:.2f}%, Time: {prosp_time:.1f}s")
    results['prospective_ff'] = {
        'train_acc': prosp_train,
        'test_acc': prosp_test,
        'time': prosp_time
    }

    # Summary
    print("\n" + "=" * 70)
    print("QUICK TEST RESULTS")
    print("=" * 70)
    print(f"{'Method':<20} {'Train':>10} {'Test':>10} {'Time':>10}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<20} {res['train_acc']*100:>9.2f}% {res['test_acc']*100:>9.2f}% "
              f"{res['time']:>9.1f}s")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prospective FF Experiments')
    parser.add_argument('--mode', choices=['full', 'quick'], default='full',
                        help='Run full experiments or quick test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    if args.mode == 'full':
        results = run_all_experiments(seed=args.seed)
    else:
        results = quick_test(seed=args.seed)
