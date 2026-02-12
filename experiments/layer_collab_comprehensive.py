"""
Comprehensive Layer Collaboration FF Experiment

Tests the Layer Collaboration FF algorithm with different:
1. Gamma values (collaboration strength): 0.1, 0.3, 0.5, 0.7, 1.0
2. Collaboration modes:
   - gamma_all: All layers contribute to gamma
   - gamma_prev: Only previous layers contribute
   - gamma_next: Only next layers contribute

Baseline comparison: Standard FF (~93% on MNIST)

Based on: Lorberbom et al. (2024) "Layer Collaboration in the Forward-Forward Algorithm"
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict, Any
import time
import json
from pathlib import Path
from datetime import datetime
import sys

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def overlay_y_on_x(x: torch.Tensor, y) -> torch.Tensor:
    """
    Replace the first 10 pixels of data [x] with one-hot-encoded label [y].
    CRITICAL: Use x.max() as the label value, not 1.0!
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class CollabFFLayer(nn.Module):
    """FF layer with collaboration support and gamma scaling."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with layer normalization."""
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute goodness - MEAN of squared activations."""
        return x.pow(2).mean(dim=1)

    def ff_loss(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor,
                gamma_pos: torch.Tensor = None, gamma_neg: torch.Tensor = None,
                gamma_scale: float = 1.0) -> torch.Tensor:
        """
        FF loss with scaled layer collaboration.

        p = sigmoid(goodness + gamma_scale * gamma - threshold)
        """
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


class CollabFFNetwork(nn.Module):
    """FF Network with comprehensive collaboration modes."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        self.threshold = threshold

        for d in range(len(dims) - 1):
            self.layers.append(CollabFFLayer(dims[d], dims[d + 1], threshold, lr))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return activations from all layers."""
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations

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

    def compute_gamma(self, goodness_list: List[torch.Tensor], current_layer: int,
                      mode: str = 'all') -> torch.Tensor:
        """
        Compute gamma for a specific layer.

        Modes:
            - 'all': Sum of goodness from all OTHER layers
            - 'prev': Sum of goodness from previous layers only
            - 'next': Sum of goodness from next layers only
        """
        gamma = torch.zeros_like(goodness_list[0])

        for i, g in enumerate(goodness_list):
            if mode == 'all' and i != current_layer:
                gamma = gamma + g
            elif mode == 'prev' and i < current_layer:
                gamma = gamma + g
            elif mode == 'next' and i > current_layer:
                gamma = gamma + g

        return gamma

    def train_layer_collaborative(self, layer_idx: int,
                                   x_pos: torch.Tensor, x_neg: torch.Tensor,
                                   num_epochs: int = 500,
                                   gamma_mode: str = 'all',
                                   gamma_scale: float = 1.0,
                                   verbose: bool = True):
        """Train a single layer with collaboration."""
        layer = self.layers[layer_idx]

        for epoch in range(num_epochs):
            # Compute all goodness values (detached)
            pos_goodness_all = self.compute_all_goodness(x_pos)
            neg_goodness_all = self.compute_all_goodness(x_neg)

            # Compute gamma for this layer
            gamma_pos = self.compute_gamma(pos_goodness_all, layer_idx, gamma_mode)
            gamma_neg = self.compute_gamma(neg_goodness_all, layer_idx, gamma_mode)

            # Forward to this layer's input
            h_pos = x_pos
            h_neg = x_neg
            for i in range(layer_idx):
                h_pos = self.layers[i](h_pos).detach()
                h_neg = self.layers[i](h_neg).detach()

            # Forward through current layer (with gradient)
            out_pos = layer(h_pos)
            out_neg = layer(h_neg)

            # Compute goodness and loss with scaled gamma
            g_pos = layer.goodness(out_pos)
            g_neg = layer.goodness(out_neg)
            loss = layer.ff_loss(g_pos, g_neg, gamma_pos, gamma_neg, gamma_scale)

            # Backward
            layer.opt.zero_grad()
            loss.backward()
            layer.opt.step()

            if verbose and (epoch + 1) % 100 == 0:
                print(f"      Epoch {epoch+1}: loss={loss.item():.4f}, "
                      f"g_pos={g_pos.mean().item():.3f}, "
                      f"g_neg={g_neg.mean().item():.3f}")

    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 500, verbose: bool = True):
        """Standard FF training (no collaboration)."""
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\n    Training layer {i} (Standard FF)...')

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
                    print(f"      Epoch {epoch+1}: loss={loss.item():.4f}, "
                          f"g_pos={g_pos.mean().item():.3f}, "
                          f"g_neg={g_neg.mean().item():.3f}")

            h_pos = layer(h_pos).detach()
            h_neg = layer(h_neg).detach()

    def train_collaborative(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                            epochs_per_layer: int = 500,
                            gamma_mode: str = 'all',
                            gamma_scale: float = 1.0,
                            verbose: bool = True):
        """Collaborative FF training with specified mode and scale."""
        for i in range(len(self.layers)):
            if verbose:
                print(f'\n    Training layer {i} (Collab mode={gamma_mode}, scale={gamma_scale})...')
            self.train_layer_collaborative(
                i, x_pos, x_neg, epochs_per_layer, gamma_mode, gamma_scale, verbose
            )

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels."""
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
        """Compute accuracy."""
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()


def get_mnist_data(device):
    """Load MNIST data."""
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])

    train_loader = DataLoader(
        datasets.MNIST('./data/', train=True, download=True, transform=transform),
        batch_size=50000, shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST('./data/', train=False, download=True, transform=transform),
        batch_size=10000, shuffle=False
    )

    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)

    return x, y, x_te, y_te


def run_single_experiment(
    x: torch.Tensor, y: torch.Tensor,
    x_te: torch.Tensor, y_te: torch.Tensor,
    gamma_mode: str = 'all',
    gamma_scale: float = 1.0,
    epochs: int = 500,
    seed: int = 1234,
    device: torch.device = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run single experiment with specific configuration."""

    torch.manual_seed(seed)

    # Create positive/negative samples
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])

    # Create model
    model = CollabFFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    start_time = time.time()

    if gamma_scale == 0.0:
        # Standard FF (no collaboration)
        model.train_greedy(x_pos, x_neg, epochs, verbose)
    else:
        # Collaborative training
        model.train_collaborative(x_pos, x_neg, epochs, gamma_mode, gamma_scale, verbose)

    train_time = time.time() - start_time

    # Evaluate
    train_acc = model.get_accuracy(x, y)
    test_acc = model.get_accuracy(x_te, y_te)

    return {
        'gamma_mode': gamma_mode,
        'gamma_scale': gamma_scale,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_error': 1 - train_acc,
        'test_error': 1 - test_acc,
        'train_time': train_time
    }


def run_comprehensive_experiment(epochs: int = 500, seed: int = 1234) -> Dict[str, Any]:
    """Run the full comprehensive experiment."""

    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs per layer: {epochs}")
    print(f"Seed: {seed}")

    # Load data
    print("\nLoading MNIST...")
    x, y, x_te, y_te = get_mnist_data(device)
    print(f"Train: {x.shape[0]}, Test: {x_te.shape[0]}")

    # Experiment configurations
    gamma_scales = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    gamma_modes = ['all', 'prev', 'next']

    results = []

    # Run baseline (gamma=0)
    print("\n" + "="*70)
    print("BASELINE: Standard FF (gamma=0)")
    print("="*70)

    baseline_result = run_single_experiment(
        x, y, x_te, y_te,
        gamma_mode='none',
        gamma_scale=0.0,
        epochs=epochs,
        seed=seed,
        device=device,
        verbose=True
    )
    baseline_result['experiment_name'] = 'Standard FF (baseline)'
    results.append(baseline_result)

    print(f"\nBaseline: Train={baseline_result['train_accuracy']*100:.2f}%, "
          f"Test={baseline_result['test_accuracy']*100:.2f}%")

    # Run collaborative experiments
    for gamma_mode in gamma_modes:
        for gamma_scale in gamma_scales:
            if gamma_scale == 0.0:
                continue  # Already tested as baseline

            exp_name = f"Collab ({gamma_mode}, scale={gamma_scale})"
            print("\n" + "="*70)
            print(f"EXPERIMENT: {exp_name}")
            print("="*70)

            result = run_single_experiment(
                x, y, x_te, y_te,
                gamma_mode=gamma_mode,
                gamma_scale=gamma_scale,
                epochs=epochs,
                seed=seed,
                device=device,
                verbose=True
            )
            result['experiment_name'] = exp_name
            results.append(result)

            print(f"\nResult: Train={result['train_accuracy']*100:.2f}%, "
                  f"Test={result['test_accuracy']*100:.2f}%")

    # Compile final results
    final_results = {
        'experiment': 'Layer Collaboration Comprehensive Test',
        'paper': 'Lorberbom et al. (2024) "Layer Collaboration in the Forward-Forward Algorithm"',
        'dataset': 'MNIST',
        'architecture': [784, 500, 500],
        'epochs_per_layer': epochs,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'gamma_scales_tested': gamma_scales,
        'gamma_modes_tested': gamma_modes,
        'results': results,
    }

    # Find best configuration
    best_result = max(results, key=lambda r: r['test_accuracy'])
    baseline_acc = results[0]['test_accuracy']

    final_results['analysis'] = {
        'baseline_test_accuracy': baseline_acc,
        'best_test_accuracy': best_result['test_accuracy'],
        'best_configuration': {
            'gamma_mode': best_result['gamma_mode'],
            'gamma_scale': best_result['gamma_scale']
        },
        'improvement_over_baseline': best_result['test_accuracy'] - baseline_acc,
        'comparison_with_target': {
            'target': 0.93,  # Standard FF target ~93%
            'achieved': best_result['test_accuracy'],
            'difference': best_result['test_accuracy'] - 0.93
        }
    }

    # Summary by gamma mode
    mode_summary = {}
    for mode in gamma_modes:
        mode_results = [r for r in results if r['gamma_mode'] == mode]
        if mode_results:
            best_in_mode = max(mode_results, key=lambda r: r['test_accuracy'])
            mode_summary[mode] = {
                'best_test_accuracy': best_in_mode['test_accuracy'],
                'best_gamma_scale': best_in_mode['gamma_scale']
            }
    final_results['mode_summary'] = mode_summary

    # Summary by gamma scale
    scale_summary = {}
    for scale in gamma_scales:
        if scale == 0.0:
            scale_summary[str(scale)] = {'test_accuracy': baseline_acc}
        else:
            scale_results = [r for r in results if r['gamma_scale'] == scale]
            if scale_results:
                best_for_scale = max(scale_results, key=lambda r: r['test_accuracy'])
                avg_for_scale = sum(r['test_accuracy'] for r in scale_results) / len(scale_results)
                scale_summary[str(scale)] = {
                    'best_test_accuracy': best_for_scale['test_accuracy'],
                    'avg_test_accuracy': avg_for_scale,
                    'best_mode': best_for_scale['gamma_mode']
                }
    final_results['scale_summary'] = scale_summary

    return final_results


def print_summary(results: Dict[str, Any]):
    """Print formatted summary of results."""

    print("\n" + "="*70)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*70)

    print(f"\nBaseline (Standard FF): {results['analysis']['baseline_test_accuracy']*100:.2f}%")
    print(f"Best Collaborative:     {results['analysis']['best_test_accuracy']*100:.2f}%")
    print(f"Improvement:            {results['analysis']['improvement_over_baseline']*100:.2f}%")

    print("\n" + "-"*70)
    print("RESULTS BY GAMMA MODE")
    print("-"*70)
    for mode, data in results['mode_summary'].items():
        print(f"  {mode}: Best={data['best_test_accuracy']*100:.2f}% (scale={data['best_gamma_scale']})")

    print("\n" + "-"*70)
    print("RESULTS BY GAMMA SCALE")
    print("-"*70)
    for scale, data in results['scale_summary'].items():
        if 'avg_test_accuracy' in data:
            print(f"  scale={scale}: Best={data['best_test_accuracy']*100:.2f}%, "
                  f"Avg={data['avg_test_accuracy']*100:.2f}% (best_mode={data['best_mode']})")
        else:
            print(f"  scale={scale}: {data['test_accuracy']*100:.2f}% (baseline)")

    print("\n" + "-"*70)
    print("DETAILED RESULTS TABLE")
    print("-"*70)
    print(f"{'Configuration':<35} {'Train Acc':>10} {'Test Acc':>10} {'Time':>8}")
    print("-"*70)

    for r in sorted(results['results'], key=lambda x: x['test_accuracy'], reverse=True):
        name = r.get('experiment_name', f"{r['gamma_mode']}/{r['gamma_scale']}")
        print(f"{name:<35} {r['train_accuracy']*100:>9.2f}% {r['test_accuracy']*100:>9.2f}% "
              f"{r['train_time']:>7.1f}s")

    print("\n" + "-"*70)
    print("KEY FINDINGS")
    print("-"*70)

    best = results['analysis']['best_configuration']
    target_diff = results['analysis']['comparison_with_target']['difference']

    print(f"1. Best configuration: mode={best['gamma_mode']}, scale={best['gamma_scale']}")

    if target_diff >= 0:
        print(f"2. Exceeded 93% target by {target_diff*100:.2f}%")
    else:
        print(f"2. Below 93% target by {abs(target_diff)*100:.2f}%")

    improvement = results['analysis']['improvement_over_baseline']
    if improvement > 0:
        print(f"3. Layer Collaboration IMPROVES over standard FF by {improvement*100:.2f}%")
    else:
        print(f"3. Layer Collaboration does NOT improve over standard FF")

    print("\n" + "="*70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Layer Collaboration FF Comprehensive Experiment')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Epochs per layer (default: 500)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed (default: 1234)')
    args = parser.parse_args()

    print("="*70)
    print("Layer Collaboration FF - Comprehensive Experiment")
    print("="*70)

    # Run experiment
    results = run_comprehensive_experiment(epochs=args.epochs, seed=args.seed)

    # Print summary
    print_summary(results)

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'layer_collab_comprehensive.json'

    # Convert tensors to serializable format
    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    serializable_results = make_serializable(results)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
