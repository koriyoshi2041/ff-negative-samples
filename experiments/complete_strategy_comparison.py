"""
Complete Strategy Comparison Experiment

Uses the correct Forward-Forward implementation:
1. Goodness = MEAN of squared activations (not sum)
2. Full-batch training (batch_size = 50000 or available train size)
3. Layer-by-layer greedy training

Tests all 10 negative sample strategies:
1. label_embedding - Hinton's original method
2. class_confusion - Wrong labels (embedded)
3. random_noise - Random noise baseline
4. image_mixing - Mix two images
5. self_contrastive - Self-Contrastive FF (SCFF)
6. masking - Random pixel masking
7. layer_wise - Layer-adaptive generation
8. adversarial - Adversarial perturbation (FGSM-style)
9. hard_mining - Hard negative mining
10. mono_forward - No negatives (VICReg-style)

For strategies without label embedding, Linear Probe evaluation is used.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Callable
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import sys

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from negative_strategies import (
    LabelEmbeddingStrategy,
    ClassConfusionEmbeddedStrategy,
    RandomNoiseStrategy,
    ImageMixingStrategy,
    SelfContrastiveStrategy,
    MaskingStrategy,
    LayerWiseStrategy,
    FastAdversarialStrategy,
    HardMiningStrategy,
    MonoForwardStrategy,
)


def get_device() -> torch.device:
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# FF Layer and Network (Correct Implementation)
# ============================================================

class FFLayer(nn.Module):
    """
    Forward-Forward Layer - Correct Implementation.

    Key points:
    - Goodness = MEAN of squared activations (not sum!)
    - Threshold default = 2.0
    - Layer normalization before linear
    """

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

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute goodness - MEAN of squared activations.

        CRITICAL: This must be MEAN, not SUM!
        """
        return h.pow(2).mean(dim=1)

    def train_layer(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                    num_epochs: int = 200, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train this layer to convergence using standard FF loss."""
        iterator = tqdm(range(num_epochs), desc="Training layer", leave=False) if verbose else range(num_epochs)

        for _ in iterator:
            h_pos = self.forward(x_pos)
            h_neg = self.forward(x_neg)

            g_pos = self.goodness(h_pos)
            g_neg = self.goodness(h_neg)

            # FF contrastive loss
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold
            ]))).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

    def train_layer_mono(self, x_pos: torch.Tensor, mono_loss_fn: Callable,
                         num_epochs: int = 200, verbose: bool = True) -> torch.Tensor:
        """Train this layer using mono-forward (no negatives)."""
        iterator = tqdm(range(num_epochs), desc="Training layer (mono)", leave=False) if verbose else range(num_epochs)

        for _ in iterator:
            h_pos = self.forward(x_pos)

            # Mono-forward loss (VICReg-style)
            loss = mono_loss_fn(h_pos)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return self.forward(x_pos).detach()


class FFNetwork(nn.Module):
    """Forward-Forward Network - Correct Implementation."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(FFLayer(dims[d], dims[d + 1], threshold, lr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_all_layer_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get outputs from all layers."""
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs

    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 200, verbose: bool = True):
        """Greedy layer-by-layer training."""
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\n  Training layer {i}...')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg, epochs_per_layer, verbose)

    def train_greedy_mono(self, x_pos: torch.Tensor, mono_loss_fn: Callable,
                          epochs_per_layer: int = 200, verbose: bool = True):
        """Greedy layer-by-layer training without negatives."""
        h_pos = x_pos

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\n  Training layer {i} (mono-forward)...')
            h_pos = layer.train_layer_mono(h_pos, mono_loss_fn, epochs_per_layer, verbose)

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels and picking highest goodness."""
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(num_classes):
            # Overlay label
            h = x.clone()
            h[:, :num_classes] = 0.0
            h[range(batch_size), label] = x.abs().max() if x.numel() > 0 else 1.0

            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))

            total_goodness = sum(goodness)
            goodness_per_label.append(total_goodness.unsqueeze(1))

        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor,
                     num_classes: int = 10) -> float:
        """Compute accuracy using label embedding prediction."""
        predictions = self.predict(x, num_classes)
        return (predictions == y).float().mean().item()


# ============================================================
# Linear Probe for strategies without label embedding
# ============================================================

class LinearProbe(nn.Module):
    """Linear classifier for evaluating learned representations."""

    def __init__(self, in_features: int, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_probe(
    model: FFNetwork,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    Train a linear probe on frozen FF representations.

    Returns:
        (train_accuracy, test_accuracy)
    """
    device = x_train.device

    # Get representations from FF network
    with torch.no_grad():
        # Concatenate all layer outputs
        train_outputs = model.get_all_layer_outputs(x_train)
        test_outputs = model.get_all_layer_outputs(x_test)

        train_features = torch.cat(train_outputs, dim=1)
        test_features = torch.cat(test_outputs, dim=1)

    # Train linear probe
    probe = LinearProbe(train_features.size(1), num_classes=10).to(device)
    optimizer = Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    iterator = tqdm(range(epochs), desc="  Training linear probe", leave=False) if verbose else range(epochs)

    for _ in iterator:
        optimizer.zero_grad()
        logits = probe(train_features)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        train_logits = probe(train_features)
        test_logits = probe(test_features)

        train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()
        test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

    return train_acc, test_acc


# ============================================================
# Data Loading
# ============================================================

def get_mnist_data(batch_size: int = 50000) -> Tuple[DataLoader, DataLoader]:
    """Get MNIST data with full batch."""
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])

    train_loader = DataLoader(
        MNIST('./data/', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(
        MNIST('./data/', train=False, download=True, transform=transform),
        batch_size=10000, shuffle=False
    )

    return train_loader, test_loader


# ============================================================
# Strategy Training Functions
# ============================================================

@dataclass
class StrategyResult:
    """Result from training a single strategy."""
    name: str
    train_acc: float
    test_acc: float
    probe_train_acc: Optional[float]
    probe_test_acc: Optional[float]
    train_time: float
    uses_label_embedding: bool
    description: str


def overlay_label(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Embed label into first 10 pixels (Hinton's method)."""
    x_ = x.clone()
    x_[:, :10] = 0.0
    x_[range(x.shape[0]), y] = x.abs().max() if x.numel() > 0 else 1.0
    return x_


def train_strategy(
    strategy_name: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    epochs_per_layer: int = 200,
    verbose: bool = True
) -> StrategyResult:
    """Train FF network with a specific strategy and evaluate."""

    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name.upper()}")
    print(f"{'='*60}")

    start_time = time.time()

    # Default architecture
    dims = [784, 500, 500]
    model = FFNetwork(dims, threshold=2.0, lr=0.03).to(device)

    # Strategy-specific training
    uses_label_embedding = True
    description = ""

    if strategy_name == 'label_embedding':
        # Hinton's original method
        description = "Hinton's original: embed wrong label in first pixels"
        strategy = LabelEmbeddingStrategy(num_classes=10)

        x_pos = strategy.create_positive(x_train, y_train)
        x_neg = strategy.generate(x_train, y_train)

        model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)

    elif strategy_name == 'class_confusion':
        # Wrong label embedded
        description = "Correct image + wrong label embedded"
        strategy = ClassConfusionEmbeddedStrategy(num_classes=10, confusion_mode='random')

        x_pos = strategy.create_positive(x_train, y_train)
        x_neg = strategy.generate(x_train, y_train)

        model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)

    elif strategy_name == 'random_noise':
        # Random noise as negatives
        description = "Matched-statistics random noise as negatives"
        strategy = RandomNoiseStrategy(num_classes=10, noise_type='matched')

        x_pos = overlay_label(x_train, y_train)
        x_neg = strategy.generate(x_train, y_train)

        model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)

    elif strategy_name == 'image_mixing':
        # Mix two images
        description = "Mix two different images (MixUp-style)"
        uses_label_embedding = False
        strategy = ImageMixingStrategy(num_classes=10, alpha_range=(0.3, 0.7))

        # For image mixing, use raw images without label embedding
        x_pos = x_train.clone()
        x_neg = strategy.generate(x_train, y_train)

        model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)

    elif strategy_name == 'self_contrastive':
        # SCFF: Self-Contrastive FF
        description = "Self-Contrastive FF: [x||x] vs [x||x']"
        uses_label_embedding = False

        # SCFF uses concatenated input (2x dimension)
        # Need special handling - use standard network but with concatenated input
        strategy = SelfContrastiveStrategy(num_classes=10, use_augmentation=True, noise_std=0.1)
        strategy.train()

        # Get positive and negative pairs
        x_pos = strategy.create_positive(x_train, y_train)  # Shape: (B, 2*D)
        x_neg = strategy.generate(x_train, y_train)  # Shape: (B, 2*D)

        # Create model with doubled input dimension
        model = FFNetwork([784 * 2, 500, 500], threshold=2.0, lr=0.03).to(device)
        model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)

    elif strategy_name == 'masking':
        # Random masking
        description = "Random pixel masking (50% mask ratio)"
        uses_label_embedding = False
        strategy = MaskingStrategy(num_classes=10, mask_ratio=0.5, mask_mode='zero')

        x_pos = x_train.clone()
        x_neg = strategy.generate(x_train, y_train)

        model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)

    elif strategy_name == 'layer_wise':
        # Layer-adaptive generation
        description = "Layer-adaptive: simple->structural->semantic"
        strategy = LayerWiseStrategy(num_classes=10, num_layers=2, use_label_embedding=True)

        # Train each layer with its own strategy
        h_pos = strategy.create_positive(x_train, y_train)

        for layer_idx, layer in enumerate(model.layers):
            if verbose:
                print(f'\n  Training layer {layer_idx} (layer-wise)...')

            strategy.set_layer(layer_idx)
            h_neg = strategy.generate(x_train, y_train, layer_idx=layer_idx)

            h_pos_out, h_neg_out = layer.train_layer(h_pos, h_neg, epochs_per_layer, verbose)
            h_pos = h_pos_out
            # h_neg is regenerated each layer

    elif strategy_name == 'adversarial':
        # Adversarial perturbation (FGSM-style)
        description = "Fast adversarial (FGSM-style) perturbation"
        uses_label_embedding = False
        strategy = FastAdversarialStrategy(num_classes=10, epsilon=0.1)

        # Train layer by layer with adversarial negatives
        h_pos = x_train.clone()

        for layer_idx, layer in enumerate(model.layers):
            if verbose:
                print(f'\n  Training layer {layer_idx} (adversarial)...')

            # Set model for gradient computation
            strategy.set_model(layer, goodness_fn=lambda x: x.pow(2).mean(dim=1))

            h_neg = strategy.generate(h_pos, y_train, model=layer)

            h_pos_out, h_neg_out = layer.train_layer(h_pos, h_neg, epochs_per_layer, verbose)
            h_pos = h_pos_out

    elif strategy_name == 'hard_mining':
        # Hard negative mining
        description = "Select hardest negatives from candidate pool"
        strategy = HardMiningStrategy(num_classes=10, mining_mode='goodness', pool_size=128)

        x_pos = overlay_label(x_train, y_train)

        # Train layer by layer with hard mining
        h_pos = x_pos.clone()

        for layer_idx, layer in enumerate(model.layers):
            if verbose:
                print(f'\n  Training layer {layer_idx} (hard mining)...')

            strategy.set_model(layer, goodness_fn=lambda x: x.pow(2).mean(dim=1))

            h_neg = strategy.generate(h_pos, y_train, model=layer)

            h_pos_out, h_neg_out = layer.train_layer(h_pos, h_neg, epochs_per_layer, verbose)
            h_pos = h_pos_out

    elif strategy_name == 'mono_forward':
        # No negatives - VICReg-style
        description = "No negatives: VICReg-style variance+decorrelation"
        uses_label_embedding = False
        strategy = MonoForwardStrategy(
            num_classes=10,
            loss_type='vicreg',
            lambda_variance=25.0,
            lambda_covariance=1.0,
            use_label_embedding=False
        )

        x_pos = strategy.create_positive(x_train, y_train)

        def mono_loss_fn(activations):
            return strategy.compute_mono_loss(activations, y_train)

        model.train_greedy_mono(x_pos, mono_loss_fn, epochs_per_layer, verbose)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    train_time = time.time() - start_time

    # Evaluate
    if uses_label_embedding and strategy_name != 'self_contrastive':
        train_acc = model.get_accuracy(x_train, y_train)
        test_acc = model.get_accuracy(x_test, y_test)
        probe_train_acc = None
        probe_test_acc = None
    else:
        # For strategies without label embedding, use linear probe
        train_acc = 0.0  # Can't compute directly
        test_acc = 0.0

        # Prepare data for linear probe
        if strategy_name == 'self_contrastive':
            # SCFF uses concatenated input
            x_train_probe = torch.cat([x_train, x_train], dim=1)
            x_test_probe = torch.cat([x_test, x_test], dim=1)
        else:
            x_train_probe = x_train
            x_test_probe = x_test

        probe_train_acc, probe_test_acc = train_linear_probe(
            model, x_train_probe, y_train, x_test_probe, y_test,
            epochs=100, lr=0.01, verbose=verbose
        )

    result = StrategyResult(
        name=strategy_name,
        train_acc=train_acc,
        test_acc=test_acc,
        probe_train_acc=probe_train_acc,
        probe_test_acc=probe_test_acc,
        train_time=train_time,
        uses_label_embedding=uses_label_embedding,
        description=description
    )

    # Print results
    print(f"\n  Results for {strategy_name}:")
    if uses_label_embedding and strategy_name != 'self_contrastive':
        print(f"    Train Acc: {train_acc*100:.2f}%")
        print(f"    Test Acc:  {test_acc*100:.2f}%")
    else:
        print(f"    Linear Probe Train Acc: {probe_train_acc*100:.2f}%")
        print(f"    Linear Probe Test Acc:  {probe_test_acc*100:.2f}%")
    print(f"    Train Time: {train_time:.1f}s")

    return result


# ============================================================
# Main Experiment
# ============================================================

def run_complete_comparison(
    epochs_per_layer: int = 200,
    seed: int = 1234,
    strategies: Optional[List[str]] = None
) -> Dict:
    """Run complete strategy comparison experiment."""

    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # Load data (full batch)
    train_loader, test_loader = get_mnist_data(batch_size=60000)

    x_train, y_train = next(iter(train_loader))
    x_train, y_train = x_train.to(device), y_train.to(device)

    x_test, y_test = next(iter(test_loader))
    x_test, y_test = x_test.to(device), y_test.to(device)

    print(f"\n{'='*60}")
    print("COMPLETE STRATEGY COMPARISON EXPERIMENT")
    print(f"{'='*60}")
    print(f"Architecture: [784, 500, 500]")
    print(f"Epochs per layer: {epochs_per_layer}")
    print(f"Batch size: {x_train.size(0)} (full batch)")
    print(f"Threshold: 2.0")
    print(f"Learning rate: 0.03")
    print(f"Goodness: MEAN of squared activations")

    # All strategies to test
    all_strategies = [
        'label_embedding',
        'class_confusion',
        'random_noise',
        'image_mixing',
        'self_contrastive',
        'masking',
        'layer_wise',
        'adversarial',
        'hard_mining',
        'mono_forward',
    ]

    if strategies is not None:
        all_strategies = [s for s in all_strategies if s in strategies]

    results = {}

    for strategy_name in all_strategies:
        try:
            result = train_strategy(
                strategy_name,
                x_train, y_train,
                x_test, y_test,
                device,
                epochs_per_layer,
                verbose=True
            )
            results[strategy_name] = asdict(result)
        except Exception as e:
            print(f"\n  ERROR in {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
            results[strategy_name] = {
                'name': strategy_name,
                'error': str(e),
            }

    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Strategy':<20} {'Type':<15} {'Train Acc':<12} {'Test Acc':<12} {'Time (s)':<10}")
    print("-"*80)

    for name, data in results.items():
        if 'error' in data:
            print(f"{name:<20} {'ERROR':<15} {'-':<12} {'-':<12} {'-':<10}")
            continue

        if data['uses_label_embedding']:
            train_str = f"{data['train_acc']*100:.2f}%"
            test_str = f"{data['test_acc']*100:.2f}%"
            type_str = "Label Embed"
        else:
            train_str = f"{data['probe_train_acc']*100:.2f}%" if data['probe_train_acc'] else "-"
            test_str = f"{data['probe_test_acc']*100:.2f}%" if data['probe_test_acc'] else "-"
            type_str = "Linear Probe"

        print(f"{name:<20} {type_str:<15} {train_str:<12} {test_str:<12} {data['train_time']:.1f}")

    # Add metadata
    results['metadata'] = {
        'epochs_per_layer': epochs_per_layer,
        'batch_size': int(x_train.size(0)),
        'architecture': [784, 500, 500],
        'threshold': 2.0,
        'learning_rate': 0.03,
        'goodness': 'mean',
        'seed': seed,
        'device': str(device),
    }

    return results


def save_results(results: Dict, output_path: str):
    """Save results to JSON."""
    # Convert any non-serializable values
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    serializable_results = make_serializable(results)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Complete FF Strategy Comparison')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Epochs per layer (default 200)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--strategies', type=str, nargs='+', default=None,
                        help='Specific strategies to test (default: all)')
    args = parser.parse_args()

    results = run_complete_comparison(
        epochs_per_layer=args.epochs,
        seed=args.seed,
        strategies=args.strategies
    )

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'complete_strategy_comparison.json'
    save_results(results, str(output_path))

    return results


if __name__ == "__main__":
    main()
