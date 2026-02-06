"""
Fair Comparison Experiment for Forward-Forward Strategies.

This experiment provides a FAIR comparison of all 10 negative sample strategies
by using appropriate evaluation methods:
- Label-based strategies: Standard label embedding evaluation
- Label-free strategies: Linear probe evaluation

All strategies use identical training settings:
- Epochs: 10
- Batch size: 128
- Learning rate: 0.03
- Architecture: 784 -> 500 -> 500
- Threshold: 2.0

Also tests two model architectures:
- ff_correct: Standard FF baseline
- CwC-FF: Channel-wise Competitive FF (no negative samples needed)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import os

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from negative_strategies import (
    StrategyRegistry,
    LabelEmbeddingStrategy,
    ImageMixingStrategy,
    RandomNoiseStrategy,
    ClassConfusionStrategy,
    ClassConfusionEmbeddedStrategy,
    SelfContrastiveStrategy,
    MaskingStrategy,
    LayerWiseStrategy,
    AdversarialStrategy,
    HardMiningStrategy,
    MonoForwardStrategy,
)


# ============================================================
# Configuration
# ============================================================

EXPERIMENT_CONFIG = {
    'epochs': 10,
    'batch_size': 128,
    'lr': 0.03,
    'threshold': 2.0,
    'layer_sizes': [784, 500, 500],
    'num_classes': 10,
    'linear_probe_epochs': 20,
    'linear_probe_lr': 0.01,
}

# Strategies and their properties
STRATEGIES_CONFIG = {
    'label_embedding': {
        'requires_labels': True,
        'uses_negatives': True,
        'assessment_method': 'label_embedding',
        'description': "Hinton's original - embed label in first pixels",
    },
    'class_confusion': {
        'requires_labels': True,
        'uses_negatives': True,
        'assessment_method': 'label_embedding',
        'description': 'Correct image + wrong label embedding',
    },
    'random_noise': {
        'requires_labels': False,
        'uses_negatives': True,
        'assessment_method': 'linear_probe',
        'description': 'Random noise as negatives',
    },
    'image_mixing': {
        'requires_labels': False,
        'uses_negatives': True,
        'assessment_method': 'linear_probe',
        'description': 'Mix two different images',
    },
    'self_contrastive': {
        'requires_labels': False,
        'uses_negatives': True,
        'assessment_method': 'linear_probe',
        'description': 'SCFF - concatenate different images',
    },
    'masking': {
        'requires_labels': False,
        'uses_negatives': True,
        'assessment_method': 'linear_probe',
        'description': 'Random pixel masking',
    },
    'layer_wise': {
        'requires_labels': False,
        'uses_negatives': True,
        'assessment_method': 'linear_probe',
        'description': 'Layer-adaptive perturbations',
    },
    'adversarial': {
        'requires_labels': False,
        'uses_negatives': True,
        'assessment_method': 'linear_probe',
        'description': 'Gradient-based adversarial perturbation',
    },
    'hard_mining': {
        'requires_labels': True,
        'uses_negatives': True,
        'assessment_method': 'label_embedding',
        'description': 'Select hardest negatives from class pool',
    },
    'mono_forward': {
        'requires_labels': True,
        'uses_negatives': False,
        'assessment_method': 'label_embedding',
        'description': 'No negatives - positive only training',
    },
}


# ============================================================
# Model Definitions
# ============================================================

class FFLayer(nn.Module):
    """A single Forward-Forward layer with local learning."""

    def __init__(self, in_features: int, out_features: int, threshold: float = 2.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.optimizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = x / (x.norm(2, dim=1, keepdim=True) + 1e-8)
        return self.relu(self.linear(x_normalized))

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute goodness (mean of squared activations - corrected!)."""
        return (x ** 2).mean(dim=1)

    def ff_loss(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor) -> torch.Tensor:
        """FF loss: push positive goodness above threshold, negative below."""
        loss_pos = torch.log(1 + torch.exp(self.threshold - pos_goodness)).mean()
        loss_neg = torch.log(1 + torch.exp(neg_goodness - self.threshold)).mean()
        return loss_pos + loss_neg

    def mono_loss(self, pos_goodness: torch.Tensor) -> torch.Tensor:
        """Mono-forward loss: only push positive goodness above threshold."""
        return torch.log(1 + torch.exp(self.threshold - pos_goodness)).mean()


class FFNetwork(nn.Module):
    """Multi-layer Forward-Forward Network."""

    def __init__(self, layer_sizes: List[int], threshold: float = 2.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(FFLayer(layer_sizes[i], layer_sizes[i+1], threshold))
        self.feature_dim = layer_sizes[-1]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return activations from all layers."""
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get final layer features for linear probe."""
        for layer in self.layers:
            x = layer(x)
        return x

    def setup_optimizers(self, lr: float = 0.03):
        """Setup optimizers for each layer."""
        for layer in self.layers:
            layer.optimizer = optim.Adam(layer.parameters(), lr=lr)

    def train_step(self, pos_data: torch.Tensor, neg_data: torch.Tensor) -> Dict[str, float]:
        """Train all layers with FF algorithm."""
        losses = {}
        pos_input = pos_data
        neg_input = neg_data

        for i, layer in enumerate(self.layers):
            pos_output = layer(pos_input)
            neg_output = layer(neg_input)

            pos_goodness = layer.goodness(pos_output)
            neg_goodness = layer.goodness(neg_output)

            loss = layer.ff_loss(pos_goodness, neg_goodness)

            layer.optimizer.zero_grad()
            loss.backward()
            layer.optimizer.step()

            losses[f'layer_{i}'] = loss.item()

            pos_input = pos_output.detach()
            neg_input = neg_output.detach()

        return losses

    def train_step_mono(self, pos_data: torch.Tensor) -> Dict[str, float]:
        """Train with mono-forward (no negatives)."""
        losses = {}
        pos_input = pos_data

        for i, layer in enumerate(self.layers):
            pos_output = layer(pos_input)
            pos_goodness = layer.goodness(pos_output)

            loss = layer.mono_loss(pos_goodness)

            layer.optimizer.zero_grad()
            loss.backward()
            layer.optimizer.step()

            losses[f'layer_{i}'] = loss.item()
            pos_input = pos_output.detach()

        return losses


# ============================================================
# Strategy Factory
# ============================================================

def create_strategy(name: str, device: torch.device):
    """Create a strategy instance by name."""
    num_classes = EXPERIMENT_CONFIG['num_classes']

    if name == 'label_embedding':
        return LabelEmbeddingStrategy(num_classes=num_classes, device=device)
    elif name == 'class_confusion':
        return ClassConfusionEmbeddedStrategy(num_classes=num_classes, device=device)
    elif name == 'random_noise':
        return RandomNoiseStrategy(num_classes=num_classes, noise_type='matched', device=device)
    elif name == 'image_mixing':
        return ImageMixingStrategy(num_classes=num_classes, mixing_mode='interpolate', device=device)
    elif name == 'self_contrastive':
        return SelfContrastiveStrategy(num_classes=num_classes, device=device)
    elif name == 'masking':
        return MaskingStrategy(num_classes=num_classes, mask_ratio=0.5, mask_mode='zero', device=device)
    elif name == 'layer_wise':
        return LayerWiseStrategy(num_classes=num_classes, perturbation_scale=0.5, device=device)
    elif name == 'adversarial':
        return AdversarialStrategy(num_classes=num_classes, epsilon=0.1, num_steps=1, device=device)
    elif name == 'hard_mining':
        return HardMiningStrategy(num_classes=num_classes, mining_mode='class', device=device)
    elif name == 'mono_forward':
        return MonoForwardStrategy(num_classes=num_classes, device=device)
    else:
        raise ValueError(f"Unknown strategy: {name}")


# ============================================================
# Training Functions
# ============================================================

def train_epoch_standard(
    model: FFNetwork,
    train_loader: DataLoader,
    strategy,
    device: torch.device,
    use_mono: bool = False
) -> Tuple[Dict[str, float], float]:
    """Train FF network for one epoch using standard FF approach."""
    model.train()

    if hasattr(strategy, 'train'):
        strategy.train()

    total_losses = {f'layer_{i}': 0.0 for i in range(len(model.layers))}
    num_batches = 0

    start_time = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Create positive samples
        pos_data = strategy.create_positive(images, labels)

        if use_mono:
            losses = model.train_step_mono(pos_data)
        else:
            neg_data = strategy.generate(images, labels)
            losses = model.train_step(pos_data, neg_data)

        for key in losses:
            total_losses[key] += losses[key]
        num_batches += 1

    epoch_time = time.time() - start_time
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    return avg_losses, epoch_time


def train_epoch_scff(
    model: FFNetwork,
    train_loader: DataLoader,
    strategy: SelfContrastiveStrategy,
    device: torch.device
) -> Tuple[Dict[str, float], float]:
    """Train FF network for one epoch using SCFF concatenation approach."""
    model.train()
    strategy.train()

    total_losses = {f'layer_{i}': 0.0 for i in range(len(model.layers))}
    num_batches = 0

    start_time = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # SCFF: Create pos/neg using concatenation
        pos_data = strategy.create_positive(images, labels)
        neg_data = strategy.generate(images, labels)

        # For SCFF, we need to handle the 2x input dimension
        # Split and sum approach
        losses = model.train_step(pos_data, neg_data)

        for key in losses:
            total_losses[key] += losses[key]
        num_batches += 1

    epoch_time = time.time() - start_time
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    return avg_losses, epoch_time


# ============================================================
# Assessment Functions
# ============================================================

def assess_label_embedding(
    model: FFNetwork,
    test_loader: DataLoader,
    strategy,
    device: torch.device,
    num_classes: int = 10
) -> float:
    """Assess using label embedding (for label-based strategies)."""
    model.eval()

    if hasattr(strategy, 'set_inference_mode'):
        strategy.set_inference_mode()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            best_goodness = torch.zeros(batch_size, device=device) - float('inf')
            predictions = torch.zeros(batch_size, dtype=torch.long, device=device)

            for candidate_label in range(num_classes):
                candidate_labels = torch.full((batch_size,), candidate_label, device=device)
                pos_data = strategy.create_positive(images, candidate_labels)

                activations = model(pos_data)
                goodness = model.layers[-1].goodness(activations[-1])

                better = goodness > best_goodness
                predictions[better] = candidate_label
                best_goodness[better] = goodness[better]

            correct += (predictions == labels).sum().item()
            total += batch_size

    return correct / total


def assess_linear_probe(
    model: FFNetwork,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    probe_epochs: int = 20,
    probe_lr: float = 0.01
) -> float:
    """
    Assess using linear probe (for label-free strategies).

    This is the FAIR assessment method for self-supervised strategies:
    1. Extract features from frozen FF network
    2. Train a linear classifier on these features
    3. Assess the classifier
    """
    model.eval()

    feature_dim = model.feature_dim

    # Create linear probe
    probe = nn.Linear(feature_dim, num_classes).to(device)
    probe_optimizer = optim.Adam(probe.parameters(), lr=probe_lr)
    criterion = nn.CrossEntropyLoss()

    # Train linear probe
    for epoch in range(probe_epochs):
        probe.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Get features from FF network (no label embedding)
            flat = images.view(images.size(0), -1)

            with torch.no_grad():
                features = model.get_features(flat)

            # Train probe
            logits = probe(features)
            loss = criterion(logits, labels)

            probe_optimizer.zero_grad()
            loss.backward()
            probe_optimizer.step()

    # Assess
    probe.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            flat = images.view(images.size(0), -1)
            features = model.get_features(flat)

            logits = probe(features)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += images.size(0)

    return correct / total


def assess_linear_probe_scff(
    model: FFNetwork,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    probe_epochs: int = 20,
    probe_lr: float = 0.01
) -> float:
    """
    Assess SCFF using linear probe with 2x input format.

    For SCFF, the model was trained with concatenated inputs [x || x],
    so we need to use the same format during assessment.
    """
    model.eval()

    feature_dim = model.feature_dim

    # Create linear probe
    probe = nn.Linear(feature_dim, num_classes).to(device)
    probe_optimizer = optim.Adam(probe.parameters(), lr=probe_lr)
    criterion = nn.CrossEntropyLoss()

    # Train linear probe
    for epoch in range(probe_epochs):
        probe.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # For SCFF: use [x || x] format (same image concatenated with itself)
            flat = images.view(images.size(0), -1)
            scff_input = torch.cat([flat, flat], dim=1)

            with torch.no_grad():
                features = model.get_features(scff_input)

            logits = probe(features)
            loss = criterion(logits, labels)

            probe_optimizer.zero_grad()
            loss.backward()
            probe_optimizer.step()

    # Assess
    probe.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            flat = images.view(images.size(0), -1)
            scff_input = torch.cat([flat, flat], dim=1)

            features = model.get_features(scff_input)
            logits = probe(features)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += images.size(0)

    return correct / total


# ============================================================
# Single Strategy Experiment
# ============================================================

def run_strategy_experiment(
    strategy_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    seed: int = 42
) -> Dict[str, Any]:
    """Run experiment for a single strategy."""

    config = STRATEGIES_CONFIG[strategy_name]

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create strategy
    strategy = create_strategy(strategy_name, device)

    # Determine layer sizes based on strategy
    if strategy_name == 'self_contrastive':
        # SCFF uses 2x input dimension
        layer_sizes = [784 * 2, 500, 500]
    else:
        layer_sizes = EXPERIMENT_CONFIG['layer_sizes'].copy()

    # Create model
    model = FFNetwork(
        layer_sizes=layer_sizes,
        threshold=EXPERIMENT_CONFIG['threshold']
    ).to(device)
    model.setup_optimizers(lr=EXPERIMENT_CONFIG['lr'])

    # Determine training/assessment mode
    use_mono = not config['uses_negatives']
    assessment_method = config['assessment_method']

    # Training history
    history = {
        'losses': [],
        'accuracies': [],
        'epoch_times': [],
    }

    total_time = 0

    print(f"  Training {strategy_name}...", flush=True)

    for epoch in range(EXPERIMENT_CONFIG['epochs']):
        # Train
        if strategy_name == 'self_contrastive':
            losses, epoch_time = train_epoch_scff(model, train_loader, strategy, device)
        else:
            losses, epoch_time = train_epoch_standard(
                model, train_loader, strategy, device, use_mono=use_mono
            )

        total_time += epoch_time

        # Assess
        if assessment_method == 'label_embedding':
            accuracy = assess_label_embedding(
                model, test_loader, strategy, device,
                EXPERIMENT_CONFIG['num_classes']
            )
        else:  # linear_probe
            if strategy_name == 'self_contrastive':
                accuracy = assess_linear_probe_scff(
                    model, train_loader, test_loader, device,
                    EXPERIMENT_CONFIG['num_classes'],
                    EXPERIMENT_CONFIG['linear_probe_epochs'],
                    EXPERIMENT_CONFIG['linear_probe_lr']
                )
            else:
                accuracy = assess_linear_probe(
                    model, train_loader, test_loader, device,
                    EXPERIMENT_CONFIG['num_classes'],
                    EXPERIMENT_CONFIG['linear_probe_epochs'],
                    EXPERIMENT_CONFIG['linear_probe_lr']
                )

        history['losses'].append(sum(losses.values()))
        history['accuracies'].append(accuracy)
        history['epoch_times'].append(epoch_time)

        print(f"    Epoch {epoch+1}/{EXPERIMENT_CONFIG['epochs']} | "
              f"Loss: {sum(losses.values()):.4f} | "
              f"Acc: {accuracy*100:.2f}% | "
              f"Time: {epoch_time:.1f}s", flush=True)

    return {
        'final_accuracy': history['accuracies'][-1],
        'best_accuracy': max(history['accuracies']),
        'accuracies': history['accuracies'],
        'losses': history['losses'],
        'epoch_times': history['epoch_times'],
        'total_time': total_time,
        'assessment_method': assessment_method,
        'requires_labels': config['requires_labels'],
        'uses_negatives': config['uses_negatives'],
        'description': config['description'],
    }


# ============================================================
# CwC-FF Experiment
# ============================================================

def run_cwc_ff_experiment(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    seed: int = 42
) -> Dict[str, Any]:
    """Run experiment for CwC-FF model."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Import CwC-FF
    sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
    from cwc_ff import CwCFFNetwork, create_cwc_mnist

    print("  Training CwC-FF...", flush=True)

    # Create model
    model = create_cwc_mnist(use_cfse=True, loss_type='CwC_CE')
    model.to(device)

    # Update ILT schedule for 10 epochs
    model.ilt_schedule = [[0, 10]] * len(model.layers)

    # Training history
    history = {
        'losses': [],
        'accuracies': [],
        'epoch_times': [],
    }

    total_time = 0

    for epoch in range(EXPERIMENT_CONFIG['epochs']):
        start_time = time.time()

        # Train one epoch
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            h = x
            for layer_idx, layer in enumerate(model.layers):
                h = layer.train_step(h, y)

        epoch_time = time.time() - start_time
        total_time += epoch_time

        # Collect losses
        epoch_loss = sum(layer.get_epoch_loss() for layer in model.layers)

        # Assess
        accuracy = model.get_accuracy(test_loader, device, method='GA')

        history['losses'].append(epoch_loss)
        history['accuracies'].append(accuracy)
        history['epoch_times'].append(epoch_time)

        print(f"    Epoch {epoch+1}/{EXPERIMENT_CONFIG['epochs']} | "
              f"Loss: {epoch_loss:.4f} | "
              f"Acc: {accuracy*100:.2f}% | "
              f"Time: {epoch_time:.1f}s", flush=True)

    return {
        'final_accuracy': history['accuracies'][-1],
        'best_accuracy': max(history['accuracies']),
        'accuracies': history['accuracies'],
        'losses': history['losses'],
        'epoch_times': history['epoch_times'],
        'total_time': total_time,
        'assessment_method': 'channel_wise_ga',
        'requires_labels': True,
        'uses_negatives': False,
        'description': 'Channel-wise Competitive FF (no negatives needed)',
    }


# ============================================================
# Main Experiment Runner
# ============================================================

def run_fair_comparison():
    """Run the complete fair comparison experiment."""

    # Setup device
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Using device: {device}")

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=EXPERIMENT_CONFIG['batch_size'],
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EXPERIMENT_CONFIG['batch_size'],
        shuffle=False
    )

    # Results storage
    all_results = {}

    print("\n" + "=" * 70)
    print("Fair Comparison Experiment - Forward-Forward Strategies")
    print("=" * 70)
    print(f"Config: epochs={EXPERIMENT_CONFIG['epochs']}, "
          f"batch_size={EXPERIMENT_CONFIG['batch_size']}, "
          f"lr={EXPERIMENT_CONFIG['lr']}")
    print(f"Architecture: {EXPERIMENT_CONFIG['layer_sizes']}")
    print(f"Linear probe: epochs={EXPERIMENT_CONFIG['linear_probe_epochs']}, "
          f"lr={EXPERIMENT_CONFIG['linear_probe_lr']}")
    print("=" * 70)

    # Run all 10 strategies
    strategies_to_test = list(STRATEGIES_CONFIG.keys())

    for strategy_name in strategies_to_test:
        print(f"\n[{strategy_name}]", flush=True)

        try:
            result = run_strategy_experiment(
                strategy_name,
                train_loader,
                test_loader,
                device,
                seed=42
            )
            all_results[strategy_name] = result
            print(f"  -> Final: {result['final_accuracy']*100:.2f}% | "
                  f"Best: {result['best_accuracy']*100:.2f}%")
        except Exception as e:
            print(f"  -> ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results[strategy_name] = {
                'error': str(e),
                'final_accuracy': 0.0,
                'best_accuracy': 0.0,
            }

    # Run CwC-FF
    print(f"\n[CwC-FF]", flush=True)
    try:
        cwc_result = run_cwc_ff_experiment(train_loader, test_loader, device, seed=42)
        all_results['cwc_ff'] = cwc_result
        print(f"  -> Final: {cwc_result['final_accuracy']*100:.2f}% | "
              f"Best: {cwc_result['best_accuracy']*100:.2f}%")
    except Exception as e:
        print(f"  -> ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        all_results['cwc_ff'] = {
            'error': str(e),
            'final_accuracy': 0.0,
            'best_accuracy': 0.0,
        }

    return all_results


# ============================================================
# Visualization
# ============================================================

def generate_comparison_chart(results: Dict[str, Any], output_path: Path):
    """Generate comparison chart."""

    # Filter out failed experiments
    valid_results = {k: v for k, v in results.items() if 'error' not in v}

    # Sort by accuracy
    sorted_names = sorted(
        valid_results.keys(),
        key=lambda x: valid_results[x]['final_accuracy'],
        reverse=True
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Bar chart - Accuracy comparison with assessment method coloring
    ax1 = axes[0, 0]

    accuracies = [valid_results[n]['final_accuracy'] * 100 for n in sorted_names]
    colors = []
    for n in sorted_names:
        if 'assessment_method' in valid_results[n]:
            if valid_results[n]['assessment_method'] == 'label_embedding':
                colors.append('#2ecc71')  # Green for label embedding
            elif valid_results[n]['assessment_method'] == 'channel_wise_ga':
                colors.append('#9b59b6')  # Purple for CwC-FF
            else:
                colors.append('#3498db')  # Blue for linear probe
        else:
            colors.append('#95a5a6')  # Gray for unknown

    bars = ax1.barh(range(len(sorted_names)), accuracies, color=colors)
    ax1.set_yticks(range(len(sorted_names)))
    ax1.set_yticklabels(sorted_names)
    ax1.set_xlabel('Test Accuracy (%)')
    ax1.set_title('Fair Comparison: Final Test Accuracy\n(Green=Label Embedding, Blue=Linear Probe, Purple=CwC)')
    ax1.set_xlim(0, 100)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax1.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=9)

    # 2. Training curves
    ax2 = axes[0, 1]
    for name in sorted_names[:6]:  # Top 6 for clarity
        if 'accuracies' in valid_results[name]:
            accs = valid_results[name]['accuracies']
            ax2.plot(range(1, len(accs)+1), [a*100 for a in accs],
                    marker='o', markersize=4, label=name)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Training Curves (Top 6 Strategies)')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)

    # 3. Accuracy by category
    ax3 = axes[1, 0]

    categories = {
        'Label-Based (Embed)': ['label_embedding', 'class_confusion', 'hard_mining', 'mono_forward'],
        'Label-Free (Probe)': ['random_noise', 'image_mixing', 'self_contrastive', 'masking', 'layer_wise', 'adversarial'],
        'Alternative Arch': ['cwc_ff'],
    }

    cat_data = []
    cat_names = []
    for cat_name, strategies in categories.items():
        accs = [valid_results[s]['final_accuracy'] * 100 for s in strategies if s in valid_results]
        if accs:
            cat_data.append(accs)
            cat_names.append(cat_name)

    bp = ax3.boxplot(cat_data, labels=cat_names, patch_artist=True)
    colors_box = ['#2ecc71', '#3498db', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.set_title('Accuracy Distribution by Category')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Training time comparison
    ax4 = axes[1, 1]
    times = [valid_results[n].get('total_time', 0) for n in sorted_names]
    ax4.barh(range(len(sorted_names)), times, color='#e74c3c', alpha=0.7)
    ax4.set_yticks(range(len(sorted_names)))
    ax4.set_yticklabels(sorted_names)
    ax4.set_xlabel('Total Training Time (s)')
    ax4.set_title('Training Time Comparison')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved chart: {output_path}")


def generate_summary_table(results: Dict[str, Any]) -> str:
    """Generate summary table as markdown."""

    lines = []
    lines.append("# Fair Comparison Results")
    lines.append("")
    lines.append("## Experiment Configuration")
    lines.append("")
    lines.append(f"- **Epochs**: {EXPERIMENT_CONFIG['epochs']}")
    lines.append(f"- **Batch Size**: {EXPERIMENT_CONFIG['batch_size']}")
    lines.append(f"- **Learning Rate**: {EXPERIMENT_CONFIG['lr']}")
    lines.append(f"- **Architecture**: {EXPERIMENT_CONFIG['layer_sizes']}")
    lines.append(f"- **Linear Probe Epochs**: {EXPERIMENT_CONFIG['linear_probe_epochs']}")
    lines.append("")

    lines.append("## Results Table")
    lines.append("")
    lines.append("| Rank | Strategy | Accuracy | Assessment Method | Uses Labels | Uses Negatives |")
    lines.append("|------|----------|----------|-------------------|-------------|----------------|")

    # Sort by accuracy
    sorted_items = sorted(
        results.items(),
        key=lambda x: x[1].get('final_accuracy', 0),
        reverse=True
    )

    for rank, (name, data) in enumerate(sorted_items, 1):
        if 'error' in data:
            acc = "ERROR"
            assessment_method = "-"
            uses_labels = "-"
            uses_neg = "-"
        else:
            acc = f"{data['final_accuracy']*100:.2f}%"
            assessment_method = data.get('assessment_method', 'unknown')
            uses_labels = "Yes" if data.get('requires_labels', False) else "No"
            uses_neg = "Yes" if data.get('uses_negatives', True) else "No"

        lines.append(f"| {rank} | {name} | {acc} | {assessment_method} | {uses_labels} | {uses_neg} |")

    lines.append("")
    lines.append("## Key Insights")
    lines.append("")

    # Find best in each category
    label_based = [(n, d) for n, d in sorted_items if d.get('assessment_method') == 'label_embedding']
    probe_based = [(n, d) for n, d in sorted_items if d.get('assessment_method') == 'linear_probe']

    if label_based:
        best_label = label_based[0]
        lines.append(f"- **Best Label-Based Strategy**: {best_label[0]} ({best_label[1]['final_accuracy']*100:.2f}%)")

    if probe_based:
        best_probe = probe_based[0]
        lines.append(f"- **Best Label-Free Strategy**: {best_probe[0]} ({best_probe[1]['final_accuracy']*100:.2f}%)")

    if 'cwc_ff' in results and 'error' not in results['cwc_ff']:
        lines.append(f"- **CwC-FF (No Negatives)**: {results['cwc_ff']['final_accuracy']*100:.2f}%")

    return "\n".join(lines)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main function."""

    print("Starting Fair Comparison Experiment...")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Run experiment
    results = run_fair_comparison()

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON
    json_path = output_dir / 'fair_comparison_results.json'

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    serializable_results = convert_to_serializable(results)
    serializable_results['experiment_config'] = EXPERIMENT_CONFIG
    serializable_results['timestamp'] = datetime.now().isoformat()

    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nSaved results: {json_path}")

    # 2. Generate chart
    chart_path = output_dir / 'fair_comparison.png'
    generate_comparison_chart(results, chart_path)

    # 3. Generate summary
    summary = generate_summary_table(results)
    md_path = output_dir / 'fair_comparison_summary.md'
    with open(md_path, 'w') as f:
        f.write(summary)
    print(f"Saved summary: {md_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    sorted_results = sorted(
        [(n, d) for n, d in results.items() if 'error' not in d],
        key=lambda x: x[1]['final_accuracy'],
        reverse=True
    )

    for rank, (name, data) in enumerate(sorted_results, 1):
        assess_type = data.get('assessment_method', 'unknown')
        print(f"  {rank}. {name}: {data['final_accuracy']*100:.2f}% [{assess_type}]")

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
