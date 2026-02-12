#!/usr/bin/env python3
"""
Progressive Unfreezing Experiment for Forward-Forward Transfer Learning
========================================================================

Core Hypothesis: Since shallow layers learn more general features and deep layers
learn more task-specific features, during transfer we should:
1. First unfreeze deep layers (allow specialization to new task)
2. Gradually unfreeze shallow layers (preserve general features)

Experiment Design:

1. Standard Freezing (baseline):
   - Freeze all FF layers
   - Train only new classification head
   - This is the current approach (~61% on Fashion-MNIST)

2. Progressive Unfreezing (Deep-to-Shallow):
   - Phase 1: Only train classification head (20 epochs)
   - Phase 2: Unfreeze Layer 2 + head (20 epochs)
   - Phase 3: Unfreeze Layer 1 + Layer 2 + head (20 epochs)
   - Phase 4: Full unfreeze (20 epochs)
   - Use decreasing learning rates

3. Reverse Progressive (Shallow-to-Deep, Control):
   - Phase 1: Only train head (20 epochs)
   - Phase 2: Unfreeze Layer 0 + head (20 epochs)
   - Phase 3: Unfreeze Layer 0 + Layer 1 + head (20 epochs)
   - Phase 4: Full unfreeze (20 epochs)
   - Should be worse than Progressive if hypothesis is correct

4. Discriminative Learning Rates:
   - Shallow layers: small lr (0.001) - protect general features
   - Deep layers: large lr (0.01) - allow task adaptation
   - All layers unfrozen from start

Expected:
- Standard frozen: ~61%
- Progressive unfreezing: Should be better (allows feature adaptation)
- Discriminative LR: Alternative approach with similar benefits

Author: Clawd (for Parafee)
Date: 2026-02-09
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from copy import deepcopy


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

def get_dataset(dataset_name: str, device: torch.device) -> Tuple:
    """Load dataset."""
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


def overlay_y_on_x(x: torch.Tensor, y) -> torch.Tensor:
    """Embed label in first 10 pixels using x.max()."""
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


# ============================================================
# Forward-Forward Layer (with freezing support)
# ============================================================

class FFLayer(nn.Module):
    """Forward-Forward Layer with freezing support for transfer learning."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.default_lr = lr
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self._frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with layer normalization."""
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Compute goodness = MEAN of squared activations."""
        return h.pow(2).mean(dim=1)

    def freeze(self):
        """Freeze layer parameters."""
        self._frozen = True
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self, lr: Optional[float] = None):
        """Unfreeze layer parameters with optional new learning rate."""
        self._frozen = False
        for param in self.parameters():
            param.requires_grad = True
        if lr is not None:
            self.opt = optim.Adam(self.parameters(), lr=lr)

    @property
    def is_frozen(self) -> bool:
        return self._frozen


# ============================================================
# Forward-Forward Network
# ============================================================

class FFNetwork(nn.Module):
    """Forward-Forward Network with transfer learning support."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.dims = dims
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(FFLayer(dims[d], dims[d + 1], threshold, lr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 500, verbose: bool = True):
        """Standard FF training (greedy, layer-by-layer)."""
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'    Training layer {i}...')

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
                    print(f"      Epoch {epoch+1}: loss={loss.item():.4f}")

            h_pos = layer(h_pos).detach()
            h_neg = layer(h_neg).detach()

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels."""
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
        """Compute accuracy."""
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()

    def get_features(self, x: torch.Tensor, label: int = 0) -> torch.Tensor:
        """Get features with label embedding."""
        with torch.no_grad():
            h = overlay_y_on_x(x, torch.full((x.shape[0],), label, dtype=torch.long, device=x.device))
            for layer in self.layers:
                h = layer(h)
            return h

    def freeze_all_layers(self):
        """Freeze all FF layers."""
        for layer in self.layers:
            layer.freeze()

    def unfreeze_layers(self, layer_indices: List[int], lr: Optional[float] = None):
        """Unfreeze specific layers."""
        for idx in layer_indices:
            if 0 <= idx < len(self.layers):
                self.layers[idx].unfreeze(lr)

    def set_discriminative_lr(self, lr_schedule: Dict[int, float]):
        """Set different learning rates for different layers."""
        for layer_idx, lr in lr_schedule.items():
            if 0 <= layer_idx < len(self.layers):
                self.layers[layer_idx].unfreeze(lr)


# ============================================================
# Transfer Learning Strategies
# ============================================================

class TransferHead(nn.Module):
    """Classification head for transfer learning."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TransferModel(nn.Module):
    """Combined FF feature extractor + classification head for transfer learning."""

    def __init__(self, ff_model: FFNetwork, num_classes: int):
        super().__init__()
        self.ff_model = ff_model
        self.feature_dim = ff_model.dims[-1]
        self.head = TransferHead(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor, label_embed: int = 0) -> torch.Tensor:
        """Forward pass with label embedding and classification."""
        h = overlay_y_on_x(x, torch.full((x.shape[0],), label_embed, dtype=torch.long, device=x.device))
        for layer in self.ff_model.layers:
            h = layer(h)
        return self.head(h)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable (non-frozen) parameters."""
        params = []
        for layer in self.ff_model.layers:
            if not layer.is_frozen:
                params.extend(layer.parameters())
        params.extend(self.head.parameters())
        return params


def train_transfer_model(model: TransferModel,
                         x_train: torch.Tensor, y_train: torch.Tensor,
                         x_test: torch.Tensor, y_test: torch.Tensor,
                         epochs: int = 20, batch_size: int = 256,
                         lr: float = 0.01, verbose: bool = True) -> Dict:
    """Train transfer model with current freezing configuration."""
    device = x_train.device
    criterion = nn.CrossEntropyLoss()

    # Get trainable parameters
    trainable_params = model.get_trainable_params()
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters!")

    optimizer = optim.Adam(trainable_params, lr=lr)

    history = {'train_acc': [], 'test_acc': [], 'loss': []}

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(x_train))
        epoch_losses = []

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_preds = model(x_train).argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()

            test_preds = model(x_test).argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['loss'].append(sum(epoch_losses) / len(epoch_losses))

        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | Loss: {history['loss'][-1]:.4f} | "
                  f"Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

    return {
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'best_test_acc': max(history['test_acc']),
        'history': history
    }


# ============================================================
# Transfer Strategies
# ============================================================

def strategy_standard_frozen(ff_model: FFNetwork,
                            x_train: torch.Tensor, y_train: torch.Tensor,
                            x_test: torch.Tensor, y_test: torch.Tensor,
                            epochs: int = 80, lr: float = 0.01,
                            verbose: bool = True) -> Dict:
    """
    Strategy 1: Standard Freezing
    - Freeze all FF layers
    - Only train classification head
    """
    if verbose:
        print("\n  [Standard Frozen] Freeze all layers, train head only")

    # Clone model to avoid modifying original
    model = deepcopy(ff_model)
    model.freeze_all_layers()

    transfer_model = TransferModel(model, num_classes=10).to(x_train.device)

    result = train_transfer_model(
        transfer_model, x_train, y_train, x_test, y_test,
        epochs=epochs, lr=lr, verbose=verbose
    )

    result['strategy'] = 'standard_frozen'
    result['frozen_layers'] = list(range(len(model.layers)))
    return result


def strategy_progressive_unfreezing(ff_model: FFNetwork,
                                    x_train: torch.Tensor, y_train: torch.Tensor,
                                    x_test: torch.Tensor, y_test: torch.Tensor,
                                    epochs_per_phase: int = 20,
                                    lr_schedule: List[float] = None,
                                    deep_to_shallow: bool = True,
                                    verbose: bool = True) -> Dict:
    """
    Strategy 2 & 3: Progressive Unfreezing

    deep_to_shallow=True (Strategy 2):
    - Phase 1: Only head (20 epochs)
    - Phase 2: Layer 1 + head (20 epochs)  [for 2-layer network: last layer]
    - Phase 3: Layer 0 + Layer 1 + head (20 epochs) [all layers]

    deep_to_shallow=False (Strategy 3, Reverse):
    - Phase 1: Only head (20 epochs)
    - Phase 2: Layer 0 + head (20 epochs)  [first layer]
    - Phase 3: Layer 0 + Layer 1 + head (20 epochs) [all layers]
    """
    num_layers = len(ff_model.layers)
    direction = "deep-to-shallow" if deep_to_shallow else "shallow-to-deep"

    if verbose:
        print(f"\n  [Progressive Unfreezing ({direction})]")

    if lr_schedule is None:
        lr_schedule = [0.01, 0.005, 0.001]  # Decreasing LR

    # Clone model
    model = deepcopy(ff_model)
    model.freeze_all_layers()

    transfer_model = TransferModel(model, num_classes=10).to(x_train.device)

    # Build unfreezing schedule
    if deep_to_shallow:
        # Deep to shallow: unfreeze from last layer to first
        phases = [
            [],  # Phase 1: head only
        ]
        for i in range(num_layers - 1, -1, -1):
            prev = phases[-1].copy() if phases else []
            prev.append(i)
            phases.append(sorted(prev))
    else:
        # Shallow to deep: unfreeze from first layer to last
        phases = [
            [],  # Phase 1: head only
        ]
        for i in range(num_layers):
            prev = phases[-1].copy() if phases else []
            prev.append(i)
            phases.append(sorted(prev))

    all_history = {'train_acc': [], 'test_acc': [], 'loss': [], 'phases': []}

    for phase_idx, layers_to_unfreeze in enumerate(phases):
        phase_num = phase_idx + 1
        lr = lr_schedule[min(phase_idx, len(lr_schedule) - 1)]

        if verbose:
            if len(layers_to_unfreeze) == 0:
                layer_str = "head only"
            else:
                layer_str = f"layers {layers_to_unfreeze} + head"
            print(f"    Phase {phase_num}: Unfreeze {layer_str} (lr={lr})")

        # Unfreeze specified layers
        for layer_idx in layers_to_unfreeze:
            model.layers[layer_idx].unfreeze(lr)

        # Get trainable params and create optimizer
        trainable_params = transfer_model.get_trainable_params()
        optimizer = optim.Adam(trainable_params, lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs_per_phase):
            transfer_model.train()
            indices = torch.randperm(len(x_train))
            epoch_losses = []

            for i in range(0, len(indices), 256):
                batch_idx = indices[i:i+256]
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]

                optimizer.zero_grad()
                outputs = transfer_model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            transfer_model.eval()
            with torch.no_grad():
                train_preds = transfer_model(x_train).argmax(dim=1)
                train_acc = (train_preds == y_train).float().mean().item()

                test_preds = transfer_model(x_test).argmax(dim=1)
                test_acc = (test_preds == y_test).float().mean().item()

            all_history['train_acc'].append(train_acc)
            all_history['test_acc'].append(test_acc)
            all_history['loss'].append(sum(epoch_losses) / len(epoch_losses))
            all_history['phases'].append(phase_num)

        if verbose:
            print(f"      End of phase {phase_num}: Test Acc = {test_acc*100:.2f}%")

    return {
        'final_train_acc': all_history['train_acc'][-1],
        'final_test_acc': all_history['test_acc'][-1],
        'best_test_acc': max(all_history['test_acc']),
        'strategy': 'progressive_unfreezing_deep_to_shallow' if deep_to_shallow else 'progressive_unfreezing_shallow_to_deep',
        'direction': direction,
        'phases': phases,
        'history': all_history
    }


def strategy_discriminative_lr(ff_model: FFNetwork,
                               x_train: torch.Tensor, y_train: torch.Tensor,
                               x_test: torch.Tensor, y_test: torch.Tensor,
                               epochs: int = 80,
                               shallow_lr: float = 0.001,
                               deep_lr: float = 0.01,
                               head_lr: float = 0.01,
                               verbose: bool = True) -> Dict:
    """
    Strategy 4: Discriminative Learning Rates
    - Shallow layers: small lr (protect general features)
    - Deep layers: large lr (allow task adaptation)
    - All layers unfrozen from start
    """
    if verbose:
        print(f"\n  [Discriminative LR] shallow_lr={shallow_lr}, deep_lr={deep_lr}, head_lr={head_lr}")

    num_layers = len(ff_model.layers)

    # Clone model
    model = deepcopy(ff_model)

    # Set up learning rates: linear interpolation from shallow to deep
    for i, layer in enumerate(model.layers):
        # Linear interpolation
        ratio = i / max(1, num_layers - 1)
        layer_lr = shallow_lr + ratio * (deep_lr - shallow_lr)
        layer.unfreeze(layer_lr)
        if verbose:
            print(f"    Layer {i}: lr={layer_lr:.6f}")

    transfer_model = TransferModel(model, num_classes=10).to(x_train.device)

    # Create parameter groups
    param_groups = []
    for i, layer in enumerate(model.layers):
        ratio = i / max(1, num_layers - 1)
        layer_lr = shallow_lr + ratio * (deep_lr - shallow_lr)
        param_groups.append({'params': list(layer.parameters()), 'lr': layer_lr})

    param_groups.append({'params': list(transfer_model.head.parameters()), 'lr': head_lr})

    optimizer = optim.Adam(param_groups)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'test_acc': [], 'loss': []}

    for epoch in range(epochs):
        transfer_model.train()
        indices = torch.randperm(len(x_train))
        epoch_losses = []

        for i in range(0, len(indices), 256):
            batch_idx = indices[i:i+256]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            outputs = transfer_model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        transfer_model.eval()
        with torch.no_grad():
            train_preds = transfer_model(x_train).argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()

            test_preds = transfer_model(x_test).argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['loss'].append(sum(epoch_losses) / len(epoch_losses))

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | Loss: {history['loss'][-1]:.4f} | "
                  f"Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

    return {
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'best_test_acc': max(history['test_acc']),
        'strategy': 'discriminative_lr',
        'lr_config': {
            'shallow_lr': shallow_lr,
            'deep_lr': deep_lr,
            'head_lr': head_lr
        },
        'history': history
    }


def strategy_full_finetune(ff_model: FFNetwork,
                          x_train: torch.Tensor, y_train: torch.Tensor,
                          x_test: torch.Tensor, y_test: torch.Tensor,
                          epochs: int = 80, lr: float = 0.001,
                          verbose: bool = True) -> Dict:
    """
    Strategy 5: Full Fine-tuning
    - All layers unfrozen with same learning rate
    - Standard fine-tuning approach
    """
    if verbose:
        print(f"\n  [Full Fine-tune] All layers unfrozen, lr={lr}")

    model = deepcopy(ff_model)

    # Unfreeze all layers
    for layer in model.layers:
        layer.unfreeze(lr)

    transfer_model = TransferModel(model, num_classes=10).to(x_train.device)

    result = train_transfer_model(
        transfer_model, x_train, y_train, x_test, y_test,
        epochs=epochs, lr=lr, verbose=verbose
    )

    result['strategy'] = 'full_finetune'
    return result


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(pretrain_epochs: int = 500,
                   transfer_epochs: int = 80,
                   epochs_per_phase: int = 20,
                   seed: int = 42) -> Dict[str, Any]:
    """Run the progressive unfreezing experiment."""

    print("="*70)
    print("PROGRESSIVE UNFREEZING EXPERIMENT")
    print("="*70)

    torch.manual_seed(seed)
    device = get_device()

    print(f"\nDevice: {device}")
    print(f"Pretrain epochs per layer: {pretrain_epochs}")
    print(f"Transfer epochs: {transfer_epochs}")
    print(f"Epochs per phase (progressive): {epochs_per_phase}")
    print(f"Seed: {seed}")

    results = {
        'experiment': 'Progressive Unfreezing for FF Transfer Learning',
        'config': {
            'pretrain_epochs': pretrain_epochs,
            'transfer_epochs': transfer_epochs,
            'epochs_per_phase': epochs_per_phase,
            'seed': seed,
            'device': str(device),
            'architecture': [784, 500, 500]
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

    # Prepare pos/neg samples
    x_pos = overlay_y_on_x(mnist_train, mnist_y_train)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg = overlay_y_on_x(mnist_train, mnist_y_train[rnd])

    # ================================================================
    # Pretrain FF on MNIST
    # ================================================================
    print("\n" + "="*60)
    print("PRETRAINING FF on MNIST")
    print("="*60)

    torch.manual_seed(seed)
    ff_model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    start_time = time.time()
    ff_model.train_greedy(x_pos, x_neg, epochs_per_layer=pretrain_epochs, verbose=True)
    pretrain_time = time.time() - start_time

    source_acc = ff_model.get_accuracy(mnist_test, mnist_y_test)
    print(f"\n  Source (MNIST) Accuracy: {source_acc*100:.2f}%")
    print(f"  Pretraining time: {pretrain_time:.1f}s")

    results['source'] = {
        'dataset': 'mnist',
        'accuracy': source_acc,
        'pretrain_time': pretrain_time
    }

    # ================================================================
    # Strategy 1: Standard Frozen
    # ================================================================
    print("\n" + "="*60)
    print("STRATEGY 1: Standard Frozen (Baseline)")
    print("="*60)

    torch.manual_seed(seed)
    result_1 = strategy_standard_frozen(
        ff_model, fmnist_train, fmnist_y_train, fmnist_test, fmnist_y_test,
        epochs=transfer_epochs, lr=0.01, verbose=True
    )
    results['strategy_1_standard_frozen'] = result_1
    print(f"\n  Best Test Accuracy: {result_1['best_test_acc']*100:.2f}%")

    # ================================================================
    # Strategy 2: Progressive Unfreezing (Deep-to-Shallow)
    # ================================================================
    print("\n" + "="*60)
    print("STRATEGY 2: Progressive Unfreezing (Deep-to-Shallow)")
    print("="*60)

    torch.manual_seed(seed)
    result_2 = strategy_progressive_unfreezing(
        ff_model, fmnist_train, fmnist_y_train, fmnist_test, fmnist_y_test,
        epochs_per_phase=epochs_per_phase,
        lr_schedule=[0.01, 0.005, 0.001],
        deep_to_shallow=True,
        verbose=True
    )
    results['strategy_2_progressive_deep_to_shallow'] = result_2
    print(f"\n  Best Test Accuracy: {result_2['best_test_acc']*100:.2f}%")

    # ================================================================
    # Strategy 3: Reverse Progressive (Shallow-to-Deep) - Control
    # ================================================================
    print("\n" + "="*60)
    print("STRATEGY 3: Reverse Progressive (Shallow-to-Deep) - Control")
    print("="*60)

    torch.manual_seed(seed)
    result_3 = strategy_progressive_unfreezing(
        ff_model, fmnist_train, fmnist_y_train, fmnist_test, fmnist_y_test,
        epochs_per_phase=epochs_per_phase,
        lr_schedule=[0.01, 0.005, 0.001],
        deep_to_shallow=False,
        verbose=True
    )
    results['strategy_3_progressive_shallow_to_deep'] = result_3
    print(f"\n  Best Test Accuracy: {result_3['best_test_acc']*100:.2f}%")

    # ================================================================
    # Strategy 4: Discriminative Learning Rates
    # ================================================================
    print("\n" + "="*60)
    print("STRATEGY 4: Discriminative Learning Rates")
    print("="*60)

    torch.manual_seed(seed)
    result_4 = strategy_discriminative_lr(
        ff_model, fmnist_train, fmnist_y_train, fmnist_test, fmnist_y_test,
        epochs=transfer_epochs,
        shallow_lr=0.001,
        deep_lr=0.01,
        head_lr=0.01,
        verbose=True
    )
    results['strategy_4_discriminative_lr'] = result_4
    print(f"\n  Best Test Accuracy: {result_4['best_test_acc']*100:.2f}%")

    # ================================================================
    # Strategy 5: Full Fine-tuning
    # ================================================================
    print("\n" + "="*60)
    print("STRATEGY 5: Full Fine-tuning")
    print("="*60)

    torch.manual_seed(seed)
    result_5 = strategy_full_finetune(
        ff_model, fmnist_train, fmnist_y_train, fmnist_test, fmnist_y_test,
        epochs=transfer_epochs, lr=0.001,
        verbose=True
    )
    results['strategy_5_full_finetune'] = result_5
    print(f"\n  Best Test Accuracy: {result_5['best_test_acc']*100:.2f}%")

    # ================================================================
    # Random Baseline
    # ================================================================
    print("\n" + "="*60)
    print("BASELINE: Random Initialization (No Pretraining)")
    print("="*60)

    torch.manual_seed(seed)
    ff_random = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    result_random = strategy_full_finetune(
        ff_random, fmnist_train, fmnist_y_train, fmnist_test, fmnist_y_test,
        epochs=transfer_epochs, lr=0.001,
        verbose=True
    )
    results['baseline_random'] = result_random
    print(f"\n  Best Test Accuracy: {result_random['best_test_acc']*100:.2f}%")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nSource (MNIST) Accuracy: {source_acc*100:.2f}%")
    print(f"\n{'Strategy':<50} {'Final':>10} {'Best':>10}")
    print("-"*72)

    strategies = [
        ('1. Standard Frozen (baseline)', result_1),
        ('2. Progressive Unfreezing (Deep-to-Shallow)', result_2),
        ('3. Progressive Unfreezing (Shallow-to-Deep)', result_3),
        ('4. Discriminative Learning Rates', result_4),
        ('5. Full Fine-tuning', result_5),
        ('Random Baseline (no pretrain)', result_random),
    ]

    for name, res in strategies:
        print(f"{name:<50} {res['final_test_acc']*100:>9.2f}% {res['best_test_acc']*100:>9.2f}%")

    # Analysis
    print("\n" + "-"*60)
    print("ANALYSIS")
    print("-"*60)

    best_strategy = max(strategies, key=lambda x: x[1]['best_test_acc'])
    baseline_acc = result_1['best_test_acc']

    print(f"\nBest strategy: {best_strategy[0]}")
    print(f"Best accuracy: {best_strategy[1]['best_test_acc']*100:.2f}%")

    print("\nImprovement over Standard Frozen baseline:")
    for name, res in strategies[1:]:
        improvement = (res['best_test_acc'] - baseline_acc) * 100
        print(f"  {name}: {improvement:+.2f}%")

    # Key findings
    print("\n" + "-"*60)
    print("KEY FINDINGS")
    print("-"*60)

    # Hypothesis 1: Progressive Deep-to-Shallow should beat Shallow-to-Deep
    h1 = result_2['best_test_acc'] > result_3['best_test_acc']
    print(f"\n1. Deep-to-Shallow > Shallow-to-Deep: {'CONFIRMED' if h1 else 'NOT CONFIRMED'}")
    print(f"   Deep-to-Shallow: {result_2['best_test_acc']*100:.2f}%")
    print(f"   Shallow-to-Deep: {result_3['best_test_acc']*100:.2f}%")
    print(f"   Difference: {(result_2['best_test_acc'] - result_3['best_test_acc'])*100:+.2f}%")

    # Hypothesis 2: Progressive should beat Standard Frozen
    h2 = result_2['best_test_acc'] > result_1['best_test_acc']
    print(f"\n2. Progressive > Standard Frozen: {'CONFIRMED' if h2 else 'NOT CONFIRMED'}")
    print(f"   Progressive: {result_2['best_test_acc']*100:.2f}%")
    print(f"   Standard Frozen: {result_1['best_test_acc']*100:.2f}%")
    print(f"   Improvement: {(result_2['best_test_acc'] - result_1['best_test_acc'])*100:+.2f}%")

    # Hypothesis 3: Discriminative LR should also improve
    h3 = result_4['best_test_acc'] > result_1['best_test_acc']
    print(f"\n3. Discriminative LR > Standard Frozen: {'CONFIRMED' if h3 else 'NOT CONFIRMED'}")
    print(f"   Discriminative LR: {result_4['best_test_acc']*100:.2f}%")
    print(f"   Improvement: {(result_4['best_test_acc'] - result_1['best_test_acc'])*100:+.2f}%")

    # Store analysis
    results['analysis'] = {
        'best_strategy': best_strategy[0],
        'best_accuracy': best_strategy[1]['best_test_acc'],
        'hypothesis_1_deep_to_shallow_better': h1,
        'hypothesis_2_progressive_beats_frozen': h2,
        'hypothesis_3_discriminative_lr_helps': h3,
        'improvements': {
            'progressive_deep_to_shallow': result_2['best_test_acc'] - baseline_acc,
            'progressive_shallow_to_deep': result_3['best_test_acc'] - baseline_acc,
            'discriminative_lr': result_4['best_test_acc'] - baseline_acc,
            'full_finetune': result_5['best_test_acc'] - baseline_acc
        }
    }

    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (float, int, str, bool, type(None))):
            return obj
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return str(obj)

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Progressive Unfreezing Experiment')
    parser.add_argument('--pretrain-epochs', type=int, default=500,
                        help='Epochs per layer for pretraining (default: 500)')
    parser.add_argument('--transfer-epochs', type=int, default=80,
                        help='Total epochs for transfer (default: 80)')
    parser.add_argument('--epochs-per-phase', type=int, default=20,
                        help='Epochs per phase for progressive unfreezing (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (100 pretrain epochs)')
    args = parser.parse_args()

    pretrain_epochs = 100 if args.quick else args.pretrain_epochs
    transfer_epochs = 40 if args.quick else args.transfer_epochs
    epochs_per_phase = 10 if args.quick else args.epochs_per_phase

    results = run_experiment(
        pretrain_epochs=pretrain_epochs,
        transfer_epochs=transfer_epochs,
        epochs_per_phase=epochs_per_phase,
        seed=args.seed
    )

    output_path = str(Path(__file__).parent.parent / 'results' / 'progressive_unfreezing.json')
    save_results(results, output_path)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
