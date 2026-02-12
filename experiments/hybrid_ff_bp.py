#!/usr/bin/env python3
"""
Hybrid Forward-Forward and Backpropagation Training Experiment
================================================================

Innovation: FF has layer isolation problem, but BP can establish global connections.
What happens if we combine them?

Experiment Design:

**Strategy A: FF Pre-training + BP Fine-tuning**
- Pre-train with FF on MNIST (500 epochs/layer)
- Fine-tune entire network with BP on MNIST (50 epochs)
- Transfer to Fashion-MNIST

**Strategy B: Alternating Training**
- Train 100 epochs FF, then 10 epochs BP
- Repeat 5 times
- Transfer to Fashion-MNIST

**Strategy C: Progressive BP Injection**
- Start with 100% FF
- Gradually increase BP loss ratio (10%, 20%, 30%...)
- Transfer to Fashion-MNIST

Baselines:
- Pure FF: ~61% transfer
- Pure BP: ~77% transfer
- Random: ~84% transfer

Hypothesis: Hybrid training may combine FF's local learning with BP's global coordination.

Author: Clawd (for Parafee)
Date: 2026-02-09
"""

import os
import sys
import json
import time
import copy
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


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

def get_mnist_data(device: torch.device):
    """Load full MNIST dataset to device."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    return (x_train.to(device), y_train.to(device)), (x_test.to(device), y_test.to(device))


def get_fashion_mnist_data(device: torch.device):
    """Load full Fashion-MNIST dataset to device."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    return (x_train.to(device), y_train.to(device)), (x_test.to(device), y_test.to(device))


# ============================================================
# Forward-Forward Implementation
# ============================================================

def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Embed label in first 10 pixels using x.max()."""
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class HybridFFLayer(nn.Module):
    """
    Hybrid Forward-Forward Layer.

    Supports both FF (local) and BP (global) training modes.
    """

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.ff_opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Forward pass with optional L2 normalization."""
        if normalize:
            x = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """MEAN of squared activations (not sum!)."""
        return h.pow(2).mean(dim=1)

    def ff_loss(self, h_pos: torch.Tensor, h_neg: torch.Tensor) -> torch.Tensor:
        """Compute FF loss for positive and negative samples."""
        g_pos = self.goodness(h_pos)
        g_neg = self.goodness(h_neg)
        return torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold
        ]))).mean()

    def train_ff_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> float:
        """Single FF training step."""
        h_pos = self.forward(x_pos)
        h_neg = self.forward(x_neg)

        loss = self.ff_loss(h_pos, h_neg)

        self.ff_opt.zero_grad()
        loss.backward()
        self.ff_opt.step()

        return loss.item()

    def train_ff_epochs(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                        num_epochs: int, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train this layer for multiple FF epochs."""
        iterator = tqdm(range(num_epochs), desc="FF Layer") if verbose else range(num_epochs)

        for _ in iterator:
            loss = self.train_ff_step(x_pos, x_neg)

        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class HybridFFBPNetwork(nn.Module):
    """
    Hybrid FF-BP Network.

    Supports multiple training strategies:
    - Pure FF (greedy layer-by-layer)
    - Pure BP (end-to-end)
    - Hybrid combinations
    """

    def __init__(self, dims: List[int], num_classes: int = 10,
                 threshold: float = 2.0, lr_ff: float = 0.03, lr_bp: float = 0.001):
        super().__init__()
        self.dims = dims
        self.num_classes = num_classes
        self.threshold = threshold

        # Feature layers (can be trained with FF or BP)
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(HybridFFLayer(dims[d], dims[d + 1], threshold, lr_ff))

        # Classification head (for BP training)
        self.classifier = nn.Linear(dims[-1], num_classes)

        # BP optimizer (created when needed)
        self.lr_bp = lr_bp
        self.bp_opt = None

    def _init_bp_optimizer(self):
        """Initialize BP optimizer if not exists."""
        if self.bp_opt is None:
            self.bp_opt = optim.Adam(self.parameters(), lr=self.lr_bp)

    def forward(self, x: torch.Tensor, for_ff: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            for_ff: If True, use L2 normalization (FF mode). If False, no normalization (BP mode).
        """
        for layer in self.layers:
            x = layer(x, normalize=for_ff)
        if not for_ff:
            x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features from last hidden layer."""
        with torch.no_grad():
            for layer in self.layers:
                x = layer(x, normalize=False)
            return x

    # =====================================================
    # FF Training Methods
    # =====================================================

    def train_ff_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                        epochs_per_layer: int = 500, verbose: bool = True):
        """Greedy layer-by-layer FF training."""
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\n  Training FF Layer {i}...')
            h_pos, h_neg = layer.train_ff_epochs(h_pos, h_neg, epochs_per_layer, verbose)

    def predict_ff(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using FF goodness measure."""
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(self.num_classes):
            h = overlay_y_on_x(x, torch.full((batch_size,), label, device=x.device))

            goodness = []
            for layer in self.layers:
                h = layer(h, normalize=True)
                goodness.append(layer.goodness(h))

            total_goodness = sum(goodness)
            goodness_per_label.append(total_goodness.unsqueeze(1))

        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)

    def get_accuracy_ff(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Get accuracy using FF prediction."""
        predictions = self.predict_ff(x)
        return (predictions == y).float().mean().item()

    # =====================================================
    # BP Training Methods
    # =====================================================

    def train_bp_epoch(self, x_train: torch.Tensor, y_train: torch.Tensor,
                       batch_size: int = 128) -> float:
        """Single BP training epoch."""
        self._init_bp_optimizer()
        self.train()
        criterion = nn.CrossEntropyLoss()

        indices = torch.randperm(len(x_train))
        epoch_losses = []

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            self.bp_opt.zero_grad()
            outputs = self.forward(x_batch, for_ff=False)
            loss = criterion(outputs, y_batch)
            loss.backward()
            self.bp_opt.step()
            epoch_losses.append(loss.item())

        return np.mean(epoch_losses)

    def train_bp_epochs(self, x_train: torch.Tensor, y_train: torch.Tensor,
                        x_test: torch.Tensor, y_test: torch.Tensor,
                        epochs: int = 50, batch_size: int = 128,
                        verbose: bool = True) -> Dict:
        """Train with BP for multiple epochs."""
        history = {'train_acc': [], 'test_acc': [], 'loss': []}

        for epoch in range(epochs):
            loss = self.train_bp_epoch(x_train, y_train, batch_size)

            train_acc = self.get_accuracy_bp(x_train, y_train)
            test_acc = self.get_accuracy_bp(x_test, y_test)

            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['loss'].append(loss)

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"  BP Epoch {epoch+1:3d}/{epochs} | Loss: {loss:.4f} | "
                      f"Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

        return history

    def get_accuracy_bp(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Get accuracy using BP classifier."""
        self.train(False)
        with torch.no_grad():
            preds = self.forward(x, for_ff=False).argmax(dim=1)
            return (preds == y).float().mean().item()

    # =====================================================
    # Hybrid Training Methods
    # =====================================================

    def hybrid_ff_loss_bp_loss(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                                x: torch.Tensor, y: torch.Tensor,
                                ff_weight: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute combined FF and BP losses.

        Returns: (ff_loss, bp_loss)
        """
        # FF loss (layer-wise)
        ff_losses = []
        h_pos, h_neg = x_pos, x_neg
        for layer in self.layers:
            h_pos_new = layer(h_pos, normalize=True)
            h_neg_new = layer(h_neg, normalize=True)
            ff_losses.append(layer.ff_loss(h_pos_new, h_neg_new))
            h_pos, h_neg = h_pos_new.detach(), h_neg_new.detach()  # Detach for locality
        ff_loss = sum(ff_losses) / len(ff_losses)

        # BP loss
        outputs = self.forward(x, for_ff=False)
        bp_loss = nn.CrossEntropyLoss()(outputs, y)

        return ff_loss, bp_loss


# ============================================================
# Training Strategies
# ============================================================

def strategy_a_ff_pretrain_bp_finetune(
    model: HybridFFBPNetwork,
    mnist_train: torch.Tensor, mnist_train_y: torch.Tensor,
    mnist_test: torch.Tensor, mnist_test_y: torch.Tensor,
    ff_epochs_per_layer: int = 500,
    bp_finetune_epochs: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Strategy A: FF Pre-training + BP Fine-tuning

    1. Pre-train with FF (greedy layer-by-layer)
    2. Fine-tune entire network with BP
    """
    results = {'strategy': 'A', 'name': 'FF_Pretrain_BP_Finetune'}

    # Phase 1: FF Pre-training
    if verbose:
        print("\n  [Strategy A] Phase 1: FF Pre-training...")

    x_pos = overlay_y_on_x(mnist_train, mnist_train_y)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg = overlay_y_on_x(mnist_train, mnist_train_y[rnd])

    ff_start = time.time()
    model.train_ff_greedy(x_pos, x_neg, ff_epochs_per_layer, verbose)
    ff_time = time.time() - ff_start

    ff_train_acc = model.get_accuracy_ff(mnist_train, mnist_train_y)
    ff_test_acc = model.get_accuracy_ff(mnist_test, mnist_test_y)

    results['ff_phase'] = {
        'train_acc': ff_train_acc,
        'test_acc': ff_test_acc,
        'time': ff_time
    }

    if verbose:
        print(f"\n  After FF: Train {ff_train_acc*100:.2f}% | Test {ff_test_acc*100:.2f}%")

    # Phase 2: BP Fine-tuning
    if verbose:
        print("\n  [Strategy A] Phase 2: BP Fine-tuning...")

    bp_start = time.time()
    bp_history = model.train_bp_epochs(
        mnist_train, mnist_train_y, mnist_test, mnist_test_y,
        epochs=bp_finetune_epochs, verbose=verbose
    )
    bp_time = time.time() - bp_start

    bp_train_acc = model.get_accuracy_bp(mnist_train, mnist_train_y)
    bp_test_acc = model.get_accuracy_bp(mnist_test, mnist_test_y)

    results['bp_phase'] = {
        'train_acc': bp_train_acc,
        'test_acc': bp_test_acc,
        'time': bp_time
    }

    results['final'] = {
        'train_acc': bp_train_acc,
        'test_acc': bp_test_acc,
        'total_time': ff_time + bp_time
    }

    if verbose:
        print(f"\n  After BP: Train {bp_train_acc*100:.2f}% | Test {bp_test_acc*100:.2f}%")

    return results


def strategy_b_alternating(
    model: HybridFFBPNetwork,
    mnist_train: torch.Tensor, mnist_train_y: torch.Tensor,
    mnist_test: torch.Tensor, mnist_test_y: torch.Tensor,
    num_cycles: int = 5,
    ff_epochs_per_cycle: int = 100,
    bp_epochs_per_cycle: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Strategy B: Alternating FF and BP Training

    Train 100 epochs FF, then 10 epochs BP, repeat N times.
    """
    results = {'strategy': 'B', 'name': 'Alternating_FF_BP', 'cycles': []}

    x_pos = overlay_y_on_x(mnist_train, mnist_train_y)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg = overlay_y_on_x(mnist_train, mnist_train_y[rnd])

    start_time = time.time()

    for cycle in range(num_cycles):
        if verbose:
            print(f"\n  [Strategy B] Cycle {cycle + 1}/{num_cycles}")

        cycle_result = {'cycle': cycle + 1}

        # FF phase
        if verbose:
            print(f"    FF Training ({ff_epochs_per_cycle} epochs/layer)...")

        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(model.layers):
            h_pos, h_neg = layer.train_ff_epochs(h_pos, h_neg, ff_epochs_per_cycle, verbose=False)

        ff_acc = model.get_accuracy_ff(mnist_test, mnist_test_y)
        cycle_result['ff_test_acc'] = ff_acc

        # BP phase
        if verbose:
            print(f"    BP Fine-tuning ({bp_epochs_per_cycle} epochs)...")

        model.train_bp_epochs(
            mnist_train, mnist_train_y, mnist_test, mnist_test_y,
            epochs=bp_epochs_per_cycle, verbose=False
        )

        bp_acc = model.get_accuracy_bp(mnist_test, mnist_test_y)
        cycle_result['bp_test_acc'] = bp_acc

        if verbose:
            print(f"    Cycle {cycle+1}: FF {ff_acc*100:.2f}% -> BP {bp_acc*100:.2f}%")

        results['cycles'].append(cycle_result)

    total_time = time.time() - start_time

    results['final'] = {
        'train_acc': model.get_accuracy_bp(mnist_train, mnist_train_y),
        'test_acc': model.get_accuracy_bp(mnist_test, mnist_test_y),
        'total_time': total_time
    }

    return results


def strategy_c_progressive_bp_injection(
    model: HybridFFBPNetwork,
    mnist_train: torch.Tensor, mnist_train_y: torch.Tensor,
    mnist_test: torch.Tensor, mnist_test_y: torch.Tensor,
    bp_ratios: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    epochs_per_stage: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Strategy C: Progressive BP Injection

    Start with 100% FF, gradually increase BP loss ratio.
    """
    results = {'strategy': 'C', 'name': 'Progressive_BP_Injection', 'stages': []}

    x_pos = overlay_y_on_x(mnist_train, mnist_train_y)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg = overlay_y_on_x(mnist_train, mnist_train_y[rnd])

    model._init_bp_optimizer()
    start_time = time.time()

    for stage_idx, bp_ratio in enumerate(bp_ratios):
        ff_ratio = 1.0 - bp_ratio

        if verbose:
            print(f"\n  [Strategy C] Stage {stage_idx + 1}/{len(bp_ratios)}: "
                  f"FF {ff_ratio*100:.0f}% / BP {bp_ratio*100:.0f}%")

        stage_result = {'stage': stage_idx + 1, 'ff_ratio': ff_ratio, 'bp_ratio': bp_ratio}

        for epoch in range(epochs_per_stage):
            model.train()

            # Compute combined loss
            ff_loss, bp_loss = model.hybrid_ff_loss_bp_loss(x_pos, x_neg, mnist_train, mnist_train_y)
            combined_loss = ff_ratio * ff_loss + bp_ratio * bp_loss

            model.bp_opt.zero_grad()
            combined_loss.backward()
            model.bp_opt.step()

            # Also update FF optimizers for their respective losses
            if ff_ratio > 0:
                for layer in model.layers:
                    layer.ff_opt.step()

        test_acc = model.get_accuracy_bp(mnist_test, mnist_test_y)
        stage_result['test_acc'] = test_acc

        if verbose:
            print(f"    Test accuracy: {test_acc*100:.2f}%")

        results['stages'].append(stage_result)

    total_time = time.time() - start_time

    results['final'] = {
        'train_acc': model.get_accuracy_bp(mnist_train, mnist_train_y),
        'test_acc': model.get_accuracy_bp(mnist_test, mnist_test_y),
        'total_time': total_time
    }

    return results


# ============================================================
# Baseline Trainers
# ============================================================

def train_pure_ff(
    dims: List[int], device: torch.device,
    mnist_train: torch.Tensor, mnist_train_y: torch.Tensor,
    mnist_test: torch.Tensor, mnist_test_y: torch.Tensor,
    epochs_per_layer: int = 500,
    verbose: bool = True
) -> Tuple[HybridFFBPNetwork, Dict]:
    """Train pure FF baseline."""
    model = HybridFFBPNetwork(dims, threshold=2.0, lr_ff=0.03, lr_bp=0.001).to(device)

    x_pos = overlay_y_on_x(mnist_train, mnist_train_y)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg = overlay_y_on_x(mnist_train, mnist_train_y[rnd])

    start_time = time.time()
    model.train_ff_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    train_time = time.time() - start_time

    results = {
        'method': 'Pure_FF',
        'source_train_acc': model.get_accuracy_ff(mnist_train, mnist_train_y),
        'source_test_acc': model.get_accuracy_ff(mnist_test, mnist_test_y),
        'train_time': train_time
    }

    return model, results


def train_pure_bp(
    dims: List[int], device: torch.device,
    mnist_train: torch.Tensor, mnist_train_y: torch.Tensor,
    mnist_test: torch.Tensor, mnist_test_y: torch.Tensor,
    epochs: int = 50,
    verbose: bool = True
) -> Tuple[HybridFFBPNetwork, Dict]:
    """Train pure BP baseline."""
    model = HybridFFBPNetwork(dims, threshold=2.0, lr_ff=0.03, lr_bp=0.001).to(device)

    start_time = time.time()
    model.train_bp_epochs(mnist_train, mnist_train_y, mnist_test, mnist_test_y,
                          epochs=epochs, verbose=verbose)
    train_time = time.time() - start_time

    results = {
        'method': 'Pure_BP',
        'source_train_acc': model.get_accuracy_bp(mnist_train, mnist_train_y),
        'source_test_acc': model.get_accuracy_bp(mnist_test, mnist_test_y),
        'train_time': train_time
    }

    return model, results


# ============================================================
# Transfer Learning
# ============================================================

class LinearHead(nn.Module):
    """Linear classification head for transfer learning."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_transfer_head(
    features_train: torch.Tensor, y_train: torch.Tensor,
    features_test: torch.Tensor, y_test: torch.Tensor,
    epochs: int = 100, batch_size: int = 256, lr: float = 0.01,
    verbose: bool = True
) -> Dict:
    """Train linear head on frozen features."""
    feature_dim = features_train.shape[1]
    num_classes = int(y_train.max().item()) + 1

    head = LinearHead(feature_dim, num_classes).to(features_train.device)
    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):
        head.train()
        indices = torch.randperm(len(features_train))
        epoch_losses = []

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = features_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            outputs = head(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Evaluate
        head.train(False)
        with torch.no_grad():
            test_preds = head(features_test).argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()
            best_acc = max(best_acc, test_acc)

        if verbose and ((epoch + 1) % 25 == 0 or epoch == 0):
            print(f"    Transfer Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {np.mean(epoch_losses):.4f} | Test: {test_acc*100:.2f}%")

    return {
        'final_accuracy': test_acc,
        'best_accuracy': best_acc
    }


def evaluate_transfer(
    model: HybridFFBPNetwork,
    fmnist_train: torch.Tensor, fmnist_train_y: torch.Tensor,
    fmnist_test: torch.Tensor, fmnist_test_y: torch.Tensor,
    transfer_epochs: int = 100,
    verbose: bool = True
) -> Dict:
    """Evaluate transfer learning performance."""
    # Extract features
    features_train = model.get_features(fmnist_train)
    features_test = model.get_features(fmnist_test)

    # Train transfer head
    return train_transfer_head(
        features_train, fmnist_train_y,
        features_test, fmnist_test_y,
        epochs=transfer_epochs, verbose=verbose
    )


# ============================================================
# Main Experiment
# ============================================================

def run_hybrid_experiment(
    ff_epochs_per_layer: int = 500,
    bp_epochs: int = 50,
    transfer_epochs: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run the complete hybrid FF-BP experiment.
    """
    print("="*70)
    print("HYBRID FF-BP TRAINING EXPERIMENT")
    print("="*70)
    print(f"\nSettings:")
    print(f"  FF epochs per layer: {ff_epochs_per_layer}")
    print(f"  BP epochs: {bp_epochs}")
    print(f"  Transfer epochs: {transfer_epochs}")
    print(f"  Seed: {seed}")

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"  Device: {device}")

    dims = [784, 500, 500]

    results = {
        'experiment': 'hybrid_ff_bp_training',
        'config': {
            'ff_epochs_per_layer': ff_epochs_per_layer,
            'bp_epochs': bp_epochs,
            'transfer_epochs': transfer_epochs,
            'seed': seed,
            'device': str(device),
            'architecture': dims
        },
        'timestamp': datetime.now().isoformat()
    }

    # Load data
    print("\n" + "-"*60)
    print("Loading datasets...")
    (mnist_train, mnist_train_y), (mnist_test, mnist_test_y) = get_mnist_data(device)
    (fmnist_train, fmnist_train_y), (fmnist_test, fmnist_test_y) = get_fashion_mnist_data(device)
    print(f"  MNIST: {len(mnist_train)} train, {len(mnist_test)} test")
    print(f"  Fashion-MNIST: {len(fmnist_train)} train, {len(fmnist_test)} test")

    # ============================================================
    # Baseline 1: Pure FF
    # ============================================================
    print("\n" + "="*60)
    print("BASELINE 1: Pure Forward-Forward")
    print("="*60)

    torch.manual_seed(seed)
    pure_ff_model, pure_ff_results = train_pure_ff(
        dims, device, mnist_train, mnist_train_y, mnist_test, mnist_test_y,
        epochs_per_layer=ff_epochs_per_layer, verbose=verbose
    )

    print(f"\n  Source: Train {pure_ff_results['source_train_acc']*100:.2f}% | "
          f"Test {pure_ff_results['source_test_acc']*100:.2f}%")

    print("  Evaluating transfer...")
    pure_ff_transfer = evaluate_transfer(
        pure_ff_model, fmnist_train, fmnist_train_y, fmnist_test, fmnist_test_y,
        transfer_epochs, verbose
    )
    pure_ff_results['transfer_accuracy'] = pure_ff_transfer['best_accuracy']
    results['pure_ff'] = pure_ff_results

    print(f"\n  Transfer: {pure_ff_transfer['best_accuracy']*100:.2f}%")

    # ============================================================
    # Baseline 2: Pure BP
    # ============================================================
    print("\n" + "="*60)
    print("BASELINE 2: Pure Backpropagation")
    print("="*60)

    torch.manual_seed(seed)
    pure_bp_model, pure_bp_results = train_pure_bp(
        dims, device, mnist_train, mnist_train_y, mnist_test, mnist_test_y,
        epochs=bp_epochs, verbose=verbose
    )

    print(f"\n  Source: Train {pure_bp_results['source_train_acc']*100:.2f}% | "
          f"Test {pure_bp_results['source_test_acc']*100:.2f}%")

    print("  Evaluating transfer...")
    pure_bp_transfer = evaluate_transfer(
        pure_bp_model, fmnist_train, fmnist_train_y, fmnist_test, fmnist_test_y,
        transfer_epochs, verbose
    )
    pure_bp_results['transfer_accuracy'] = pure_bp_transfer['best_accuracy']
    results['pure_bp'] = pure_bp_results

    print(f"\n  Transfer: {pure_bp_transfer['best_accuracy']*100:.2f}%")

    # ============================================================
    # Baseline 3: Random Init
    # ============================================================
    print("\n" + "="*60)
    print("BASELINE 3: Random Initialization")
    print("="*60)

    torch.manual_seed(seed)
    random_model = HybridFFBPNetwork(dims, threshold=2.0, lr_ff=0.03, lr_bp=0.001).to(device)

    print("  Evaluating transfer...")
    random_transfer = evaluate_transfer(
        random_model, fmnist_train, fmnist_train_y, fmnist_test, fmnist_test_y,
        transfer_epochs, verbose
    )
    results['random'] = {'transfer_accuracy': random_transfer['best_accuracy']}

    print(f"\n  Transfer: {random_transfer['best_accuracy']*100:.2f}%")

    # ============================================================
    # Strategy A: FF Pre-train + BP Fine-tune
    # ============================================================
    print("\n" + "="*60)
    print("STRATEGY A: FF Pre-training + BP Fine-tuning")
    print("="*60)

    torch.manual_seed(seed)
    model_a = HybridFFBPNetwork(dims, threshold=2.0, lr_ff=0.03, lr_bp=0.001).to(device)
    strategy_a_results = strategy_a_ff_pretrain_bp_finetune(
        model_a, mnist_train, mnist_train_y, mnist_test, mnist_test_y,
        ff_epochs_per_layer=ff_epochs_per_layer,
        bp_finetune_epochs=bp_epochs,
        verbose=verbose
    )

    print("\n  Evaluating transfer...")
    strategy_a_transfer = evaluate_transfer(
        model_a, fmnist_train, fmnist_train_y, fmnist_test, fmnist_test_y,
        transfer_epochs, verbose
    )
    strategy_a_results['transfer_accuracy'] = strategy_a_transfer['best_accuracy']
    results['strategy_a'] = strategy_a_results

    print(f"\n  Strategy A Transfer: {strategy_a_transfer['best_accuracy']*100:.2f}%")

    # ============================================================
    # Strategy B: Alternating Training
    # ============================================================
    print("\n" + "="*60)
    print("STRATEGY B: Alternating FF and BP")
    print("="*60)

    torch.manual_seed(seed)
    model_b = HybridFFBPNetwork(dims, threshold=2.0, lr_ff=0.03, lr_bp=0.001).to(device)
    strategy_b_results = strategy_b_alternating(
        model_b, mnist_train, mnist_train_y, mnist_test, mnist_test_y,
        num_cycles=5,
        ff_epochs_per_cycle=100,
        bp_epochs_per_cycle=10,
        verbose=verbose
    )

    print("\n  Evaluating transfer...")
    strategy_b_transfer = evaluate_transfer(
        model_b, fmnist_train, fmnist_train_y, fmnist_test, fmnist_test_y,
        transfer_epochs, verbose
    )
    strategy_b_results['transfer_accuracy'] = strategy_b_transfer['best_accuracy']
    results['strategy_b'] = strategy_b_results

    print(f"\n  Strategy B Transfer: {strategy_b_transfer['best_accuracy']*100:.2f}%")

    # ============================================================
    # Strategy C: Progressive BP Injection
    # ============================================================
    print("\n" + "="*60)
    print("STRATEGY C: Progressive BP Injection")
    print("="*60)

    torch.manual_seed(seed)
    model_c = HybridFFBPNetwork(dims, threshold=2.0, lr_ff=0.03, lr_bp=0.001).to(device)
    strategy_c_results = strategy_c_progressive_bp_injection(
        model_c, mnist_train, mnist_train_y, mnist_test, mnist_test_y,
        bp_ratios=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        epochs_per_stage=50,
        verbose=verbose
    )

    print("\n  Evaluating transfer...")
    strategy_c_transfer = evaluate_transfer(
        model_c, fmnist_train, fmnist_train_y, fmnist_test, fmnist_test_y,
        transfer_epochs, verbose
    )
    strategy_c_results['transfer_accuracy'] = strategy_c_transfer['best_accuracy']
    results['strategy_c'] = strategy_c_results

    print(f"\n  Strategy C Transfer: {strategy_c_transfer['best_accuracy']*100:.2f}%")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Method':<35} {'Source Acc':>12} {'Transfer Acc':>14}")
    print("-" * 63)
    print(f"{'Pure FF':<35} {pure_ff_results['source_test_acc']*100:>11.2f}% "
          f"{pure_ff_results['transfer_accuracy']*100:>13.2f}%")
    print(f"{'Pure BP':<35} {pure_bp_results['source_test_acc']*100:>11.2f}% "
          f"{pure_bp_results['transfer_accuracy']*100:>13.2f}%")
    print(f"{'Random Init':<35} {'N/A':>12} "
          f"{results['random']['transfer_accuracy']*100:>13.2f}%")
    print("-" * 63)
    print(f"{'Strategy A (FF+BP Finetune)':<35} {strategy_a_results['final']['test_acc']*100:>11.2f}% "
          f"{strategy_a_results['transfer_accuracy']*100:>13.2f}%")
    print(f"{'Strategy B (Alternating)':<35} {strategy_b_results['final']['test_acc']*100:>11.2f}% "
          f"{strategy_b_results['transfer_accuracy']*100:>13.2f}%")
    print(f"{'Strategy C (Progressive BP)':<35} {strategy_c_results['final']['test_acc']*100:>11.2f}% "
          f"{strategy_c_results['transfer_accuracy']*100:>13.2f}%")

    # Analysis
    print("\n" + "-"*60)
    print("ANALYSIS")
    print("-"*60)

    random_baseline = results['random']['transfer_accuracy']

    print(f"\nTransfer accuracy gain over random init:")
    print(f"  Pure FF:    {(pure_ff_results['transfer_accuracy'] - random_baseline)*100:+.2f}%")
    print(f"  Pure BP:    {(pure_bp_results['transfer_accuracy'] - random_baseline)*100:+.2f}%")
    print(f"  Strategy A: {(strategy_a_results['transfer_accuracy'] - random_baseline)*100:+.2f}%")
    print(f"  Strategy B: {(strategy_b_results['transfer_accuracy'] - random_baseline)*100:+.2f}%")
    print(f"  Strategy C: {(strategy_c_results['transfer_accuracy'] - random_baseline)*100:+.2f}%")

    best_hybrid = max(
        strategy_a_results['transfer_accuracy'],
        strategy_b_results['transfer_accuracy'],
        strategy_c_results['transfer_accuracy']
    )
    best_baseline = max(pure_ff_results['transfer_accuracy'], pure_bp_results['transfer_accuracy'])

    print(f"\nBest hybrid vs best baseline: {(best_hybrid - best_baseline)*100:+.2f}%")

    if best_hybrid > best_baseline:
        print("CONCLUSION: Hybrid training shows improvement over pure methods!")
    else:
        print("CONCLUSION: Hybrid training does not improve over pure methods.")

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
    parser = argparse.ArgumentParser(description="Hybrid FF-BP Training Experiment")
    parser.add_argument('--ff-epochs', type=int, default=500,
                        help='FF epochs per layer')
    parser.add_argument('--bp-epochs', type=int, default=50,
                        help='BP training epochs')
    parser.add_argument('--transfer-epochs', type=int, default=100,
                        help='Transfer head training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (FF: 100 epochs/layer)')
    parser.add_argument('--full', action='store_true',
                        help='Full experiment (FF: 500 epochs/layer)')
    args = parser.parse_args()

    ff_epochs = args.ff_epochs
    if args.quick:
        ff_epochs = 100
    elif args.full:
        ff_epochs = 500

    # Run experiment
    results = run_hybrid_experiment(
        ff_epochs_per_layer=ff_epochs,
        bp_epochs=args.bp_epochs,
        transfer_epochs=args.transfer_epochs,
        seed=args.seed
    )

    # Save results
    output_path = './results/hybrid_training.json'
    save_results(results, output_path)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
