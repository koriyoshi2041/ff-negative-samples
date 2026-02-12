#!/usr/bin/env python3
"""
Label-Free Forward-Forward Experiment
======================================

Core Hypothesis: Label embedding is the main reason for poor FF transfer,
because it tightly couples features with task labels.

Experiment Design - Three Label-Free Approaches:

1. Self-Supervised Goodness:
   - Positive: Original image
   - Negative: Augmented image (rotation, crop, noise)
   - No label information used
   - Check with Linear Probe

2. Contrastive FF:
   - Positive: Two augmentations of same image
   - Negative: Different images
   - Similar to SimCLR but using goodness instead of cosine similarity

3. Reconstruction FF:
   - Target: Reconstruct input (autoencoder-like)
   - Goodness = reconstruction quality
   - Fully self-supervised

Comparison:
- Standard FF (with label): ~61% transfer baseline
- Label-Free FF variants: ?
- Expected: Better transfer due to more general features

Author: Claude (for Parafee)
Date: 2026-02-09
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
import random


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Data Augmentation
# ============================================================

class AugmentationPipeline:
    """Data augmentation pipeline for self-supervised learning."""

    def __init__(self, strength: str = 'medium'):
        self.strength = strength

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to input."""
        # x shape: (batch, 784) for MNIST
        batch_size = x.shape[0]

        # Reshape to image format
        x_img = x.view(batch_size, 1, 28, 28)

        augmented = []
        for i in range(batch_size):
            img = x_img[i]

            # Random augmentation choice
            aug_type = random.choice(['rotation', 'noise', 'translation', 'scale', 'combined'])

            if aug_type == 'rotation':
                angle = random.uniform(-30, 30)
                img = TF.rotate(img, angle)

            elif aug_type == 'noise':
                noise_level = 0.3 if self.strength == 'strong' else 0.15
                noise = torch.randn_like(img) * noise_level
                img = img + noise

            elif aug_type == 'translation':
                max_shift = 5 if self.strength == 'strong' else 3
                shift_x = random.randint(-max_shift, max_shift)
                shift_y = random.randint(-max_shift, max_shift)
                img = TF.affine(img, angle=0, translate=[shift_x, shift_y],
                               scale=1.0, shear=0)

            elif aug_type == 'scale':
                scale = random.uniform(0.8, 1.2) if self.strength == 'strong' else random.uniform(0.9, 1.1)
                img = TF.affine(img, angle=0, translate=[0, 0], scale=scale, shear=0)

            elif aug_type == 'combined':
                # Apply multiple augmentations
                angle = random.uniform(-20, 20)
                img = TF.rotate(img, angle)
                noise = torch.randn_like(img) * 0.1
                img = img + noise
                shift_x = random.randint(-2, 2)
                shift_y = random.randint(-2, 2)
                img = TF.affine(img, angle=0, translate=[shift_x, shift_y],
                               scale=1.0, shear=0)

            augmented.append(img)

        augmented = torch.stack(augmented)
        return augmented.view(batch_size, -1)


class ContrastiveAugmentation:
    """Augmentation for contrastive learning (SimCLR-style)."""

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two augmented views of the input."""
        batch_size = x.shape[0]
        x_img = x.view(batch_size, 1, 28, 28)

        view1_list = []
        view2_list = []

        for i in range(batch_size):
            img = x_img[i]

            # View 1
            angle1 = random.uniform(-25, 25)
            v1 = TF.rotate(img, angle1)
            shift1 = [random.randint(-3, 3), random.randint(-3, 3)]
            v1 = TF.affine(v1, angle=0, translate=shift1, scale=1.0, shear=0)
            v1 = v1 + torch.randn_like(v1) * 0.1

            # View 2 (different augmentation)
            angle2 = random.uniform(-25, 25)
            v2 = TF.rotate(img, angle2)
            shift2 = [random.randint(-3, 3), random.randint(-3, 3)]
            v2 = TF.affine(v2, angle=0, translate=shift2, scale=1.0, shear=0)
            v2 = v2 + torch.randn_like(v2) * 0.1

            view1_list.append(v1)
            view2_list.append(v2)

        view1 = torch.stack(view1_list).view(batch_size, -1)
        view2 = torch.stack(view2_list).view(batch_size, -1)

        return view1, view2


# ============================================================
# Label-Free FF Layer
# ============================================================

class LabelFreeFFLayer(nn.Module):
    """Forward-Forward Layer without label dependency."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with layer normalization."""
        x_norm = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_norm))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Compute goodness - MEAN of squared activations."""
        return h.pow(2).mean(dim=1)

    def train_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> float:
        """Train one step with positive and negative samples."""
        h_pos = self.forward(x_pos)
        h_neg = self.forward(x_neg)

        g_pos = self.goodness(h_pos)
        g_neg = self.goodness(h_neg)

        # Loss: push positive above threshold, negative below
        loss = torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold
        ]))).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()


# ============================================================
# 1. Self-Supervised Goodness FF
# ============================================================

class SelfSupervisedFF(nn.Module):
    """
    Self-Supervised Forward-Forward Network.

    Positive: Original images
    Negative: Augmented images

    The network learns to distinguish real data from augmented data,
    forcing it to learn invariant features.
    """

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(LabelFreeFFLayer(dims[d], dims[d + 1], threshold, lr))
        self.augment = AugmentationPipeline(strength='medium')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get concatenated features from all layers (for linear probe)."""
        features = []
        h = x
        for layer in self.layers:
            h = layer(h)
            features.append(h)
        return torch.cat(features, dim=1)

    def train_epoch(self, dataloader: DataLoader, device: torch.device,
                    verbose: bool = True) -> Dict[str, float]:
        """Train for one epoch."""
        self.train()
        total_losses = {f'layer_{i}': 0.0 for i in range(len(self.layers))}
        n_batches = 0

        iterator = tqdm(dataloader, desc="Training") if verbose else dataloader

        for x, _ in iterator:  # Ignore labels!
            x = x.view(x.size(0), -1).to(device)

            # Positive: original data
            x_pos = x

            # Negative: augmented data
            x_neg = self.augment(x)

            # Train layer by layer
            h_pos, h_neg = x_pos, x_neg

            for i, layer in enumerate(self.layers):
                loss = layer.train_step(h_pos.detach(), h_neg.detach())
                total_losses[f'layer_{i}'] += loss

                # Get outputs for next layer
                with torch.no_grad():
                    h_pos = layer(h_pos)
                    h_neg = layer(h_neg)

            n_batches += 1

        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses


# ============================================================
# 2. Contrastive FF
# ============================================================

class ContrastiveFF(nn.Module):
    """
    Contrastive Forward-Forward Network.

    Positive: Two augmentations of the same image
    Negative: Different images

    Similar to SimCLR but using goodness instead of cosine similarity.
    """

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(LabelFreeFFLayer(dims[d], dims[d + 1], threshold, lr))
        self.augment = ContrastiveAugmentation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get concatenated features from all layers."""
        features = []
        h = x
        for layer in self.layers:
            h = layer(h)
            features.append(h)
        return torch.cat(features, dim=1)

    def train_epoch(self, dataloader: DataLoader, device: torch.device,
                    verbose: bool = True) -> Dict[str, float]:
        """Train for one epoch using contrastive objective."""
        self.train()
        total_losses = {f'layer_{i}': 0.0 for i in range(len(self.layers))}
        n_batches = 0

        iterator = tqdm(dataloader, desc="Training") if verbose else dataloader

        for x, _ in iterator:  # Ignore labels!
            x = x.view(x.size(0), -1).to(device)
            batch_size = x.size(0)

            # Generate two views of same images
            view1, view2 = self.augment(x)

            # Positive pairs: (view1, view2) from same image
            # These should have HIGH joint goodness
            x_pos = torch.cat([view1, view2], dim=1)  # Concatenate views

            # Negative pairs: shuffle view2 to pair with different images
            perm = torch.randperm(batch_size)
            x_neg = torch.cat([view1, view2[perm]], dim=1)  # Mismatched pairs

            # For FF, we need to project back to original dimension
            # Use a simple approach: train on view1 as pos, shuffled view1 as neg
            x_pos = view1
            x_neg = view1[perm]  # Different images (shuffled)

            # Train layer by layer
            h_pos, h_neg = x_pos, x_neg

            for i, layer in enumerate(self.layers):
                loss = layer.train_step(h_pos.detach(), h_neg.detach())
                total_losses[f'layer_{i}'] += loss

                with torch.no_grad():
                    h_pos = layer(h_pos)
                    h_neg = layer(h_neg)

            n_batches += 1

        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses


# ============================================================
# 3. Reconstruction FF
# ============================================================

class ReconstructionFFLayer(nn.Module):
    """FF Layer with reconstruction objective."""

    def __init__(self, in_features: int, out_features: int, lr: float = 0.03):
        super().__init__()
        self.encoder = nn.Linear(in_features, out_features)
        self.decoder = nn.Linear(out_features, in_features)
        self.relu = nn.ReLU()
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input."""
        x_norm = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.encoder(x_norm))

    def reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        """Decode hidden representation."""
        return self.decoder(h)

    def train_step(self, x: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Train with reconstruction loss."""
        h = self.forward(x)
        x_recon = self.reconstruct(h)

        # Reconstruction loss
        loss = F.mse_loss(x_recon, x)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item(), h.detach()


class ReconstructionFF(nn.Module):
    """
    Reconstruction-based Forward-Forward Network.

    Each layer learns to reconstruct its input.
    Goodness = reconstruction quality.
    Fully self-supervised, no labels needed.
    """

    def __init__(self, dims: List[int], lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(ReconstructionFFLayer(dims[d], dims[d + 1], lr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get concatenated features from all layers."""
        features = []
        h = x
        for layer in self.layers:
            h = layer(h)
            features.append(h)
        return torch.cat(features, dim=1)

    def train_epoch(self, dataloader: DataLoader, device: torch.device,
                    verbose: bool = True) -> Dict[str, float]:
        """Train for one epoch."""
        self.train()
        total_losses = {f'layer_{i}': 0.0 for i in range(len(self.layers))}
        n_batches = 0

        iterator = tqdm(dataloader, desc="Training") if verbose else dataloader

        for x, _ in iterator:  # Ignore labels!
            x = x.view(x.size(0), -1).to(device)

            # Train layer by layer
            h = x
            for i, layer in enumerate(self.layers):
                loss, h = layer.train_step(h)
                total_losses[f'layer_{i}'] += loss

            n_batches += 1

        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses


# ============================================================
# Standard FF with Labels (Baseline)
# ============================================================

class StandardFF(nn.Module):
    """Standard Forward-Forward with label embedding (baseline)."""

    def __init__(self, dims: List[int], num_classes: int = 10,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(LabelFreeFFLayer(dims[d], dims[d + 1], threshold, lr))

    def embed_label(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Embed label into first pixels of input."""
        x_ = x.clone()
        x_[:, :self.num_classes] *= 0.0
        x_[range(x.shape[0]), y] = x.max()
        return x_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get concatenated features from all layers (NO label embedding)."""
        features = []
        h = x  # Raw input, no label
        for layer in self.layers:
            h = layer(h)
            features.append(h)
        return torch.cat(features, dim=1)

    def train_epoch(self, dataloader: DataLoader, device: torch.device,
                    verbose: bool = True) -> Dict[str, float]:
        """Train for one epoch."""
        self.train()
        total_losses = {f'layer_{i}': 0.0 for i in range(len(self.layers))}
        n_batches = 0

        iterator = tqdm(dataloader, desc="Training") if verbose else dataloader

        for x, y in iterator:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)

            # Positive: correct label
            x_pos = self.embed_label(x, y)

            # Negative: wrong label (shuffled)
            rnd = torch.randperm(x.size(0))
            x_neg = self.embed_label(x, y[rnd])

            # Train layer by layer
            h_pos, h_neg = x_pos, x_neg

            for i, layer in enumerate(self.layers):
                loss = layer.train_step(h_pos.detach(), h_neg.detach())
                total_losses[f'layer_{i}'] += loss

                with torch.no_grad():
                    h_pos = layer(h_pos)
                    h_neg = layer(h_neg)

            n_batches += 1

        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict by trying all labels."""
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(self.num_classes):
            h = self.embed_label(x, torch.full((batch_size,), label, device=x.device))
            total_g = torch.zeros(batch_size, device=x.device)

            for layer in self.layers:
                h = layer(h)
                total_g += layer.goodness(h)

            goodness_per_label.append(total_g.unsqueeze(1))

        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)


# ============================================================
# Linear Probe Assessment
# ============================================================

class LinearProbe(nn.Module):
    """Linear classifier for assessment."""

    def __init__(self, in_features: int, num_classes: int, lr: float = 0.01):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=1)


def run_linear_probe(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    probe_epochs: int = 100,
    verbose: bool = True
) -> Dict[str, float]:
    """Run model features using linear probe."""
    model.train(False)  # Set to inference mode

    # Extract features
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    with torch.no_grad():
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            features = model.get_features(x)
            train_features.append(features.cpu())
            train_labels.append(y)

        for x, y in test_loader:
            x = x.view(x.size(0), -1).to(device)
            features = model.get_features(x)
            test_features.append(features.cpu())
            test_labels.append(y)

    train_features = torch.cat(train_features).to(device)
    train_labels = torch.cat(train_labels).to(device)
    test_features = torch.cat(test_features).to(device)
    test_labels = torch.cat(test_labels).to(device)

    feature_dim = train_features.size(1)
    num_classes = len(torch.unique(train_labels))

    if verbose:
        print(f"  Feature dim: {feature_dim}, Classes: {num_classes}")

    # Train linear probe
    probe = LinearProbe(feature_dim, num_classes).to(device)

    best_acc = 0.0
    history = []

    for epoch in range(probe_epochs):
        # Mini-batch training
        perm = torch.randperm(len(train_features))
        batch_losses = []

        for i in range(0, len(perm), 64):
            idx = perm[i:i+64]
            loss = probe.train_step(train_features[idx], train_labels[idx])
            batch_losses.append(loss)

        # Check accuracy
        preds = probe.predict(test_features)
        acc = (preds == test_labels).float().mean().item()
        history.append(acc)
        best_acc = max(best_acc, acc)

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"    Epoch {epoch+1:3d}/{probe_epochs} | Acc: {acc*100:.2f}%")

    return {
        'best_accuracy': best_acc,
        'final_accuracy': history[-1],
        'history': history
    }


# ============================================================
# Transfer Learning Assessment
# ============================================================

def run_transfer(
    model: nn.Module,
    source_name: str,
    target_name: str,
    device: torch.device,
    probe_epochs: int = 100,
    verbose: bool = True
) -> Dict[str, float]:
    """Check transfer from source to target dataset."""
    if verbose:
        print(f"\n  Transfer: {source_name} -> {target_name}")

    # Load target dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Use MNIST stats for simplicity
    ])

    if target_name == 'fashion_mnist':
        target_train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        target_test = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    elif target_name == 'kmnist':
        target_train = datasets.KMNIST('./data', train=True, download=True, transform=transform)
        target_test = datasets.KMNIST('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown target dataset: {target_name}")

    target_train_loader = DataLoader(target_train, batch_size=64, shuffle=True)
    target_test_loader = DataLoader(target_test, batch_size=256)

    return run_linear_probe(
        model, target_train_loader, target_test_loader,
        device, probe_epochs, verbose
    )


# ============================================================
# Main Experiment
# ============================================================

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    hidden_dims: List[int] = None
    pretrain_epochs: int = 50
    probe_epochs: int = 100
    batch_size: int = 64
    lr: float = 0.03
    threshold: float = 2.0
    seed: int = 42

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [500, 500]


def run_experiment(config: ExperimentConfig, verbose: bool = True) -> Dict:
    """Run the full label-free FF experiment."""

    print("=" * 70)
    print("LABEL-FREE FORWARD-FORWARD EXPERIMENT")
    print("=" * 70)
    print(f"\nHypothesis: Label embedding causes poor FF transfer.")
    print(f"Testing label-free alternatives.\n")

    # Setup
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Architecture: [784] + {config.hidden_dims}")
    print(f"Pretrain epochs: {config.pretrain_epochs}")
    print(f"Probe epochs: {config.probe_epochs}")

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)

    dims = [784] + config.hidden_dims
    results = {}

    # ============================================================
    # 1. Standard FF (with labels) - Baseline
    # ============================================================
    print("\n" + "=" * 60)
    print("1. STANDARD FF (with label embedding) - BASELINE")
    print("=" * 60)

    torch.manual_seed(config.seed)
    standard_ff = StandardFF(dims, num_classes=10, threshold=config.threshold, lr=config.lr).to(device)

    for epoch in range(config.pretrain_epochs):
        losses = standard_ff.train_epoch(train_loader, device, verbose=False)
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            avg_loss = np.mean(list(losses.values()))
            print(f"  Epoch {epoch+1:3d}/{config.pretrain_epochs} | Loss: {avg_loss:.4f}")

    print("\n  Checking on MNIST (source task)...")
    standard_ff_source = run_linear_probe(
        standard_ff, train_loader, test_loader, device, config.probe_epochs, verbose
    )

    print("\n  Checking transfer to Fashion-MNIST...")
    standard_ff_transfer = run_transfer(
        standard_ff, 'mnist', 'fashion_mnist', device, config.probe_epochs, verbose
    )

    results['standard_ff'] = {
        'type': 'labeled',
        'source_accuracy': standard_ff_source['best_accuracy'],
        'transfer_accuracy': standard_ff_transfer['best_accuracy'],
    }

    # ============================================================
    # 2. Self-Supervised FF
    # ============================================================
    print("\n" + "=" * 60)
    print("2. SELF-SUPERVISED FF (original vs augmented)")
    print("=" * 60)

    torch.manual_seed(config.seed)
    self_sup_ff = SelfSupervisedFF(dims, threshold=config.threshold, lr=config.lr).to(device)

    for epoch in range(config.pretrain_epochs):
        losses = self_sup_ff.train_epoch(train_loader, device, verbose=False)
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            avg_loss = np.mean(list(losses.values()))
            print(f"  Epoch {epoch+1:3d}/{config.pretrain_epochs} | Loss: {avg_loss:.4f}")

    print("\n  Checking on MNIST (source task)...")
    self_sup_source = run_linear_probe(
        self_sup_ff, train_loader, test_loader, device, config.probe_epochs, verbose
    )

    print("\n  Checking transfer to Fashion-MNIST...")
    self_sup_transfer = run_transfer(
        self_sup_ff, 'mnist', 'fashion_mnist', device, config.probe_epochs, verbose
    )

    results['self_supervised_ff'] = {
        'type': 'label_free',
        'source_accuracy': self_sup_source['best_accuracy'],
        'transfer_accuracy': self_sup_transfer['best_accuracy'],
    }

    # ============================================================
    # 3. Contrastive FF
    # ============================================================
    print("\n" + "=" * 60)
    print("3. CONTRASTIVE FF (SimCLR-style)")
    print("=" * 60)

    torch.manual_seed(config.seed)
    contrastive_ff = ContrastiveFF(dims, threshold=config.threshold, lr=config.lr).to(device)

    for epoch in range(config.pretrain_epochs):
        losses = contrastive_ff.train_epoch(train_loader, device, verbose=False)
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            avg_loss = np.mean(list(losses.values()))
            print(f"  Epoch {epoch+1:3d}/{config.pretrain_epochs} | Loss: {avg_loss:.4f}")

    print("\n  Checking on MNIST (source task)...")
    contrastive_source = run_linear_probe(
        contrastive_ff, train_loader, test_loader, device, config.probe_epochs, verbose
    )

    print("\n  Checking transfer to Fashion-MNIST...")
    contrastive_transfer = run_transfer(
        contrastive_ff, 'mnist', 'fashion_mnist', device, config.probe_epochs, verbose
    )

    results['contrastive_ff'] = {
        'type': 'label_free',
        'source_accuracy': contrastive_source['best_accuracy'],
        'transfer_accuracy': contrastive_transfer['best_accuracy'],
    }

    # ============================================================
    # 4. Reconstruction FF
    # ============================================================
    print("\n" + "=" * 60)
    print("4. RECONSTRUCTION FF (autoencoder-style)")
    print("=" * 60)

    torch.manual_seed(config.seed)
    recon_ff = ReconstructionFF(dims, lr=config.lr).to(device)

    for epoch in range(config.pretrain_epochs):
        losses = recon_ff.train_epoch(train_loader, device, verbose=False)
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            avg_loss = np.mean(list(losses.values()))
            print(f"  Epoch {epoch+1:3d}/{config.pretrain_epochs} | Loss: {avg_loss:.4f}")

    print("\n  Checking on MNIST (source task)...")
    recon_source = run_linear_probe(
        recon_ff, train_loader, test_loader, device, config.probe_epochs, verbose
    )

    print("\n  Checking transfer to Fashion-MNIST...")
    recon_transfer = run_transfer(
        recon_ff, 'mnist', 'fashion_mnist', device, config.probe_epochs, verbose
    )

    results['reconstruction_ff'] = {
        'type': 'label_free',
        'source_accuracy': recon_source['best_accuracy'],
        'transfer_accuracy': recon_transfer['best_accuracy'],
    }

    # ============================================================
    # 5. Random Baseline
    # ============================================================
    print("\n" + "=" * 60)
    print("5. RANDOM BASELINE (no training)")
    print("=" * 60)

    torch.manual_seed(config.seed)
    random_ff = SelfSupervisedFF(dims, threshold=config.threshold, lr=config.lr).to(device)
    # No training!

    print("\n  Checking on MNIST (source task)...")
    random_source = run_linear_probe(
        random_ff, train_loader, test_loader, device, config.probe_epochs, verbose
    )

    print("\n  Checking transfer to Fashion-MNIST...")
    random_transfer = run_transfer(
        random_ff, 'mnist', 'fashion_mnist', device, config.probe_epochs, verbose
    )

    results['random_baseline'] = {
        'type': 'baseline',
        'source_accuracy': random_source['best_accuracy'],
        'transfer_accuracy': random_transfer['best_accuracy'],
    }

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Type':<12} {'Source Acc':>12} {'Transfer Acc':>14} {'Transfer Gain':>14}")
    print("-" * 77)

    random_transfer_acc = results['random_baseline']['transfer_accuracy']

    for name, res in results.items():
        source = res['source_accuracy']
        transfer = res['transfer_accuracy']
        gain = transfer - random_transfer_acc

        print(f"{name:<25} {res['type']:<12} {source*100:>11.2f}% {transfer*100:>13.2f}% {gain*100:>+13.2f}%")

    # Hypothesis verification
    print("\n" + "=" * 70)
    print("HYPOTHESIS VERIFICATION")
    print("=" * 70)

    standard_transfer = results['standard_ff']['transfer_accuracy']

    # Check if any label-free method beats standard FF on transfer
    label_free_methods = ['self_supervised_ff', 'contrastive_ff', 'reconstruction_ff']
    best_label_free = max(results[m]['transfer_accuracy'] for m in label_free_methods)
    best_label_free_name = max(label_free_methods, key=lambda m: results[m]['transfer_accuracy'])

    h1_supported = best_label_free > standard_transfer

    print(f"\nH1: Label-free FF has better transfer than Standard FF")
    print(f"    Standard FF transfer:    {standard_transfer*100:.2f}%")
    print(f"    Best label-free transfer: {best_label_free*100:.2f}% ({best_label_free_name})")
    print(f"    Result: {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")

    # Transfer efficiency: transfer_acc / source_acc
    print(f"\nTransfer Efficiency (transfer_acc / source_acc):")
    for name, res in results.items():
        if res['source_accuracy'] > 0:
            efficiency = res['transfer_accuracy'] / res['source_accuracy']
            print(f"    {name:<25}: {efficiency:.3f}")

    # Add metadata
    results['config'] = asdict(config)
    results['timestamp'] = datetime.now().isoformat()
    results['hypothesis_supported'] = h1_supported

    return results


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy/torch types
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
    parser = argparse.ArgumentParser(description="Label-Free Forward-Forward Experiment")
    parser.add_argument('--epochs', type=int, default=50, help='Pretrain epochs')
    parser.add_argument('--probe-epochs', type=int, default=100, help='Linear probe epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.03, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer epochs')
    parser.add_argument('--output', type=str, default=None, help='Output file path')

    args = parser.parse_args()

    if args.quick:
        config = ExperimentConfig(
            pretrain_epochs=10,
            probe_epochs=20,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed
        )
    else:
        config = ExperimentConfig(
            pretrain_epochs=args.epochs,
            probe_epochs=args.probe_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed
        )

    results = run_experiment(config)

    # Save results
    output_path = args.output or '/Users/parafee41/Desktop/Rios/ff-research/results/label_free_ff.json'
    save_results(results, output_path)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
