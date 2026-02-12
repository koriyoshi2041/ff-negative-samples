#!/usr/bin/env python3
"""
Knowledge Distillation for Forward-Forward
============================================

Core Hypothesis: BP features transfer better (77% vs 61%), so we can use BP as a
"teacher" to guide FF to learn more generalizable features.

Distillation Strategies:

1. Feature Alignment Distillation:
   - Train a BP teacher model first
   - FF training: L_total = L_goodness + alpha * L_align
   - L_align = MSE(FF_features, BP_features.detach())

2. Soft Label Distillation:
   - BP teacher outputs soft labels (temperature=3)
   - FF uses soft labels for embedding instead of hard labels
   - May learn smoother decision boundaries

3. Progressive Distillation:
   - Start with alpha=0 (pure FF)
   - Gradually increase alpha to 0.5
   - Let FF learn basic classification first, then align to BP

4. Layer-wise Distillation:
   - Each FF layer aligns to corresponding BP layer
   - Not just final features

Expected Outcome: Distilled FF may gain BP's transfer ability while keeping FF's
local learning properties.

Author: Clawd (for Parafee)
Date: 2026-02-09
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
# Label Embedding Utilities
# ============================================================

def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Embed label in first 10 pixels.
    CRITICAL: Use x.max() as the label value, not 1.0!
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


def overlay_soft_labels(x: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    """
    Embed soft labels (probabilities) in first 10 pixels.
    soft_labels: [batch_size, 10] probabilities summing to 1
    """
    x_ = x.clone()
    x_[:, :10] = soft_labels * x.max()
    return x_


# ============================================================
# BP Teacher Network
# ============================================================

class BPTeacher(nn.Module):
    """BP network that serves as teacher for distillation."""

    def __init__(self, dims: List[int], num_classes: int = 10):
        super().__init__()
        self.feature_layers = nn.ModuleList()

        # Build feature extraction layers
        for i in range(len(dims) - 1):
            self.feature_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.feature_layers.append(nn.ReLU())

        # Classification head
        self.classifier = nn.Linear(dims[-1], num_classes)

        # Store layer dimensions for alignment
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.feature_layers:
            x = layer(x)
        return self.classifier(x)

    def get_layer_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get features from each layer (for layer-wise distillation)."""
        features = []
        h = x
        for i, layer in enumerate(self.feature_layers):
            h = layer(h)
            # Capture features after ReLU (even indices are Linear, odd are ReLU)
            if i % 2 == 1:  # After ReLU
                features.append(h)
        return features

    def get_final_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features from last hidden layer."""
        with torch.no_grad():
            h = x
            for layer in self.feature_layers:
                h = layer(h)
            return h

    def get_soft_labels(self, x: torch.Tensor, temperature: float = 3.0) -> torch.Tensor:
        """Get soft labels (softmax with temperature)."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits / temperature, dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            preds = self.forward(x).argmax(dim=1)
            return (preds == y).float().mean().item()


def train_bp_teacher(model: BPTeacher, x_train: torch.Tensor, y_train: torch.Tensor,
                     x_test: torch.Tensor, y_test: torch.Tensor,
                     epochs: int = 50, batch_size: int = 128, lr: float = 0.001,
                     verbose: bool = True) -> Dict:
    """Train BP teacher network."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(x_train))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            model.train(False)
            train_acc = model.get_accuracy(x_train, y_train)
            test_acc = model.get_accuracy(x_test, y_test)
            print(f"  Teacher Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

    model.train(False)
    return {
        'train_acc': model.get_accuracy(x_train, y_train),
        'test_acc': model.get_accuracy(x_test, y_test)
    }


# ============================================================
# FF Layer with Distillation Support
# ============================================================

class FFLayerDistillation(nn.Module):
    """FF Layer with knowledge distillation support."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with L2 normalization."""
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """MEAN of squared activations."""
        return h.pow(2).mean(dim=1)

    def goodness_loss(self, g_pos: torch.Tensor, g_neg: torch.Tensor) -> torch.Tensor:
        """Standard FF goodness loss."""
        return torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold
        ]))).mean()

    def train_with_distillation(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        teacher_features: Optional[torch.Tensor],
        num_epochs: int = 1000,
        alpha: float = 0.0,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train layer with optional feature alignment distillation.

        Args:
            x_pos: Positive samples (correct label embedded)
            x_neg: Negative samples (wrong label embedded)
            teacher_features: Target features from BP teacher (optional)
            num_epochs: Number of training epochs
            alpha: Weight for alignment loss (0 = pure FF, 1 = pure alignment)
            verbose: Print progress
        """
        iterator = tqdm(range(num_epochs), desc="Training layer") if verbose else range(num_epochs)

        for _ in iterator:
            # Forward pass
            h_pos = self.forward(x_pos)
            h_neg = self.forward(x_neg)

            # Goodness loss
            g_pos = self.goodness(h_pos)
            g_neg = self.goodness(h_neg)
            loss_goodness = self.goodness_loss(g_pos, g_neg)

            # Feature alignment loss (if teacher features provided)
            if teacher_features is not None and alpha > 0:
                # Align positive sample features to teacher features
                # Note: We only align positive samples since teacher doesn't have negative concept
                loss_align = F.mse_loss(h_pos, teacher_features.detach())
                loss = (1 - alpha) * loss_goodness + alpha * loss_align
            else:
                loss = loss_goodness

            # Backward
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFNetworkDistillation(nn.Module):
    """FF Network with knowledge distillation support."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(FFLayerDistillation(dims[d], dims[d + 1], threshold, lr))
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def train_greedy_standard(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                               epochs_per_layer: int = 1000, verbose: bool = True):
        """Standard greedy layer-by-layer training (no distillation)."""
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\nTraining layer {i} (standard FF)...')
            h_pos, h_neg = layer.train_with_distillation(
                h_pos, h_neg, None, epochs_per_layer, alpha=0.0, verbose=verbose
            )

    def train_greedy_with_feature_alignment(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        x_raw: torch.Tensor,
        teacher: BPTeacher,
        epochs_per_layer: int = 1000,
        alpha: float = 0.3,
        verbose: bool = True
    ):
        """
        Strategy 1: Feature Alignment Distillation
        Each FF layer is trained to both maximize goodness AND align with BP features.
        """
        # Get teacher layer features
        teacher_features = teacher.get_layer_features(x_raw)

        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\nTraining layer {i} with feature alignment (alpha={alpha})...')

            # Get corresponding teacher features (if available)
            t_features = teacher_features[i] if i < len(teacher_features) else None

            h_pos, h_neg = layer.train_with_distillation(
                h_pos, h_neg, t_features, epochs_per_layer, alpha=alpha, verbose=verbose
            )

    def train_with_soft_labels(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        teacher: BPTeacher,
        epochs_per_layer: int = 1000,
        temperature: float = 3.0,
        verbose: bool = True
    ):
        """
        Strategy 2: Soft Label Distillation
        Use teacher's soft labels for embedding instead of hard labels.
        """
        # Get soft labels from teacher
        soft_labels = teacher.get_soft_labels(x_train, temperature)

        # Create positive samples with soft labels
        x_pos = overlay_soft_labels(x_train, soft_labels)

        # For negative samples, we need to perturb the soft labels
        # Option: Use shuffled soft labels
        rnd = torch.randperm(x_train.size(0))
        x_neg = overlay_soft_labels(x_train, soft_labels[rnd])

        # Train layer by layer (standard goodness training, but with soft labels)
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\nTraining layer {i} with soft labels (T={temperature})...')
            h_pos, h_neg = layer.train_with_distillation(
                h_pos, h_neg, None, epochs_per_layer, alpha=0.0, verbose=verbose
            )

    def train_progressive_distillation(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        x_raw: torch.Tensor,
        teacher: BPTeacher,
        epochs_per_layer: int = 1000,
        alpha_start: float = 0.0,
        alpha_end: float = 0.5,
        verbose: bool = True
    ):
        """
        Strategy 3: Progressive Distillation
        Start with pure FF (alpha=0), gradually increase to alpha_end.
        """
        teacher_features = teacher.get_layer_features(x_raw)

        h_pos, h_neg = x_pos, x_neg
        num_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            # Progressive alpha: increases as we go deeper
            alpha = alpha_start + (alpha_end - alpha_start) * (i / max(1, num_layers - 1))

            if verbose:
                print(f'\nTraining layer {i} with progressive alpha={alpha:.3f}...')

            t_features = teacher_features[i] if i < len(teacher_features) else None

            h_pos, h_neg = layer.train_with_distillation(
                h_pos, h_neg, t_features, epochs_per_layer, alpha=alpha, verbose=verbose
            )

    def train_layerwise_distillation(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        x_raw: torch.Tensor,
        teacher: BPTeacher,
        epochs_per_layer: int = 1000,
        alpha: float = 0.3,
        verbose: bool = True
    ):
        """
        Strategy 4: Layer-wise Distillation
        Each FF layer explicitly aligns to corresponding BP layer.
        """
        # Get all teacher layer features at once
        teacher_features = teacher.get_layer_features(x_raw)

        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\nTraining layer {i} with layer-wise alignment...')

            # Match to corresponding teacher layer
            if i < len(teacher_features):
                t_features = teacher_features[i]
                if verbose:
                    print(f'  Aligning to teacher layer {i} (shape: {t_features.shape})')
            else:
                t_features = None
                if verbose:
                    print(f'  No matching teacher layer, using pure FF')

            h_pos, h_neg = layer.train_with_distillation(
                h_pos, h_neg, t_features, epochs_per_layer, alpha=alpha, verbose=verbose
            )

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels."""
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(num_classes):
            h = overlay_y_on_x(x, torch.full((batch_size,), label, device=x.device))

            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))

            total_goodness = sum(goodness)
            goodness_per_label.append(total_goodness.unsqueeze(1))

        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute accuracy."""
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()

    def get_features(self, x: torch.Tensor, label: int = 0) -> torch.Tensor:
        """Get features (using a fixed label for consistency)."""
        with torch.no_grad():
            batch_size = x.shape[0]
            h = overlay_y_on_x(x, torch.full((batch_size,), label, dtype=torch.long, device=x.device))
            for layer in self.layers:
                h = layer(h)
            return h


# ============================================================
# Transfer Learning Evaluation
# ============================================================

class LinearHead(nn.Module):
    """Linear classification head for transfer learning."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def compute_transfer_accuracy(features_train: torch.Tensor, y_train: torch.Tensor,
                              features_test: torch.Tensor, y_test: torch.Tensor,
                              epochs: int = 100, lr: float = 0.01,
                              verbose: bool = False) -> Dict:
    """Train linear probe on frozen features and return accuracy."""
    feature_dim = features_train.shape[1]
    num_classes = int(y_train.max().item()) + 1

    head = LinearHead(feature_dim, num_classes).to(features_train.device)
    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    final_acc = 0.0
    for epoch in range(epochs):
        head.train()
        optimizer.zero_grad()
        outputs = head(features_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        head.train(False)
        with torch.no_grad():
            test_preds = head(features_test).argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()
            best_acc = max(best_acc, test_acc)
            final_acc = test_acc

    return {
        'final_accuracy': final_acc,
        'best_accuracy': best_acc
    }


# ============================================================
# Main Experiment
# ============================================================

def run_distillation_experiment(
    epochs_per_layer: int = 300,
    bp_epochs: int = 50,
    transfer_epochs: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run knowledge distillation experiments comparing different strategies.
    """
    print("="*70)
    print("KNOWLEDGE DISTILLATION FOR FORWARD-FORWARD")
    print("="*70)
    print(f"\nSettings:")
    print(f"  FF epochs per layer: {epochs_per_layer}")
    print(f"  BP epochs: {bp_epochs}")
    print(f"  Transfer epochs: {transfer_epochs}")
    print(f"  Seed: {seed}")

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"  Device: {device}")

    results = {
        'experiment': 'knowledge_distillation',
        'hypothesis': 'BP features transfer better (77% vs 61%), use BP as teacher to guide FF',
        'config': {
            'epochs_per_layer': epochs_per_layer,
            'bp_epochs': bp_epochs,
            'transfer_epochs': transfer_epochs,
            'seed': seed,
            'device': str(device),
            'architecture': [784, 500, 500],
        }
    }

    # Load data
    print("\n" + "-"*60)
    print("Loading datasets...")
    (mnist_train, mnist_train_y), (mnist_test, mnist_test_y) = get_mnist_data(device)
    (fmnist_train, fmnist_train_y), (fmnist_test, fmnist_test_y) = get_fashion_mnist_data(device)
    print(f"  MNIST: {len(mnist_train)} train, {len(mnist_test)} test")
    print(f"  Fashion-MNIST: {len(fmnist_train)} train, {len(fmnist_test)} test")

    # ============================================================
    # Phase 1: Train BP Teacher
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 1: Training BP Teacher")
    print("="*60)

    torch.manual_seed(seed)
    teacher = BPTeacher([784, 500, 500], num_classes=10).to(device)
    teacher_results = train_bp_teacher(
        teacher, mnist_train, mnist_train_y, mnist_test, mnist_test_y,
        epochs=bp_epochs, verbose=verbose
    )
    print(f"\nTeacher Results: Train={teacher_results['train_acc']*100:.2f}%, "
          f"Test={teacher_results['test_acc']*100:.2f}%")
    results['teacher'] = teacher_results

    # Teacher transfer baseline
    teacher_features_train = teacher.get_final_features(fmnist_train)
    teacher_features_test = teacher.get_final_features(fmnist_test)
    teacher_transfer = compute_transfer_accuracy(
        teacher_features_train, fmnist_train_y,
        teacher_features_test, fmnist_test_y,
        epochs=transfer_epochs
    )
    print(f"Teacher Transfer: {teacher_transfer['best_accuracy']*100:.2f}%")
    results['teacher']['transfer'] = teacher_transfer

    # Prepare positive/negative samples for FF
    x_pos = overlay_y_on_x(mnist_train, mnist_train_y)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg = overlay_y_on_x(mnist_train, mnist_train_y[rnd])

    # ============================================================
    # Phase 2: Baseline - Standard FF (No Distillation)
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 2: Baseline - Standard FF (No Distillation)")
    print("="*60)

    torch.manual_seed(seed)
    ff_baseline = FFNetworkDistillation([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    ff_baseline.train_greedy_standard(x_pos, x_neg, epochs_per_layer, verbose)

    baseline_acc = ff_baseline.get_accuracy(mnist_test, mnist_test_y)
    baseline_features_train = ff_baseline.get_features(fmnist_train)
    baseline_features_test = ff_baseline.get_features(fmnist_test)
    baseline_transfer = compute_transfer_accuracy(
        baseline_features_train, fmnist_train_y,
        baseline_features_test, fmnist_test_y,
        epochs=transfer_epochs
    )

    print(f"\nBaseline FF: Source={baseline_acc*100:.2f}%, Transfer={baseline_transfer['best_accuracy']*100:.2f}%")
    results['baseline_ff'] = {
        'source_acc': baseline_acc,
        'transfer': baseline_transfer
    }

    # ============================================================
    # Phase 3: Strategy 1 - Feature Alignment Distillation
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 3: Strategy 1 - Feature Alignment Distillation")
    print("="*60)

    alphas_to_test = [0.1, 0.3, 0.5]
    strategy1_results = {}

    for alpha in alphas_to_test:
        print(f"\n--- Testing alpha = {alpha} ---")
        torch.manual_seed(seed)
        ff_align = FFNetworkDistillation([784, 500, 500], threshold=2.0, lr=0.03).to(device)
        ff_align.train_greedy_with_feature_alignment(
            x_pos, x_neg, mnist_train, teacher, epochs_per_layer, alpha=alpha, verbose=verbose
        )

        align_acc = ff_align.get_accuracy(mnist_test, mnist_test_y)
        align_features_train = ff_align.get_features(fmnist_train)
        align_features_test = ff_align.get_features(fmnist_test)
        align_transfer = compute_transfer_accuracy(
            align_features_train, fmnist_train_y,
            align_features_test, fmnist_test_y,
            epochs=transfer_epochs
        )

        print(f"Alpha={alpha}: Source={align_acc*100:.2f}%, Transfer={align_transfer['best_accuracy']*100:.2f}%")
        strategy1_results[f'alpha_{alpha}'] = {
            'source_acc': align_acc,
            'transfer': align_transfer
        }

    results['strategy1_feature_alignment'] = strategy1_results

    # ============================================================
    # Phase 4: Strategy 2 - Soft Label Distillation
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 4: Strategy 2 - Soft Label Distillation")
    print("="*60)

    temperatures = [1.0, 3.0, 5.0]
    strategy2_results = {}

    for temp in temperatures:
        print(f"\n--- Testing temperature = {temp} ---")
        torch.manual_seed(seed)
        ff_soft = FFNetworkDistillation([784, 500, 500], threshold=2.0, lr=0.03).to(device)
        ff_soft.train_with_soft_labels(
            mnist_train, mnist_train_y, teacher, epochs_per_layer, temperature=temp, verbose=verbose
        )

        soft_acc = ff_soft.get_accuracy(mnist_test, mnist_test_y)
        soft_features_train = ff_soft.get_features(fmnist_train)
        soft_features_test = ff_soft.get_features(fmnist_test)
        soft_transfer = compute_transfer_accuracy(
            soft_features_train, fmnist_train_y,
            soft_features_test, fmnist_test_y,
            epochs=transfer_epochs
        )

        print(f"T={temp}: Source={soft_acc*100:.2f}%, Transfer={soft_transfer['best_accuracy']*100:.2f}%")
        strategy2_results[f'temp_{temp}'] = {
            'source_acc': soft_acc,
            'transfer': soft_transfer
        }

    results['strategy2_soft_labels'] = strategy2_results

    # ============================================================
    # Phase 5: Strategy 3 - Progressive Distillation
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 5: Strategy 3 - Progressive Distillation")
    print("="*60)

    alpha_ends = [0.3, 0.5, 0.7]
    strategy3_results = {}

    for alpha_end in alpha_ends:
        print(f"\n--- Testing alpha 0.0 -> {alpha_end} ---")
        torch.manual_seed(seed)
        ff_prog = FFNetworkDistillation([784, 500, 500], threshold=2.0, lr=0.03).to(device)
        ff_prog.train_progressive_distillation(
            x_pos, x_neg, mnist_train, teacher, epochs_per_layer,
            alpha_start=0.0, alpha_end=alpha_end, verbose=verbose
        )

        prog_acc = ff_prog.get_accuracy(mnist_test, mnist_test_y)
        prog_features_train = ff_prog.get_features(fmnist_train)
        prog_features_test = ff_prog.get_features(fmnist_test)
        prog_transfer = compute_transfer_accuracy(
            prog_features_train, fmnist_train_y,
            prog_features_test, fmnist_test_y,
            epochs=transfer_epochs
        )

        print(f"Progressive (0->{alpha_end}): Source={prog_acc*100:.2f}%, Transfer={prog_transfer['best_accuracy']*100:.2f}%")
        strategy3_results[f'alpha_0_to_{alpha_end}'] = {
            'source_acc': prog_acc,
            'transfer': prog_transfer
        }

    results['strategy3_progressive'] = strategy3_results

    # ============================================================
    # Phase 6: Strategy 4 - Layer-wise Distillation
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 6: Strategy 4 - Layer-wise Distillation")
    print("="*60)

    torch.manual_seed(seed)
    ff_layerwise = FFNetworkDistillation([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    ff_layerwise.train_layerwise_distillation(
        x_pos, x_neg, mnist_train, teacher, epochs_per_layer, alpha=0.3, verbose=verbose
    )

    layerwise_acc = ff_layerwise.get_accuracy(mnist_test, mnist_test_y)
    layerwise_features_train = ff_layerwise.get_features(fmnist_train)
    layerwise_features_test = ff_layerwise.get_features(fmnist_test)
    layerwise_transfer = compute_transfer_accuracy(
        layerwise_features_train, fmnist_train_y,
        layerwise_features_test, fmnist_test_y,
        epochs=transfer_epochs
    )

    print(f"Layer-wise: Source={layerwise_acc*100:.2f}%, Transfer={layerwise_transfer['best_accuracy']*100:.2f}%")
    results['strategy4_layerwise'] = {
        'source_acc': layerwise_acc,
        'transfer': layerwise_transfer
    }

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    print(f"\n{'Strategy':<40} {'Source Acc':>12} {'Transfer Acc':>14}")
    print("-" * 68)
    print(f"{'BP Teacher':<40} {teacher_results['test_acc']*100:>11.2f}% {teacher_transfer['best_accuracy']*100:>13.2f}%")
    print(f"{'FF Baseline (no distillation)':<40} {baseline_acc*100:>11.2f}% {baseline_transfer['best_accuracy']*100:>13.2f}%")

    # Find best from each strategy
    print("-" * 68)

    # Strategy 1
    best_s1 = max(strategy1_results.items(), key=lambda x: x[1]['transfer']['best_accuracy'])
    print(f"{'S1: Feature Align (' + best_s1[0] + ')':<40} {best_s1[1]['source_acc']*100:>11.2f}% {best_s1[1]['transfer']['best_accuracy']*100:>13.2f}%")

    # Strategy 2
    best_s2 = max(strategy2_results.items(), key=lambda x: x[1]['transfer']['best_accuracy'])
    print(f"{'S2: Soft Labels (' + best_s2[0] + ')':<40} {best_s2[1]['source_acc']*100:>11.2f}% {best_s2[1]['transfer']['best_accuracy']*100:>13.2f}%")

    # Strategy 3
    best_s3 = max(strategy3_results.items(), key=lambda x: x[1]['transfer']['best_accuracy'])
    print(f"{'S3: Progressive (' + best_s3[0] + ')':<40} {best_s3[1]['source_acc']*100:>11.2f}% {best_s3[1]['transfer']['best_accuracy']*100:>13.2f}%")

    # Strategy 4
    print(f"{'S4: Layer-wise':<40} {layerwise_acc*100:>11.2f}% {layerwise_transfer['best_accuracy']*100:>13.2f}%")

    # Find overall best
    all_transfers = [
        ('Baseline FF', baseline_transfer['best_accuracy']),
        ('S1: Feature Align', best_s1[1]['transfer']['best_accuracy']),
        ('S2: Soft Labels', best_s2[1]['transfer']['best_accuracy']),
        ('S3: Progressive', best_s3[1]['transfer']['best_accuracy']),
        ('S4: Layer-wise', layerwise_transfer['best_accuracy']),
    ]
    best_ff = max(all_transfers, key=lambda x: x[1])

    print("-" * 68)
    print(f"\nBest FF Transfer: {best_ff[0]} ({best_ff[1]*100:.2f}%)")
    print(f"BP Teacher Transfer: {teacher_transfer['best_accuracy']*100:.2f}%")
    print(f"Improvement over Baseline: {(best_ff[1] - baseline_transfer['best_accuracy'])*100:.2f}%")
    print(f"Gap vs BP Teacher: {(best_ff[1] - teacher_transfer['best_accuracy'])*100:.2f}%")

    results['summary'] = {
        'best_ff_strategy': best_ff[0],
        'best_ff_transfer': best_ff[1],
        'baseline_transfer': baseline_transfer['best_accuracy'],
        'teacher_transfer': teacher_transfer['best_accuracy'],
        'improvement_over_baseline': best_ff[1] - baseline_transfer['best_accuracy'],
        'gap_vs_teacher': best_ff[1] - teacher_transfer['best_accuracy']
    }

    results['timestamp'] = datetime.now().isoformat()

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
    parser = argparse.ArgumentParser(description="Knowledge Distillation for FF")
    parser.add_argument('--ff-epochs', type=int, default=300,
                        help='FF epochs per layer')
    parser.add_argument('--bp-epochs', type=int, default=50,
                        help='BP teacher training epochs')
    parser.add_argument('--transfer-epochs', type=int, default=100,
                        help='Transfer head training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (100 epochs/layer)')
    args = parser.parse_args()

    ff_epochs = 100 if args.quick else args.ff_epochs

    results = run_distillation_experiment(
        epochs_per_layer=ff_epochs,
        bp_epochs=args.bp_epochs,
        transfer_epochs=args.transfer_epochs,
        seed=args.seed
    )

    output_path = './results/knowledge_distillation.json'
    save_results(results, output_path)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
