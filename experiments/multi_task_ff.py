#!/usr/bin/env python3
"""
Multi-Task Pretraining Strategy for Forward-Forward
====================================================

Core Hypothesis: Single-task pretraining leads to over-specialized features.
Multi-task pretraining may force FF to learn more generalizable features.

Experiment Design:

1. Joint MNIST + Fashion-MNIST Pretraining:
   - Simultaneously train on both datasets
   - Alternating batches: MNIST batch, Fashion batch, MNIST batch...
   - Merged label space (0-9: MNIST digits, 10-19: Fashion categories)
   - Then fine-tune only on Fashion-MNIST

2. Sequential Multi-task:
   - Train on MNIST first
   - Then on KMNIST
   - Then on EMNIST
   - Finally transfer to Fashion-MNIST
   - Test if this produces more general features

3. Auxiliary Task Training:
   - Main task: MNIST classification
   - Auxiliary task: Rotation prediction (0, 90, 180, 270 degrees)
   - Simultaneous optimization of both objectives
   - Auxiliary task forces learning of rotation-invariant features

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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


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
    """Load dataset to device memory."""
    dataset_configs = {
        'mnist': ((0.1307,), (0.3081,), datasets.MNIST),
        'fashion_mnist': ((0.2860,), (0.3530,), datasets.FashionMNIST),
        'kmnist': ((0.1904,), (0.3475,), datasets.KMNIST),
        'emnist': ((0.1722,), (0.3309,), datasets.EMNIST),
    }

    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    mean, std, dataset_class = dataset_configs[dataset_name]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    kwargs = {'split': 'letters'} if dataset_name == 'emnist' else {}

    train_dataset = dataset_class('./data', train=True, download=True, transform=transform, **kwargs)
    test_dataset = dataset_class('./data', train=False, download=True, transform=transform, **kwargs)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    return (x_train.to(device), y_train.to(device)), (x_test.to(device), y_test.to(device))


def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """
    Embed label in first N pixels using x.max().
    Supports variable number of classes for merged label spaces.
    """
    x_ = x.clone()
    x_[:, :num_classes] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


def rotate_image(x: torch.Tensor, angle: int) -> torch.Tensor:
    """
    Rotate flattened 28x28 image by specified angle.
    angle: 0, 90, 180, or 270 degrees
    """
    batch_size = x.shape[0]
    img = x.view(batch_size, 28, 28)

    if angle == 0:
        rotated = img
    elif angle == 90:
        rotated = torch.rot90(img, k=1, dims=(1, 2))
    elif angle == 180:
        rotated = torch.rot90(img, k=2, dims=(1, 2))
    elif angle == 270:
        rotated = torch.rot90(img, k=3, dims=(1, 2))
    else:
        raise ValueError(f"Unsupported angle: {angle}")

    return rotated.view(batch_size, 784)


# ============================================================
# Forward-Forward Layer
# ============================================================

class FFLayer(nn.Module):
    """Forward-Forward Layer with correct implementation."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with L2 normalization."""
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """MEAN of squared activations."""
        return h.pow(2).mean(dim=1)

    def ff_loss(self, g_pos: torch.Tensor, g_neg: torch.Tensor) -> torch.Tensor:
        """Standard FF loss."""
        loss = torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold
        ]))).mean()
        return loss


# ============================================================
# Forward-Forward Network
# ============================================================

class FFNetwork(nn.Module):
    """Forward-Forward Network supporting multi-task training."""

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

    def train_standard_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                               epochs_per_layer: int = 500, verbose: bool = True):
        """Standard greedy layer-by-layer training."""
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'    Training layer {i}...')

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

    def train_alternating_multitask(self, datasets: List[Tuple[torch.Tensor, torch.Tensor]],
                                     epochs_per_layer: int = 500,
                                     num_classes: int = 20,
                                     verbose: bool = True):
        """
        Multi-task training with alternating datasets.

        Args:
            datasets: List of (x_pos, x_neg) tuples for each dataset
            epochs_per_layer: Total epochs per layer (divided among datasets)
            num_classes: Total number of classes in merged label space
        """
        num_datasets = len(datasets)
        epochs_per_dataset = epochs_per_layer // num_datasets

        h_datasets = [(d[0], d[1]) for d in datasets]

        for layer_idx, layer in enumerate(self.layers):
            if verbose:
                print(f'    Training layer {layer_idx} (alternating {num_datasets} datasets)...')

            for epoch in range(epochs_per_dataset):
                for dataset_idx, (h_pos, h_neg) in enumerate(h_datasets):
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

            # Update hidden states for next layer
            h_datasets = [(layer(h[0]).detach(), layer(h[1]).detach()) for h in h_datasets]

    def train_sequential_multitask(self, dataset_sequence: List[Tuple[torch.Tensor, torch.Tensor, str]],
                                    epochs_per_layer: int = 500,
                                    verbose: bool = True):
        """
        Sequential multi-task training across multiple datasets.

        Args:
            dataset_sequence: List of (x_pos, x_neg, dataset_name) tuples
        """
        for dataset_idx, (x_pos, x_neg, dataset_name) in enumerate(dataset_sequence):
            if verbose:
                print(f'\n  === Training on dataset {dataset_idx+1}/{len(dataset_sequence)}: {dataset_name} ===')

            h_pos, h_neg = x_pos, x_neg

            for layer_idx, layer in enumerate(self.layers):
                if verbose:
                    print(f'    Training layer {layer_idx}...')

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

    def train_with_auxiliary_task(self, x: torch.Tensor, y: torch.Tensor,
                                   epochs_per_layer: int = 500,
                                   aux_weight: float = 0.5,
                                   verbose: bool = True):
        """
        Train with auxiliary rotation prediction task.

        Main task: Classification
        Auxiliary task: Rotation prediction (0, 90, 180, 270 degrees)

        Args:
            x: Input images (flattened)
            y: Labels
            aux_weight: Weight of auxiliary task loss (0 to 1)
        """
        batch_size = x.shape[0]

        # Generate rotated versions and auxiliary labels
        angles = [0, 90, 180, 270]
        x_all_rotations = []
        aux_labels_all = []

        for rot_idx, angle in enumerate(angles):
            x_rotated = rotate_image(x, angle)
            x_all_rotations.append(x_rotated)
            aux_labels_all.append(torch.full((batch_size,), rot_idx, device=x.device))

        # Concatenate all rotations
        x_augmented = torch.cat(x_all_rotations, dim=0)
        y_augmented = y.repeat(4)
        aux_labels = torch.cat(aux_labels_all, dim=0)

        # Create positive samples (correct classification label)
        x_pos_main = overlay_y_on_x(x_augmented, y_augmented, num_classes=10)

        # Create positive samples for auxiliary task (correct rotation label)
        # Use separate embedding space (pixels 10-13 for rotation)
        x_pos_aux = x_pos_main.clone()
        x_pos_aux[:, 10:14] *= 0.0
        x_pos_aux[range(len(aux_labels)), 10 + aux_labels] = x_augmented.max()

        # Create negative samples (shuffled labels for both tasks)
        rnd_main = torch.randperm(x_augmented.size(0))
        rnd_aux = torch.randperm(x_augmented.size(0))

        x_neg_main = overlay_y_on_x(x_augmented, y_augmented[rnd_main], num_classes=10)
        x_neg_aux = x_neg_main.clone()
        x_neg_aux[:, 10:14] *= 0.0
        x_neg_aux[range(len(aux_labels)), 10 + aux_labels[rnd_aux]] = x_augmented.max()

        h_pos_main, h_neg_main = x_pos_main, x_neg_main
        h_pos_aux, h_neg_aux = x_pos_aux, x_neg_aux

        for layer_idx, layer in enumerate(self.layers):
            if verbose:
                print(f'    Training layer {layer_idx} (main + auxiliary task)...')

            for epoch in range(epochs_per_layer):
                # Main task loss
                out_pos_main = layer(h_pos_main)
                out_neg_main = layer(h_neg_main)
                g_pos_main = layer.goodness(out_pos_main)
                g_neg_main = layer.goodness(out_neg_main)
                loss_main = layer.ff_loss(g_pos_main, g_neg_main)

                # Auxiliary task loss
                out_pos_aux = layer(h_pos_aux)
                out_neg_aux = layer(h_neg_aux)
                g_pos_aux = layer.goodness(out_pos_aux)
                g_neg_aux = layer.goodness(out_neg_aux)
                loss_aux = layer.ff_loss(g_pos_aux, g_neg_aux)

                # Combined loss
                loss = (1 - aux_weight) * loss_main + aux_weight * loss_aux

                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()

                if verbose and (epoch + 1) % 100 == 0:
                    print(f"      Epoch {epoch+1}: loss_main={loss_main.item():.4f}, "
                          f"loss_aux={loss_aux.item():.4f}, combined={loss.item():.4f}")

            h_pos_main = layer(h_pos_main).detach()
            h_neg_main = layer(h_neg_main).detach()
            h_pos_aux = layer(h_pos_aux).detach()
            h_neg_aux = layer(h_neg_aux).detach()

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels."""
        goodness_per_label = []

        for label in range(num_classes):
            h = overlay_y_on_x(x, torch.full((x.shape[0],), label, device=x.device), num_classes)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))
            goodness_per_label.append(sum(goodness).unsqueeze(1))

        return torch.cat(goodness_per_label, dim=1).argmax(dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor, num_classes: int = 10) -> float:
        """Compute accuracy."""
        predictions = self.predict(x, num_classes)
        return (predictions == y).float().mean().item()

    def get_features(self, x: torch.Tensor, label: int = 0, num_classes: int = 10) -> torch.Tensor:
        """Get features with label embedding."""
        with torch.no_grad():
            h = overlay_y_on_x(x, torch.full((x.shape[0],), label, dtype=torch.long, device=x.device), num_classes)
            for layer in self.layers:
                h = layer(h)
            return h


# ============================================================
# Linear Head for Transfer Learning
# ============================================================

class LinearHead(nn.Module):
    """Linear classification head."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_head(features_train: torch.Tensor, y_train: torch.Tensor,
                      features_test: torch.Tensor, y_test: torch.Tensor,
                      epochs: int = 50, batch_size: int = 256, lr: float = 0.01) -> Dict:
    """Train linear head on frozen features."""
    device = features_train.device
    feature_dim = features_train.shape[1]
    num_classes = int(y_train.max().item()) + 1

    head = LinearHead(feature_dim, num_classes).to(device)
    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0

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
            test_preds = head(features_test).argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()
            best_test_acc = max(best_test_acc, test_acc)

    return {'best_test_acc': best_test_acc}


# ============================================================
# Experiment Functions
# ============================================================

def run_baseline_experiment(device: torch.device, epochs_per_layer: int, seed: int) -> Dict:
    """Baseline: Standard FF trained on MNIST, transfer to Fashion-MNIST."""
    print("\n" + "="*60)
    print("BASELINE: Standard FF (MNIST) -> Fashion-MNIST")
    print("="*60)

    torch.manual_seed(seed)

    # Load data
    (mnist_train, mnist_y), (mnist_test, mnist_test_y) = get_dataset('mnist', device)
    (fmnist_train, fmnist_y), (fmnist_test, fmnist_test_y) = get_dataset('fashion_mnist', device)

    # Prepare pos/neg samples
    x_pos = overlay_y_on_x(mnist_train, mnist_y)
    rnd = torch.randperm(mnist_train.size(0))
    x_neg = overlay_y_on_x(mnist_train, mnist_y[rnd])

    # Train model
    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    start_time = time.time()
    model.train_standard_greedy(x_pos, x_neg, epochs_per_layer, verbose=True)
    train_time = time.time() - start_time

    # Evaluate on source
    source_acc = model.get_accuracy(mnist_test, mnist_test_y)
    print(f"\n  Source (MNIST) accuracy: {source_acc*100:.2f}%")

    # Transfer to Fashion-MNIST
    features_train = model.get_features(fmnist_train)
    features_test = model.get_features(fmnist_test)
    transfer_result = train_linear_head(features_train, fmnist_y, features_test, fmnist_test_y)

    print(f"  Transfer (Fashion-MNIST) accuracy: {transfer_result['best_test_acc']*100:.2f}%")

    return {
        'source_accuracy': source_acc,
        'transfer_accuracy': transfer_result['best_test_acc'],
        'train_time': train_time
    }


def run_joint_pretraining_experiment(device: torch.device, epochs_per_layer: int, seed: int) -> Dict:
    """
    Experiment 1: Joint MNIST + Fashion-MNIST Pretraining.

    - Train on both datasets simultaneously with alternating batches
    - Merged label space: 0-9 MNIST, 10-19 Fashion-MNIST
    - Then fine-tune/transfer on Fashion-MNIST
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Joint MNIST + Fashion-MNIST Pretraining")
    print("="*60)

    torch.manual_seed(seed)

    # Load data
    (mnist_train, mnist_y), (mnist_test, mnist_test_y) = get_dataset('mnist', device)
    (fmnist_train, fmnist_y), (fmnist_test, fmnist_test_y) = get_dataset('fashion_mnist', device)

    # Shift Fashion-MNIST labels to 10-19
    fmnist_y_shifted = fmnist_y + 10
    fmnist_test_y_shifted = fmnist_test_y + 10

    # Use 20 classes for merged label space
    num_classes = 20

    # Prepare pos/neg samples for MNIST
    x_pos_mnist = overlay_y_on_x(mnist_train, mnist_y, num_classes)
    rnd_mnist = torch.randperm(mnist_train.size(0))
    x_neg_mnist = overlay_y_on_x(mnist_train, mnist_y[rnd_mnist], num_classes)

    # Prepare pos/neg samples for Fashion-MNIST (shifted labels)
    x_pos_fmnist = overlay_y_on_x(fmnist_train, fmnist_y_shifted, num_classes)
    rnd_fmnist = torch.randperm(fmnist_train.size(0))
    x_neg_fmnist = overlay_y_on_x(fmnist_train, fmnist_y_shifted[rnd_fmnist], num_classes)

    # Train model with alternating datasets
    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    start_time = time.time()
    model.train_alternating_multitask(
        datasets=[(x_pos_mnist, x_neg_mnist), (x_pos_fmnist, x_neg_fmnist)],
        epochs_per_layer=epochs_per_layer,
        num_classes=num_classes,
        verbose=True
    )
    train_time = time.time() - start_time

    # Evaluate on both source tasks
    mnist_acc = model.get_accuracy(mnist_test, mnist_test_y, num_classes)
    fmnist_acc_joint = model.get_accuracy(fmnist_test, fmnist_test_y_shifted, num_classes)

    print(f"\n  Joint model MNIST accuracy: {mnist_acc*100:.2f}%")
    print(f"  Joint model Fashion-MNIST accuracy (shifted labels): {fmnist_acc_joint*100:.2f}%")

    # Also evaluate transfer with linear head (original Fashion-MNIST labels 0-9)
    features_train = model.get_features(fmnist_train, label=10, num_classes=num_classes)
    features_test = model.get_features(fmnist_test, label=10, num_classes=num_classes)
    transfer_result = train_linear_head(features_train, fmnist_y, features_test, fmnist_test_y)

    print(f"  Transfer (Fashion-MNIST, original labels) accuracy: {transfer_result['best_test_acc']*100:.2f}%")

    return {
        'mnist_accuracy': mnist_acc,
        'fashion_mnist_accuracy_joint': fmnist_acc_joint,
        'transfer_accuracy': transfer_result['best_test_acc'],
        'train_time': train_time
    }


def run_sequential_pretraining_experiment(device: torch.device, epochs_per_layer: int, seed: int) -> Dict:
    """
    Experiment 2: Sequential Multi-task Pretraining.

    Train sequentially on: MNIST -> KMNIST -> Fashion-MNIST (transfer target)
    Test if this produces more general features.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Sequential Multi-task (MNIST -> KMNIST -> Transfer)")
    print("="*60)

    torch.manual_seed(seed)

    # Load all datasets
    (mnist_train, mnist_y), (mnist_test, mnist_test_y) = get_dataset('mnist', device)
    (kmnist_train, kmnist_y), (kmnist_test, kmnist_test_y) = get_dataset('kmnist', device)
    (fmnist_train, fmnist_y), (fmnist_test, fmnist_test_y) = get_dataset('fashion_mnist', device)

    # Prepare pos/neg samples for each dataset
    def prepare_samples(x, y):
        x_pos = overlay_y_on_x(x, y)
        rnd = torch.randperm(x.size(0))
        x_neg = overlay_y_on_x(x, y[rnd])
        return x_pos, x_neg

    x_pos_mnist, x_neg_mnist = prepare_samples(mnist_train, mnist_y)
    x_pos_kmnist, x_neg_kmnist = prepare_samples(kmnist_train, kmnist_y)

    # Train model sequentially
    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Reduce epochs per dataset to keep total training time comparable
    epochs_per_dataset = epochs_per_layer // 2

    start_time = time.time()
    model.train_sequential_multitask(
        dataset_sequence=[
            (x_pos_mnist, x_neg_mnist, 'MNIST'),
            (x_pos_kmnist, x_neg_kmnist, 'KMNIST'),
        ],
        epochs_per_layer=epochs_per_dataset,
        verbose=True
    )
    train_time = time.time() - start_time

    # Evaluate on source tasks
    mnist_acc = model.get_accuracy(mnist_test, mnist_test_y)
    kmnist_acc = model.get_accuracy(kmnist_test, kmnist_test_y)

    print(f"\n  MNIST accuracy (after sequential training): {mnist_acc*100:.2f}%")
    print(f"  KMNIST accuracy: {kmnist_acc*100:.2f}%")

    # Transfer to Fashion-MNIST
    features_train = model.get_features(fmnist_train)
    features_test = model.get_features(fmnist_test)
    transfer_result = train_linear_head(features_train, fmnist_y, features_test, fmnist_test_y)

    print(f"  Transfer (Fashion-MNIST) accuracy: {transfer_result['best_test_acc']*100:.2f}%")

    return {
        'mnist_accuracy': mnist_acc,
        'kmnist_accuracy': kmnist_acc,
        'transfer_accuracy': transfer_result['best_test_acc'],
        'train_time': train_time
    }


def run_auxiliary_task_experiment(device: torch.device, epochs_per_layer: int, seed: int,
                                   aux_weight: float = 0.3) -> Dict:
    """
    Experiment 3: Auxiliary Task Training.

    Main task: MNIST classification
    Auxiliary task: Rotation prediction (0, 90, 180, 270 degrees)

    The auxiliary task forces learning of orientation-invariant features.
    """
    print("\n" + "="*60)
    print(f"EXPERIMENT 3: Auxiliary Task Training (aux_weight={aux_weight})")
    print("="*60)

    torch.manual_seed(seed)

    # Load data
    (mnist_train, mnist_y), (mnist_test, mnist_test_y) = get_dataset('mnist', device)
    (fmnist_train, fmnist_y), (fmnist_test, fmnist_test_y) = get_dataset('fashion_mnist', device)

    # Use smaller subset due to 4x data augmentation
    # This keeps training time comparable
    subset_size = min(12500, len(mnist_train))
    indices = torch.randperm(len(mnist_train))[:subset_size]
    mnist_train_subset = mnist_train[indices]
    mnist_y_subset = mnist_y[indices]

    # Train model with auxiliary task
    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    start_time = time.time()
    model.train_with_auxiliary_task(
        mnist_train_subset, mnist_y_subset,
        epochs_per_layer=epochs_per_layer,
        aux_weight=aux_weight,
        verbose=True
    )
    train_time = time.time() - start_time

    # Evaluate on source
    source_acc = model.get_accuracy(mnist_test, mnist_test_y)
    print(f"\n  Source (MNIST) accuracy: {source_acc*100:.2f}%")

    # Test rotation invariance: evaluate on rotated images
    rotation_accs = {}
    for angle in [0, 90, 180, 270]:
        mnist_test_rotated = rotate_image(mnist_test, angle)
        rot_acc = model.get_accuracy(mnist_test_rotated, mnist_test_y)
        rotation_accs[angle] = rot_acc
        print(f"  MNIST accuracy ({angle}deg): {rot_acc*100:.2f}%")

    # Transfer to Fashion-MNIST
    features_train = model.get_features(fmnist_train)
    features_test = model.get_features(fmnist_test)
    transfer_result = train_linear_head(features_train, fmnist_y, features_test, fmnist_test_y)

    print(f"  Transfer (Fashion-MNIST) accuracy: {transfer_result['best_test_acc']*100:.2f}%")

    return {
        'source_accuracy': source_acc,
        'rotation_invariance': rotation_accs,
        'transfer_accuracy': transfer_result['best_test_acc'],
        'aux_weight': aux_weight,
        'train_time': train_time
    }


# ============================================================
# Main Experiment Runner
# ============================================================

def run_multi_task_experiments(epochs_per_layer: int = 300, seed: int = 42) -> Dict[str, Any]:
    """Run all multi-task pretraining experiments."""

    print("="*70)
    print("MULTI-TASK PRETRAINING STRATEGY EXPERIMENTS")
    print("="*70)

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Epochs per layer: {epochs_per_layer}")
    print(f"Seed: {seed}")

    results = {
        'experiment': 'Multi-Task Pretraining Strategy',
        'config': {
            'epochs_per_layer': epochs_per_layer,
            'seed': seed,
            'device': str(device),
            'architecture': [784, 500, 500]
        },
        'timestamp': datetime.now().isoformat()
    }

    # Run all experiments
    results['baseline'] = run_baseline_experiment(device, epochs_per_layer, seed)
    results['joint_pretraining'] = run_joint_pretraining_experiment(device, epochs_per_layer, seed)
    results['sequential_pretraining'] = run_sequential_pretraining_experiment(device, epochs_per_layer, seed)
    results['auxiliary_task_0.3'] = run_auxiliary_task_experiment(device, epochs_per_layer, seed, aux_weight=0.3)
    results['auxiliary_task_0.5'] = run_auxiliary_task_experiment(device, epochs_per_layer, seed, aux_weight=0.5)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Strategy':<45} {'Source':>10} {'Transfer':>10}")
    print("-"*65)

    baseline_transfer = results['baseline']['transfer_accuracy']
    print(f"{'Baseline (Single-task MNIST)':<45} {results['baseline']['source_accuracy']*100:>9.2f}% {baseline_transfer*100:>9.2f}%")
    print(f"{'Joint MNIST + Fashion-MNIST':<45} {results['joint_pretraining']['mnist_accuracy']*100:>9.2f}% {results['joint_pretraining']['transfer_accuracy']*100:>9.2f}%")
    print(f"{'Sequential MNIST -> KMNIST':<45} {results['sequential_pretraining']['kmnist_accuracy']*100:>9.2f}% {results['sequential_pretraining']['transfer_accuracy']*100:>9.2f}%")
    print(f"{'Auxiliary Task (weight=0.3)':<45} {results['auxiliary_task_0.3']['source_accuracy']*100:>9.2f}% {results['auxiliary_task_0.3']['transfer_accuracy']*100:>9.2f}%")
    print(f"{'Auxiliary Task (weight=0.5)':<45} {results['auxiliary_task_0.5']['source_accuracy']*100:>9.2f}% {results['auxiliary_task_0.5']['transfer_accuracy']*100:>9.2f}%")

    # Analysis
    results['analysis'] = {
        'joint_improvement': results['joint_pretraining']['transfer_accuracy'] - baseline_transfer,
        'sequential_improvement': results['sequential_pretraining']['transfer_accuracy'] - baseline_transfer,
        'auxiliary_0.3_improvement': results['auxiliary_task_0.3']['transfer_accuracy'] - baseline_transfer,
        'auxiliary_0.5_improvement': results['auxiliary_task_0.5']['transfer_accuracy'] - baseline_transfer,
    }

    print("\n" + "-"*60)
    print("IMPROVEMENT OVER BASELINE:")
    print("-"*60)

    for strategy, improvement in results['analysis'].items():
        sign = '+' if improvement > 0 else ''
        print(f"  {strategy}: {sign}{improvement*100:.2f}%")

    # Find best strategy
    best_strategy = max(results['analysis'], key=results['analysis'].get)
    results['analysis']['best_strategy'] = best_strategy

    print(f"\n  Best strategy: {best_strategy}")

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

    parser = argparse.ArgumentParser(description='Multi-Task Pretraining Strategy Experiments')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Epochs per layer (default: 300)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (100 epochs)')
    args = parser.parse_args()

    epochs = 100 if args.quick else args.epochs

    results = run_multi_task_experiments(epochs_per_layer=epochs, seed=args.seed)

    output_path = str(Path(__file__).parent.parent / 'results' / 'multi_task_pretraining.json')
    save_results(results, output_path)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
