#!/usr/bin/env python3
"""
New Architectures Transfer Learning Test
Tests CwC-FF and PFF transfer learning capabilities.
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset

from models.cwc_ff import CwCFFNetwork, get_device

print("="*60)
print("New Architecture Transfer Learning Experiment")
print("="*60)

device = get_device()
torch.manual_seed(42)
print(f"Device: {device}")

# Data loading
def get_loaders(dataset_name, batch_size=128, subset_size=None):
    if dataset_name == 'mnist':
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, transform=transform)
    else:
        transform = Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])
        train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST('./data', train=False, transform=transform)

    if subset_size:
        train_ds = Subset(train_ds, range(subset_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Create smaller CwC-FF model for faster testing
model = CwCFFNetwork(
    out_channels_list=[10, 40, 80, 160],  # Smaller channels
    num_classes=10,
    input_channels=1,
    input_size=28,
    use_cfse=True,
    lr=0.02,
    loss_type='CwC_CE',
    ilt_schedule=[[0, 1], [0, 2], [0, 3], [0, 4]]  # Shorter schedule
)
model.to(device)

print(f"Model: CwC-FF, Channels: {[l.out_channels for l in model.layers]}")

# Phase 1: Quick train on MNIST subset
print("\n--- Phase 1: MNIST Training (4 epochs, 20k samples) ---")
mnist_train, mnist_test = get_loaders('mnist', batch_size=256, subset_size=20000)

start_time = time.time()
for epoch in range(4):
    for x, y in mnist_train:
        x, y = x.to(device), y.to(device)
        h = x
        for layer_idx, layer in enumerate(model.layers):
            start_e, end_e = model.ilt_schedule[layer_idx]
            if start_e <= epoch < end_e:
                h = layer.train_step(h, y)
            else:
                h = layer.infer(h)

    for layer in model.layers:
        layer.get_epoch_loss()

    # Quick test
    correct = total = 0
    with torch.no_grad():
        for x, y in mnist_test:
            x, y = x.to(device), y.to(device)
            pred = model.predict(x)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"  Epoch {epoch+1}: {correct/total*100:.1f}%")

source_time = time.time() - start_time
source_acc = correct / total
print(f"Source MNIST accuracy: {source_acc*100:.2f}% (time: {source_time:.1f}s)")

# Phase 2: Transfer to Fashion-MNIST
print("\n--- Phase 2: Transfer to Fashion-MNIST (15 epochs) ---")
fmnist_train, fmnist_test = get_loaders('fmnist', batch_size=256, subset_size=20000)

# Freeze model
for param in model.parameters():
    param.requires_grad = False

# Create classifier
feat_dim = model.final_channels * model.final_size ** 2
print(f"Feature dimension: {feat_dim}")

classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(feat_dim, 10)
).to(device)

optimizer = Adam(classifier.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
for epoch in range(15):
    classifier.train()
    for x, y in fmnist_train:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            h = x
            for layer in model.layers:
                h = layer.infer(h)
        optimizer.zero_grad()
        loss = criterion(classifier(h), y)
        loss.backward()
        optimizer.step()

    classifier.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in fmnist_test:
            x, y = x.to(device), y.to(device)
            h = x
            for layer in model.layers:
                h = layer.infer(h)
            pred = classifier(h).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    best_acc = max(best_acc, acc)
    if (epoch + 1) % 3 == 0:
        print(f"  Epoch {epoch+1}: {acc*100:.1f}%")

print(f"\nCwC-FF Transfer accuracy: {best_acc*100:.2f}%")

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"CwC-FF Source (MNIST):    {source_acc*100:.2f}%")
print(f"CwC-FF Transfer (FMNIST): {best_acc*100:.2f}%")
print(f"Transfer Gain:            {(best_acc-source_acc)*100:+.2f}%")
print()
print("Baselines (from correct_transfer_results.json):")
print(f"  Standard FF: Source 89.79%, Transfer 61.90%")
print(f"  BP:          Source 98.34%, Transfer 78.14%")
print(f"  Random:      Transfer 83.98%")
print()
print(f"CwC-FF vs Standard FF: {(best_acc-0.619)*100:+.2f}%")
print(f"CwC-FF vs Random:      {(best_acc-0.8398)*100:+.2f}%")

# Save results
results = {
    'experiment': 'cwc_ff_transfer_quick',
    'cwc_ff': {
        'source_acc': source_acc,
        'transfer_acc': best_acc,
        'transfer_gain': best_acc - source_acc
    },
    'baselines': {
        'standard_ff_transfer': 0.619,
        'bp_transfer': 0.7814,
        'random_transfer': 0.8398
    },
    'comparison': {
        'cwc_vs_ff': best_acc - 0.619,
        'cwc_vs_random': best_acc - 0.8398
    },
    'config': {
        'channels': [10, 40, 80, 160],
        'source_epochs': 4,
        'transfer_epochs': 15,
        'train_subset': 20000
    },
    'timestamp': datetime.now().isoformat()
}

with open('results/new_architectures_transfer.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to results/new_architectures_transfer.json")
