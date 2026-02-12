#!/usr/bin/env python3
"""
Transfer Learning Comparison
============================
Compare Standard FF, Bio-inspired FF variants, and BP on transfer learning.

MNIST -> Fashion-MNIST transfer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class FFLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, threshold: float = 2.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = x / (x.norm(2, dim=1, keepdim=True) + 1e-8)
        return self.relu(self.linear(x_normalized))

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        return (x ** 2).mean(dim=1)

    def ff_loss(self, pos_g: torch.Tensor, neg_g: torch.Tensor) -> torch.Tensor:
        return (torch.log(1 + torch.exp(self.threshold - pos_g)).mean() +
                torch.log(1 + torch.exp(neg_g - self.threshold)).mean())


class FFNetwork(nn.Module):
    def __init__(self, layer_sizes: List[int], threshold: float = 2.0):
        super().__init__()
        self.layers = nn.ModuleList([
            FFLayer(layer_sizes[i], layer_sizes[i+1], threshold)
            for i in range(len(layer_sizes) - 1)
        ])
        self.layer_sizes = layer_sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)


class BPNetwork(nn.Module):
    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = x
            for layer in list(self.network)[:-1]:
                h = layer(h)
            return h


def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    x = x.clone()
    label_value = x.max()
    x[:, :num_classes] = 0
    x[torch.arange(len(y)), y] = label_value
    return x


def create_negative(x: torch.Tensor, y: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    wrong_labels = (torch.randint(0, num_classes - 1, (len(y),), device=y.device) + y + 1) % num_classes
    return overlay_y_on_x(x, wrong_labels, num_classes)


def train_ff(model: FFNetwork, x: torch.Tensor, y: torch.Tensor, epochs: int = 500) -> None:
    """Train FF model layer-by-layer."""
    h_pos = overlay_y_on_x(x, y)
    h_neg = create_negative(x, y)

    for layer_idx, layer in enumerate(model.layers):
        optimizer = optim.Adam(layer.parameters(), lr=0.03)

        for epoch in tqdm(range(epochs), desc=f"FF Layer {layer_idx}"):
            optimizer.zero_grad()
            pos_out = layer(h_pos)
            neg_out = layer(h_neg)
            loss = layer.ff_loss(layer.goodness(pos_out), layer.goodness(neg_out))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            h_pos = layer(h_pos)
            h_neg = layer(h_neg)


def train_bp(model: BPNetwork, x: torch.Tensor, y: torch.Tensor, epochs: int = 50) -> None:
    """Train BP model end-to-end."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="BP Training"):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


def test_ff_accuracy(model: FFNetwork, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        all_goodness = []
        for label in range(10):
            test_labels = torch.full((len(x),), label, device=x.device)
            x_labeled = overlay_y_on_x(x, test_labels)
            h = x_labeled
            total_g = torch.zeros(len(x), device=x.device)
            for layer in model.layers:
                h = layer(h)
                total_g += layer.goodness(h)
            all_goodness.append(total_g)
        all_goodness = torch.stack(all_goodness, dim=1)
        return (all_goodness.argmax(1) == y).float().mean().item()


def test_bp_accuracy(model: BPNetwork, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(x)
        return (logits.argmax(1) == y).float().mean().item()


def transfer_test(features_train: torch.Tensor, y_train: torch.Tensor,
                  features_test: torch.Tensor, y_test: torch.Tensor,
                  epochs: int = 100) -> float:
    """Train linear classifier on features and test."""
    device = features_train.device
    classifier = nn.Linear(features_train.shape[1], 10).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(classifier(features_train), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return (classifier(features_test).argmax(1) == y_test).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='FF epochs per layer')
    parser.add_argument('--output', type=str, default='results/transfer_comparison.json')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    fmnist_train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    # Full batch
    x_mnist_train, y_mnist_train = next(iter(DataLoader(mnist_train, batch_size=len(mnist_train))))
    x_mnist_test, y_mnist_test = next(iter(DataLoader(mnist_test, batch_size=len(mnist_test))))
    x_fmnist_train, y_fmnist_train = next(iter(DataLoader(fmnist_train, batch_size=len(fmnist_train))))
    x_fmnist_test, y_fmnist_test = next(iter(DataLoader(fmnist_test, batch_size=len(fmnist_test))))

    x_mnist_train, y_mnist_train = x_mnist_train.to(device), y_mnist_train.to(device)
    x_mnist_test, y_mnist_test = x_mnist_test.to(device), y_mnist_test.to(device)
    x_fmnist_train, y_fmnist_train = x_fmnist_train.to(device), y_fmnist_train.to(device)
    x_fmnist_test, y_fmnist_test = x_fmnist_test.to(device), y_fmnist_test.to(device)

    results = {}

    # 1. Standard FF
    print("\n" + "="*60)
    print("Training Standard FF")
    print("="*60)
    ff_model = FFNetwork([784, 500, 500]).to(device)
    train_ff(ff_model, x_mnist_train, y_mnist_train, epochs=args.epochs)
    ff_source_acc = test_ff_accuracy(ff_model, x_mnist_test, y_mnist_test)
    print(f"FF Source (MNIST): {ff_source_acc*100:.2f}%")

    ff_features_train = ff_model.get_features(x_fmnist_train)
    ff_features_test = ff_model.get_features(x_fmnist_test)
    ff_transfer_acc = transfer_test(ff_features_train, y_fmnist_train, ff_features_test, y_fmnist_test)
    print(f"FF Transfer (FMNIST): {ff_transfer_acc*100:.2f}%")

    results['standard_ff'] = {
        'source_acc': ff_source_acc,
        'transfer_acc': ff_transfer_acc,
    }

    # 2. BP Baseline
    print("\n" + "="*60)
    print("Training BP Baseline")
    print("="*60)
    bp_model = BPNetwork([784, 500, 500, 10]).to(device)
    train_bp(bp_model, x_mnist_train, y_mnist_train, epochs=50)
    bp_source_acc = test_bp_accuracy(bp_model, x_mnist_test, y_mnist_test)
    print(f"BP Source (MNIST): {bp_source_acc*100:.2f}%")

    bp_features_train = bp_model.get_features(x_fmnist_train)
    bp_features_test = bp_model.get_features(x_fmnist_test)
    bp_transfer_acc = transfer_test(bp_features_train, y_fmnist_train, bp_features_test, y_fmnist_test)
    print(f"BP Transfer (FMNIST): {bp_transfer_acc*100:.2f}%")

    results['backprop'] = {
        'source_acc': bp_source_acc,
        'transfer_acc': bp_transfer_acc,
    }

    # 3. Random Init Baseline
    print("\n" + "="*60)
    print("Random Init Baseline")
    print("="*60)
    random_model = FFNetwork([784, 500, 500]).to(device)
    random_features_train = random_model.get_features(x_fmnist_train)
    random_features_test = random_model.get_features(x_fmnist_test)
    random_transfer_acc = transfer_test(random_features_train, y_fmnist_train, random_features_test, y_fmnist_test)
    print(f"Random Transfer (FMNIST): {random_transfer_acc*100:.2f}%")

    results['random_init'] = {
        'source_acc': 0.0,
        'transfer_acc': random_transfer_acc,
    }

    # Summary
    print("\n" + "="*60)
    print("TRANSFER LEARNING SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Source':<12} {'Transfer':<12} {'Gain':<12}")
    print("-"*51)
    for name, r in results.items():
        gain = r['transfer_acc'] - results['random_init']['transfer_acc']
        print(f"{name:<15} {r['source_acc']*100:>10.2f}% {r['transfer_acc']*100:>10.2f}% {gain*100:>+10.2f}%")

    results['metadata'] = {
        'epochs_per_layer': args.epochs,
        'device': str(device),
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
