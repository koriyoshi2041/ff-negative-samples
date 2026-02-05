#!/usr/bin/env python3
"""
Quick Linear Probe Analysis (10 epochs)
"""

import sys
import os
sys.path.insert(0, os.path.expanduser('~/Desktop/Rios/ff-experiment/corrected'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ff_core import FFNetwork, embed_label, normalize


class BPNetwork(nn.Module):
    def __init__(self, layer_sizes: List[int], lr: float = 0.001):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_layer_activations(self, images):
        x = images.view(images.size(0), -1)
        activations = {'input': x.clone()}
        layer_idx = 0
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations[f'layer_{layer_idx}'] = x.clone()
                layer_idx += 1
        return activations
    
    def train_step(self, images, labels):
        self.optimizer.zero_grad()
        loss = self.criterion(self.forward(images), labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class FFWrapper:
    def __init__(self, model):
        self.model = model
    
    def get_layer_activations(self, images, labels):
        batch_size = images.size(0)
        images_flat = images.view(batch_size, -1)
        x = embed_label(images_flat, labels, self.model.num_classes)
        x = normalize(x)
        activations = {'input': images_flat.clone()}
        for i, layer in enumerate(self.model.layers):
            x = layer.get_output(x)
            activations[f'layer_{i}'] = x.clone()
            x = normalize(x)
        return activations


def collect_features(model, dataloader, device, model_type='ff'):
    all_acts = {}
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Collecting {model_type} features"):
            images, labels = images.to(device), labels.to(device)
            if model_type == 'ff':
                acts = model.get_layer_activations(images, labels)
            else:
                acts = model.get_layer_activations(images)
            for name, act in acts.items():
                if name not in all_acts:
                    all_acts[name] = []
                all_acts[name].append(act.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return {k: np.concatenate(v, 0) for k, v in all_acts.items()}, np.concatenate(all_labels)


def linear_probe(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=500, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
    clf.fit(X_tr, y_train)
    return accuracy_score(y_test, clf.predict(X_te))


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    results_dir = os.path.expanduser('~/Desktop/Rios/ff-research/results')
    
    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)
    
    hidden = [500, 500]  # 两个隐藏层
    epochs = 15  # 快速训练
    
    # Train FF
    print("\nTraining FF (15 epochs)...")
    ff = FFNetwork(784, hidden, 10, 2.0, 0.03).to(device)
    for e in range(epochs):
        for imgs, lbls in tqdm(train_loader, desc=f"FF {e+1}/{epochs}", leave=False):
            ff.train_batch(imgs.to(device), lbls.to(device))
        ff.eval()
        correct = sum((ff.predict(i.to(device)) == l.to(device)).sum().item() 
                      for i, l in test_loader)
        print(f"  Epoch {e+1}: {correct/10000*100:.1f}%")
    
    # Train BP
    print("\nTraining BP (15 epochs)...")
    bp = BPNetwork([784] + hidden + [10], 0.001).to(device)
    for e in range(epochs):
        bp.train()
        for imgs, lbls in tqdm(train_loader, desc=f"BP {e+1}/{epochs}", leave=False):
            bp.train_step(imgs.to(device), lbls.to(device))
        bp.eval()
        correct = sum((bp.forward(i.to(device)).argmax(1) == l.to(device)).sum().item() 
                      for i, l in test_loader)
        print(f"  Epoch {e+1}: {correct/10000*100:.1f}%")
    
    # Final accuracies
    ff.eval(); bp.eval()
    ff_acc = sum((ff.predict(i.to(device)) == l.to(device)).sum().item() for i, l in test_loader) / 10000
    bp_acc = sum((bp.forward(i.to(device)).argmax(1) == l.to(device)).sum().item() for i, l in test_loader) / 10000
    print(f"\nFinal: FF={ff_acc*100:.2f}%, BP={bp_acc*100:.2f}%")
    
    # Collect features
    print("\nCollecting features...")
    ff_wrap = FFWrapper(ff)
    ff_train, train_y = collect_features(ff_wrap, train_loader, device, 'ff')
    ff_test, test_y = collect_features(ff_wrap, test_loader, device, 'ff')
    bp_train, _ = collect_features(bp, train_loader, device, 'bp')
    bp_test, _ = collect_features(bp, test_loader, device, 'bp')
    
    # Linear probe
    print("\nLinear Probing...")
    ff_results = {}
    bp_results = {}
    
    layers = ['input'] + [f'layer_{i}' for i in range(len(hidden))]
    for l in layers:
        ff_acc_l = linear_probe(ff_train[l], train_y, ff_test[l], test_y)
        ff_results[l] = ff_acc_l
        print(f"  FF {l}: {ff_acc_l*100:.2f}%")
    
    for l in layers:
        if l in bp_train:
            bp_acc_l = linear_probe(bp_train[l], train_y, bp_test[l], test_y)
            bp_results[l] = bp_acc_l
            print(f"  BP {l}: {bp_acc_l*100:.2f}%")
    
    # Save results
    output = {
        'ff_results': ff_results,
        'bp_results': bp_results,
        'ff_final_acc': ff_acc,
        'bp_final_acc': bp_acc,
        'architecture': '784 → 500 → 500',
        'epochs': epochs
    }
    
    with open(os.path.join(results_dir, 'linear_probe_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(layers))
    width = 0.35
    
    ff_accs = [ff_results[l]*100 for l in layers]
    bp_accs = [bp_results[l]*100 for l in layers]
    
    ax.bar(x - width/2, ff_accs, width, label='FF', color='#e74c3c')
    ax.bar(x + width/2, bp_accs, width, label='BP', color='#3498db')
    
    for i, (f, b) in enumerate(zip(ff_accs, bp_accs)):
        ax.annotate(f'{f:.1f}', (i-width/2, f+1), ha='center', fontsize=9)
        ax.annotate(f'{b:.1f}', (i+width/2, b+1), ha='center', fontsize=9)
    
    ax.set_ylabel('Linear Probe Accuracy (%)')
    ax.set_title('Linear Probe: FF vs BP (784→500→500, 15 epochs)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Input'] + [f'Layer {i}' for i in range(len(hidden))])
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'linear_probe_comparison.png'), dpi=150)
    plt.close()
    
    print(f"\nSaved: {results_dir}/linear_probe_results.json")
    print(f"Saved: {results_dir}/linear_probe_comparison.png")
    
    # Print summary
    print("\n" + "="*50)
    print("Linear Probe Summary")
    print("="*50)
    print(f"\n| Layer | FF Acc | BP Acc | Gap |")
    print("|-------|--------|--------|-----|")
    for l in layers:
        gap = bp_results[l]*100 - ff_results[l]*100
        print(f"| {l:6} | {ff_results[l]*100:5.1f}% | {bp_results[l]*100:5.1f}% | {gap:+.1f}% |")
    
    ff_worst = min(ff_results, key=ff_results.get)
    print(f"\nFF worst layer: {ff_worst} ({ff_results[ff_worst]*100:.1f}%)")
    
    return output


if __name__ == '__main__':
    main()
