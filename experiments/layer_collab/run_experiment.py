#!/usr/bin/env python3
"""
Layer Collaboration FF Experiment - Compact Version

Based on: Lorberbom et al. (2024) "Layer Collaboration in the Forward-Forward Algorithm" AAAI

Core idea: Instead of training each layer independently, introduce a global goodness
signal γ from other layers to enable collaboration.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path
from datetime import datetime

print("="*70)
print("Layer Collaboration in Forward-Forward Algorithm")
print("Based on Lorberbom et al. (2024) AAAI")
print("="*70)

# Device
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Device: {device}")

# Data
print("\nLoading MNIST...")
transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,)),
    Lambda(lambda x: torch.flatten(x))
])
train_data = datasets.MNIST('./data/', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data/', train=False, download=True, transform=transform)

# Use full batch for FF
train_loader = DataLoader(train_data, batch_size=50000, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10000, shuffle=False)

x_train, y_train = next(iter(train_loader))
x_test, y_test = next(iter(test_loader))
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)
print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# Label embedding
def overlay_y(x, y):
    x_ = x.clone()
    x_[:, :10] = 0.0
    x_[range(len(x)), y] = x.max()
    return x_

# Create pos/neg samples
x_pos = overlay_y(x_train, y_train)
rnd = torch.randperm(len(x_train))
x_neg = overlay_y(x_train, y_train[rnd])

# FF Layer
class FFLayer(nn.Module):
    def __init__(self, in_f, out_f, th=2.0, lr=0.03):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.relu = nn.ReLU()
        self.th = th
        self.opt = Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        return self.relu(self.linear(x / (x.norm(2, 1, keepdim=True) + 1e-4)))
    
    def goodness(self, h):
        return h.pow(2).mean(1)

# FF Network
class FFNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([FFLayer(dims[i], dims[i+1]) for i in range(len(dims)-1)])
    
    def train_standard(self, xp, xn, epochs=500):
        """Standard FF: greedy layer-by-layer."""
        hp, hn = xp, xn
        for i, L in enumerate(self.layers):
            print(f"  Layer {i}...", end="", flush=True)
            for ep in range(epochs):
                hp_o = L.forward(hp)
                hn_o = L.forward(hn)
                gp, gn = L.goodness(hp_o), L.goodness(hn_o)
                loss = torch.log(1 + torch.exp(torch.cat([-gp + L.th, gn - L.th]))).mean()
                L.opt.zero_grad()
                loss.backward()
                L.opt.step()
                if (ep+1) % 100 == 0:
                    print(f" [{ep+1}:{loss.item():.3f}]", end="", flush=True)
            hp = L.forward(hp).detach()
            hn = L.forward(hn).detach()
            print()
    
    def train_collab(self, xp, xn, epochs=500, gamma=0.1):
        """Layer Collaboration: joint training with global goodness signal."""
        for ep in range(epochs):
            # Get current goodness values (no grad)
            with torch.no_grad():
                gp_all, gn_all = [], []
                hp, hn = xp, xn
                for L in self.layers:
                    hp, hn = L.forward(hp), L.forward(hn)
                    gp_all.append(L.goodness(hp))
                    gn_all.append(L.goodness(hn))
            
            # Update each layer with collaboration
            hp_in, hn_in = xp, xn
            for k, L in enumerate(self.layers):
                # Global goodness from OTHER layers
                glob_gp = sum(g for j, g in enumerate(gp_all) if j != k)
                glob_gn = sum(g for j, g in enumerate(gn_all) if j != k)
                
                hp_o = L.forward(hp_in)
                hn_o = L.forward(hn_in)
                gp = L.goodness(hp_o) + gamma * glob_gp
                gn = L.goodness(hn_o) + gamma * glob_gn
                
                loss = torch.log(1 + torch.exp(torch.cat([-gp + L.th, gn - L.th]))).mean()
                L.opt.zero_grad()
                loss.backward()
                L.opt.step()
                
                hp_in = L.forward(hp_in).detach()
                hn_in = L.forward(hn_in).detach()
            
            if (ep+1) % 100 == 0:
                print(f"  Epoch {ep+1}: loss={loss.item():.4f}")
    
    def predict(self, x, nc=10):
        g_per_label = []
        for label in range(nc):
            h = overlay_y(x, label)
            g = []
            for L in self.layers:
                h = L(h)
                g.append(L.goodness(h))
            g_per_label.append(sum(g).unsqueeze(1))
        return torch.cat(g_per_label, 1).argmax(1)
    
    def accuracy(self, x, y):
        return (self.predict(x) == y).float().mean().item()

# Run experiments
results = []
configs = [
    ("Standard FF (γ=0)", "standard", 0.0),
    ("Layer Collab (γ=0.1)", "collab", 0.1),
    ("Layer Collab (γ=0.2)", "collab", 0.2),
    ("Layer Collab (γ=0.3)", "collab", 0.3),
]

for name, mode, gamma in configs:
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    
    torch.manual_seed(1234)
    model = FFNet([784, 500, 500]).to(device)
    
    t0 = time.time()
    if mode == "standard":
        model.train_standard(x_pos, x_neg, epochs=500)
    else:
        model.train_collab(x_pos, x_neg, epochs=500, gamma=gamma)
    train_time = time.time() - t0
    
    train_acc = model.accuracy(x_train, y_train)
    test_acc = model.accuracy(x_test, y_test)
    
    print(f"\n  Train Acc: {train_acc*100:.2f}%")
    print(f"  Test Acc:  {test_acc*100:.2f}%")
    print(f"  Time: {train_time:.1f}s")
    
    results.append({
        'method': name,
        'gamma': gamma,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_error': 1.0 - train_acc,
        'test_error': 1.0 - test_acc,
        'train_time': train_time
    })

# Summary
print("\n" + "="*70)
print("SUMMARY: Layer Collaboration vs Standard FF")
print("="*70)
print(f"{'Method':<25} {'γ':>5} {'Train':>10} {'Test':>10} {'Δ Test':>10}")
print("-"*60)
baseline_test = results[0]['test_accuracy']
for r in results:
    delta = (r['test_accuracy'] - baseline_test) * 100
    sign = '+' if delta >= 0 else ''
    print(f"{r['method']:<25} {r['gamma']:>5.2f} {r['train_accuracy']*100:>9.2f}% "
          f"{r['test_accuracy']*100:>9.2f}% {sign}{delta:>9.2f}%")

# Save results
output = {
    'experiment': 'Layer Collaboration in Forward-Forward Algorithm',
    'paper': 'Lorberbom et al. (2024) AAAI',
    'dataset': 'MNIST',
    'architecture': [784, 500, 500],
    'epochs': 500,
    'threshold': 2.0,
    'learning_rate': 0.03,
    'batch_size': 50000,
    'seed': 1234,
    'timestamp': datetime.now().isoformat(),
    'results': results,
    'conclusion': {
        'baseline_test_acc': baseline_test,
        'best_collab_test_acc': max(r['test_accuracy'] for r in results if r['gamma'] > 0),
        'best_gamma': max((r for r in results if r['gamma'] > 0), key=lambda x: x['test_accuracy'])['gamma'],
        'improvement': max(r['test_accuracy'] for r in results if r['gamma'] > 0) - baseline_test
    }
}

# Save
Path('results').mkdir(exist_ok=True)
with open('results/layer_collab_verified.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to results/layer_collab_verified.json")
print(f"\nConclusion:")
print(f"  - Baseline (Standard FF): {baseline_test*100:.2f}% test accuracy")
print(f"  - Best Layer Collaboration: {output['conclusion']['best_collab_test_acc']*100:.2f}% (γ={output['conclusion']['best_gamma']})")
print(f"  - Improvement: {'+' if output['conclusion']['improvement'] > 0 else ''}{output['conclusion']['improvement']*100:.2f}%")
