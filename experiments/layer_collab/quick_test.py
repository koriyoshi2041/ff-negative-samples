"""Quick test of Layer Collaboration FF."""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path

print("Starting quick test...")

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

class FFLayer(nn.Module):
    def __init__(self, in_f, out_f, threshold=2.0, lr=0.03):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = Adam(self.parameters(), lr=lr)
        
    def forward(self, x):
        x_dir = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_dir))
    
    def goodness(self, h):
        return h.pow(2).mean(dim=1)

class FFNet(nn.Module):
    def __init__(self, dims, threshold=2.0, lr=0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(FFLayer(dims[d], dims[d+1], threshold, lr))
    
    def train_standard(self, x_pos, x_neg, epochs=500, verbose=True):
        """Standard FF: greedy layer-by-layer."""
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f"  Layer {i}...")
            for ep in range(epochs):
                hp = layer.forward(h_pos)
                hn = layer.forward(h_neg)
                gp = layer.goodness(hp)
                gn = layer.goodness(hn)
                loss = torch.log(1 + torch.exp(torch.cat([
                    -gp + layer.threshold,
                    gn - layer.threshold
                ]))).mean()
                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()
                if verbose and (ep+1) % 100 == 0:
                    print(f"    Ep {ep+1}: loss={loss.item():.4f}")
            h_pos = layer.forward(h_pos).detach()
            h_neg = layer.forward(h_neg).detach()
    
    def train_collab(self, x_pos, x_neg, epochs=500, gamma=0.1, verbose=True):
        """Layer Collaboration: joint training with global goodness."""
        for ep in range(epochs):
            # Forward all (no grad) to get goodness
            with torch.no_grad():
                g_pos_all = []
                g_neg_all = []
                hp, hn = x_pos, x_neg
                for layer in self.layers:
                    hp = layer.forward(hp)
                    hn = layer.forward(hn)
                    g_pos_all.append(layer.goodness(hp))
                    g_neg_all.append(layer.goodness(hn))
            
            # Update each layer with collaboration
            total_loss = 0
            hp_in, hn_in = x_pos, x_neg
            for k, layer in enumerate(self.layers):
                # Global goodness from OTHER layers
                global_gp = sum(g for j, g in enumerate(g_pos_all) if j != k)
                global_gn = sum(g for j, g in enumerate(g_neg_all) if j != k)
                
                hp = layer.forward(hp_in)
                hn = layer.forward(hn_in)
                gp = layer.goodness(hp)
                gn = layer.goodness(hn)
                
                # Collaborative goodness
                gp_c = gp + gamma * global_gp
                gn_c = gn + gamma * global_gn
                
                loss = torch.log(1 + torch.exp(torch.cat([
                    -gp_c + layer.threshold,
                    gn_c - layer.threshold
                ]))).mean()
                
                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()
                total_loss += loss.item()
                
                hp_in = layer.forward(hp_in).detach()
                hn_in = layer.forward(hn_in).detach()
            
            if verbose and (ep+1) % 100 == 0:
                print(f"  Ep {ep+1}: loss={total_loss/len(self.layers):.4f}")
    
    def predict(self, x, num_classes=10):
        goodness_per_label = []
        for label in range(num_classes):
            h = overlay_y_on_x(x, label)
            g_total = []
            for layer in self.layers:
                h = layer(h)
                g_total.append(layer.goodness(h))
            goodness_per_label.append(sum(g_total).unsqueeze(1))
        return torch.cat(goodness_per_label, dim=1).argmax(dim=1)
    
    def accuracy(self, x, y):
        return (self.predict(x) == y).float().mean().item()

def load_data(batch_size=50000):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])
    train = DataLoader(
        datasets.MNIST('./data/', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test = DataLoader(
        datasets.MNIST('./data/', train=False, download=True, transform=transform),
        batch_size=10000, shuffle=False
    )
    return train, test

def run_comparison():
    device = get_device()
    print(f"Device: {device}")
    
    train_loader, test_loader = load_data()
    x_tr, y_tr = next(iter(train_loader))
    x_te, y_te = next(iter(test_loader))
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    
    x_pos = overlay_y_on_x(x_tr, y_tr)
    rnd = torch.randperm(x_tr.size(0))
    x_neg = overlay_y_on_x(x_tr, y_tr[rnd])
    
    results = []
    
    # Test different configurations
    configs = [
        ("Standard FF (γ=0)", "standard", 0.0),
        ("Layer Collab (γ=0.1)", "collab", 0.1),
        ("Layer Collab (γ=0.2)", "collab", 0.2),
    ]
    
    for name, mode, gamma in configs:
        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")
        
        torch.manual_seed(1234)
        model = FFNet([784, 500, 500]).to(device)
        
        start = time.time()
        if mode == "standard":
            model.train_standard(x_pos, x_neg, epochs=500, verbose=True)
        else:
            model.train_collab(x_pos, x_neg, epochs=500, gamma=gamma, verbose=True)
        train_time = time.time() - start
        
        train_acc = model.accuracy(x_tr, y_tr)
        test_acc = model.accuracy(x_te, y_te)
        
        print(f"\nResults:")
        print(f"  Train Accuracy: {train_acc*100:.2f}%")
        print(f"  Test Accuracy:  {test_acc*100:.2f}%")
        print(f"  Training Time:  {train_time:.1f}s")
        
        results.append({
            'name': name,
            'mode': mode,
            'gamma': gamma,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_error': 1.0 - train_acc,
            'test_error': 1.0 - test_acc,
            'train_time': train_time
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Train Acc':>12} {'Test Acc':>12}")
    print("-"*50)
    for r in results:
        print(f"{r['name']:<25} {r['train_accuracy']*100:>11.2f}% {r['test_accuracy']*100:>11.2f}%")
    
    return results

if __name__ == "__main__":
    results = run_comparison()
    
    # Save results
    output = {
        'experiment': 'Layer Collaboration FF vs Standard FF',
        'paper': 'Lorberbom et al. (2024) AAAI',
        'dataset': 'MNIST',
        'architecture': [784, 500, 500],
        'results': results
    }
    
    Path('results').mkdir(exist_ok=True)
    with open('results/layer_collab_quick.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results/layer_collab_quick.json")
