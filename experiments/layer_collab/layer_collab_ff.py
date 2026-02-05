"""
Layer Collaboration in the Forward-Forward Algorithm

Implementation based on:
Lorberbom, G., Gat, I., Adi, Y., Schwing, A., & Hazan, T. (2024). 
"Layer Collaboration in the Forward-Forward Algorithm."
Proceedings of the AAAI Conference on Artificial Intelligence, 38(13), 14141-14148.

Core Idea:
- Original FF: Each layer is trained independently, using only local goodness
- Layer Collaboration: Introduce a global goodness signal γ from other layers
  to enable collaboration between layers while maintaining local training

Key mechanism:
The goodness for layer k incorporates signals from all layers:
  g_k_collab = g_k + γ * Σ_{j≠k} g_j

where γ (gamma) is a collaboration coefficient that controls how much
information flows between layers.

Theoretical basis: Functional entropy theory suggests that enabling
layer collaboration reduces entropy and improves information flow.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict, Optional
import time
import json
from pathlib import Path


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def overlay_y_on_x(x: torch.Tensor, y) -> torch.Tensor:
    """
    Replace the first 10 pixels of data [x] with one-hot-encoded label [y].
    Uses x.max() as the label value (Hinton's method).
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


# ============================================================
# Layer Collaboration FF Layer
# ============================================================

class LayerCollabFFLayer(nn.Module):
    """
    Forward-Forward Layer with Layer Collaboration support.
    
    Key difference from standard FF:
    - Receives global goodness signal from other layers during training
    - Uses this signal to compute collaborative loss
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = Adam(self.parameters(), lr=lr)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with L2 normalization."""
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))
    
    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Compute goodness - mean of squared activations."""
        return h.pow(2).mean(dim=1)
    
    def train_step_with_collab(self, x: torch.Tensor, is_positive: bool,
                                global_goodness: torch.Tensor,
                                gamma: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single training step with layer collaboration.
        
        Args:
            x: Input to this layer
            is_positive: Whether this is positive data
            global_goodness: Sum of goodness from other layers [batch_size]
            gamma: Collaboration coefficient
            
        Returns:
            h: Layer output (detached)
            g: Local goodness
        """
        h = self.forward(x)
        g_local = self.goodness(h)
        
        # Collaborative goodness: local + gamma * global
        g_collab = g_local + gamma * global_goodness
        
        # Loss based on collaborative goodness
        if is_positive:
            loss = torch.log(1 + torch.exp(-g_collab + self.threshold)).mean()
        else:
            loss = torch.log(1 + torch.exp(g_collab - self.threshold)).mean()
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return h.detach(), g_local.detach()


# ============================================================
# Layer Collaboration FF Network
# ============================================================

class LayerCollabFFNetwork(nn.Module):
    """
    Forward-Forward Network with Layer Collaboration.
    
    Training modes:
    1. Standard FF (baseline): gamma = 0, layers train independently
    2. Layer Collaboration: gamma > 0, layers receive global signal
    
    Training procedure:
    - Unlike standard greedy training, we do multi-pass training
    - In each iteration:
      1. Forward all layers to compute all goodness values
      2. For each layer, compute global goodness (sum of other layers)
      3. Update each layer using collaborative loss
    """
    
    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.dims = dims
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(LayerCollabFFLayer(dims[d], dims[d + 1], threshold, lr))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def forward_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward and return all layer outputs."""
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs
    
    def compute_all_goodness(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute activations and goodness for all layers.
        
        Returns:
            activations: List of layer outputs
            goodness_values: List of goodness values per layer
        """
        activations = []
        goodness_values = []
        
        h = x
        for layer in self.layers:
            h = layer.forward(h)
            g = layer.goodness(h)
            activations.append(h)
            goodness_values.append(g)
        
        return activations, goodness_values
    
    def train_with_collaboration(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                                  num_epochs: int = 1000, 
                                  gamma: float = 0.1,
                                  warmup_epochs: int = 100,
                                  verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train with layer collaboration.
        
        Training procedure (per epoch):
        1. Forward all layers on pos/neg data (no grad)
        2. Compute goodness for each layer
        3. For each layer k:
           - global_goodness = sum of goodness from other layers
           - Update layer k using: g_k + gamma * global_goodness
        
        Args:
            x_pos: Positive samples [batch_size, features]
            x_neg: Negative samples [batch_size, features]
            num_epochs: Total training epochs
            gamma: Collaboration coefficient (0 = standard FF, >0 = collaboration)
            warmup_epochs: Epochs to train with gamma=0 first (stabilization)
            verbose: Print progress
            
        Returns:
            history: Training history
        """
        history = {
            'epoch': [],
            'loss': [],
            'g_pos_mean': [],
            'g_neg_mean': [],
            'layer_goodness': []
        }
        
        for epoch in range(num_epochs):
            # Use gamma=0 during warmup (standard FF)
            current_gamma = 0.0 if epoch < warmup_epochs else gamma
            
            # ============ Step 1: Forward pass (no grad) to get all goodness ============
            with torch.no_grad():
                # Process through network to get inputs for each layer
                h_pos_list = [x_pos]
                h_neg_list = [x_neg]
                
                h_pos, h_neg = x_pos, x_neg
                for layer in self.layers[:-1]:  # All but last
                    h_pos = layer.forward(h_pos)
                    h_neg = layer.forward(h_neg)
                    h_pos_list.append(h_pos)
                    h_neg_list.append(h_neg)
                
                # Compute current goodness for all layers
                g_pos_all = []
                g_neg_all = []
                h_pos, h_neg = x_pos, x_neg
                for layer in self.layers:
                    h_pos = layer.forward(h_pos)
                    h_neg = layer.forward(h_neg)
                    g_pos_all.append(layer.goodness(h_pos))
                    g_neg_all.append(layer.goodness(h_neg))
            
            # ============ Step 2: Update each layer with collaboration ============
            total_loss = 0.0
            layer_g_pos = []
            layer_g_neg = []
            
            for k, layer in enumerate(self.layers):
                # Get input for this layer
                if k == 0:
                    h_pos_in = x_pos
                    h_neg_in = x_neg
                else:
                    # Need fresh forward from previous layers (with current weights)
                    h_pos_in = x_pos
                    h_neg_in = x_neg
                    for prev_layer in self.layers[:k]:
                        h_pos_in = prev_layer.forward(h_pos_in)
                        h_neg_in = prev_layer.forward(h_neg_in)
                    h_pos_in = h_pos_in.detach()
                    h_neg_in = h_neg_in.detach()
                
                # Compute global goodness (sum of OTHER layers)
                with torch.no_grad():
                    global_g_pos = torch.zeros_like(g_pos_all[0])
                    global_g_neg = torch.zeros_like(g_neg_all[0])
                    for j, (gp, gn) in enumerate(zip(g_pos_all, g_neg_all)):
                        if j != k:  # Exclude current layer
                            global_g_pos = global_g_pos + gp
                            global_g_neg = global_g_neg + gn
                
                # Forward this layer
                h_pos = layer.forward(h_pos_in)
                h_neg = layer.forward(h_neg_in)
                
                g_pos = layer.goodness(h_pos)
                g_neg = layer.goodness(h_neg)
                
                layer_g_pos.append(g_pos.mean().item())
                layer_g_neg.append(g_neg.mean().item())
                
                # Collaborative goodness
                g_pos_collab = g_pos + current_gamma * global_g_pos
                g_neg_collab = g_neg + current_gamma * global_g_neg
                
                # Loss
                loss_pos = torch.log(1 + torch.exp(-g_pos_collab + layer.threshold)).mean()
                loss_neg = torch.log(1 + torch.exp(g_neg_collab - layer.threshold)).mean()
                loss = loss_pos + loss_neg
                
                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()
                
                total_loss += loss.item()
            
            # Record history
            if (epoch + 1) % 100 == 0 or epoch == 0:
                history['epoch'].append(epoch + 1)
                history['loss'].append(total_loss / len(self.layers))
                history['g_pos_mean'].append(sum(layer_g_pos) / len(layer_g_pos))
                history['g_neg_mean'].append(sum(layer_g_neg) / len(layer_g_neg))
                history['layer_goodness'].append({
                    'pos': layer_g_pos.copy(),
                    'neg': layer_g_neg.copy()
                })
                
                if verbose:
                    phase = "warmup" if epoch < warmup_epochs else "collab"
                    print(f"  Epoch {epoch+1:4d} [{phase}]: loss={total_loss/len(self.layers):.4f}, "
                          f"g_pos={sum(layer_g_pos)/len(layer_g_pos):.3f}, "
                          f"g_neg={sum(layer_g_neg)/len(layer_g_neg):.3f}")
        
        return history
    
    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels and picking highest total goodness."""
        goodness_per_label = []
        
        for label in range(num_classes):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))
            total_goodness = sum(goodness)
            goodness_per_label.append(total_goodness.unsqueeze(1))
        
        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)
    
    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor, 
                     num_classes: int = 10) -> float:
        """Compute accuracy."""
        predictions = self.predict(x, num_classes)
        return (predictions == y).float().mean().item()


# ============================================================
# Alternative: Per-Layer Sequential Collaboration
# ============================================================

class LayerCollabFFNetwork_V2(LayerCollabFFNetwork):
    """
    Alternative implementation: Sequential training with collaboration signal.
    
    This version trains layer-by-layer (like original FF) but passes
    a collaboration signal from previously trained layers.
    """
    
    def train_sequential_collab(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                                 epochs_per_layer: int = 500,
                                 gamma: float = 0.1,
                                 verbose: bool = True) -> Dict[str, List[float]]:
        """
        Sequential training with collaboration from previous layers.
        
        For layer k:
        - Train using local goodness + gamma * sum(goodness of layers 0..k-1)
        """
        history = {'layer_results': []}
        
        h_pos, h_neg = x_pos, x_neg
        accumulated_g_pos = torch.zeros(x_pos.shape[0], device=x_pos.device)
        accumulated_g_neg = torch.zeros(x_neg.shape[0], device=x_neg.device)
        
        for layer_idx, layer in enumerate(self.layers):
            if verbose:
                print(f"\n  Training layer {layer_idx}...")
            
            layer_history = []
            
            for epoch in range(epochs_per_layer):
                h_pos_out = layer.forward(h_pos)
                h_neg_out = layer.forward(h_neg)
                
                g_pos = layer.goodness(h_pos_out)
                g_neg = layer.goodness(h_neg_out)
                
                # Collaborative goodness (from previous layers)
                if layer_idx > 0:
                    g_pos_collab = g_pos + gamma * accumulated_g_pos
                    g_neg_collab = g_neg + gamma * accumulated_g_neg
                else:
                    g_pos_collab = g_pos
                    g_neg_collab = g_neg
                
                # Loss
                loss = torch.log(1 + torch.exp(torch.cat([
                    -g_pos_collab + layer.threshold,
                    g_neg_collab - layer.threshold
                ]))).mean()
                
                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()
                
                if (epoch + 1) % 100 == 0:
                    layer_history.append({
                        'epoch': epoch + 1,
                        'loss': loss.item(),
                        'g_pos': g_pos.mean().item(),
                        'g_neg': g_neg.mean().item()
                    })
                    if verbose:
                        print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, "
                              f"g_pos={g_pos.mean().item():.3f}, "
                              f"g_neg={g_neg.mean().item():.3f}")
            
            # Update accumulated goodness and move to next layer
            with torch.no_grad():
                h_pos_final = layer.forward(h_pos)
                h_neg_final = layer.forward(h_neg)
                accumulated_g_pos = accumulated_g_pos + layer.goodness(h_pos_final)
                accumulated_g_neg = accumulated_g_neg + layer.goodness(h_neg_final)
                h_pos = h_pos_final.detach()
                h_neg = h_neg_final.detach()
            
            history['layer_results'].append({
                'layer': layer_idx,
                'history': layer_history
            })
        
        return history


# ============================================================
# Data Loading
# ============================================================

def get_mnist_loaders(train_batch_size: int = 50000, test_batch_size: int = 10000):
    """Get MNIST data loaders."""
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])
    
    train_loader = DataLoader(
        datasets.MNIST('./data/', train=True, download=True, transform=transform),
        batch_size=train_batch_size, shuffle=True
    )
    
    test_loader = DataLoader(
        datasets.MNIST('./data/', train=False, download=True, transform=transform),
        batch_size=test_batch_size, shuffle=False
    )
    
    return train_loader, test_loader


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment(gamma: float = 0.1, 
                   num_epochs: int = 1000,
                   warmup_epochs: int = 100,
                   seed: int = 1234,
                   version: str = "v1",
                   verbose: bool = True) -> Dict:
    """
    Run Layer Collaboration experiment.
    
    Args:
        gamma: Collaboration coefficient (0 = standard FF)
        num_epochs: Total training epochs
        warmup_epochs: Epochs with gamma=0 for warmup
        seed: Random seed
        version: "v1" (simultaneous) or "v2" (sequential)
        verbose: Print progress
    """
    torch.manual_seed(seed)
    device = get_device()
    
    # Data
    train_loader, test_loader = get_mnist_loaders()
    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    # Create pos/neg samples
    x_pos = overlay_y_on_x(x_train, y_train)
    rnd = torch.randperm(x_train.size(0))
    x_neg = overlay_y_on_x(x_train, y_train[rnd])
    
    # Model
    if version == "v1":
        model = LayerCollabFFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    else:
        model = LayerCollabFFNetwork_V2([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    
    # Train
    start_time = time.time()
    if version == "v1":
        history = model.train_with_collaboration(
            x_pos, x_neg, 
            num_epochs=num_epochs,
            gamma=gamma,
            warmup_epochs=warmup_epochs,
            verbose=verbose
        )
    else:
        history = model.train_sequential_collab(
            x_pos, x_neg,
            epochs_per_layer=num_epochs // len(model.layers),
            gamma=gamma,
            verbose=verbose
        )
    train_time = time.time() - start_time
    
    # Evaluate
    train_acc = model.get_accuracy(x_train, y_train)
    test_acc = model.get_accuracy(x_test, y_test)
    
    results = {
        'gamma': gamma,
        'version': version,
        'num_epochs': num_epochs,
        'warmup_epochs': warmup_epochs,
        'seed': seed,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_error': 1.0 - train_acc,
        'test_error': 1.0 - test_acc,
        'train_time': train_time,
        'device': str(device),
        'architecture': [784, 500, 500]
    }
    
    return results, model, history


def compare_gammas(gammas: List[float] = [0.0, 0.05, 0.1, 0.2, 0.5],
                   num_epochs: int = 1000,
                   warmup_epochs: int = 100,
                   seeds: List[int] = [1234, 2345, 3456],
                   verbose: bool = True) -> List[Dict]:
    """
    Compare different gamma values.
    """
    all_results = []
    
    for gamma in gammas:
        print(f"\n{'='*60}")
        print(f"Testing gamma = {gamma}")
        print(f"{'='*60}")
        
        gamma_results = []
        for seed in seeds:
            print(f"\n  Seed {seed}:")
            results, _, _ = run_experiment(
                gamma=gamma,
                num_epochs=num_epochs,
                warmup_epochs=warmup_epochs,
                seed=seed,
                verbose=verbose
            )
            gamma_results.append(results)
            print(f"  → Train: {results['train_accuracy']*100:.2f}%, "
                  f"Test: {results['test_accuracy']*100:.2f}%")
        
        # Average results
        avg_train = sum(r['train_accuracy'] for r in gamma_results) / len(gamma_results)
        avg_test = sum(r['test_accuracy'] for r in gamma_results) / len(gamma_results)
        
        all_results.append({
            'gamma': gamma,
            'runs': gamma_results,
            'avg_train_accuracy': avg_train,
            'avg_test_accuracy': avg_test,
            'std_test_accuracy': torch.tensor([r['test_accuracy'] for r in gamma_results]).std().item()
        })
    
    return all_results


def main():
    """Main experiment."""
    print("="*70)
    print("Layer Collaboration in Forward-Forward Algorithm")
    print("Based on Lorberbom et al. (2024) AAAI")
    print("="*70)
    
    # Quick comparison
    gammas = [0.0, 0.1, 0.2]  # 0.0 = baseline FF
    results = compare_gammas(
        gammas=gammas,
        num_epochs=600,
        warmup_epochs=50,
        seeds=[1234],
        verbose=True
    )
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Gamma':<10} {'Train Acc':<12} {'Test Acc':<12}")
    print("-"*40)
    for r in results:
        print(f"{r['gamma']:<10} {r['avg_train_accuracy']*100:.2f}%{'':<6} "
              f"{r['avg_test_accuracy']*100:.2f}%")
    
    return results


if __name__ == "__main__":
    results = main()
