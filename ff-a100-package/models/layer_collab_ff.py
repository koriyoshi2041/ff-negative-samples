"""
Layer Collaboration Forward-Forward Implementation

Based on: Lorberbom et al. (2024) "Layer Collaboration in the Forward-Forward Algorithm"
Paper: https://arxiv.org/abs/2305.12393

FIXED VERSION - Incorporates correct base FF implementation:
1. Goodness = MEAN of squared activations (not sum!)
2. Layer-by-layer training with collaboration
3. Label embedding uses x.max() (not 1.0)
4. Full-batch training

Key modifications from original FF:
1. Add γ (global goodness from other layers) to probability calculation
2. γ is detached from gradient computation
3. Can train by alternating across layers or layer-by-layer
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict
import time


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
    
    CRITICAL: Use x.max() as the label value, not 1.0!
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


# ============================================================
# Layer Collaboration FF Layer
# ============================================================

class CollabFFLayer(nn.Module):
    """
    Forward-Forward layer with layer collaboration support.
    
    FIXED: Uses MEAN for goodness calculation.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = Adam(self.parameters(), lr=lr)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with layer normalization."""
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))
    
    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute goodness - MEAN of squared activations.
        CRITICAL: Must be MEAN, not SUM!
        """
        return x.pow(2).mean(dim=1)
    
    def ff_loss(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor,
                gamma_pos: torch.Tensor = None, gamma_neg: torch.Tensor = None) -> torch.Tensor:
        """
        FF loss with optional layer collaboration (gamma).
        
        Original FF:  p = sigmoid(goodness - θ)
        Collab FF:    p = sigmoid(goodness + γ - θ)
        
        where γ = sum of goodness from other layers (detached)
        """
        if gamma_pos is None:
            gamma_pos = torch.zeros_like(pos_goodness)
        if gamma_neg is None:
            gamma_neg = torch.zeros_like(neg_goodness)
            
        # Collaborative logits
        pos_logit = pos_goodness + gamma_pos - self.threshold
        neg_logit = neg_goodness + gamma_neg - self.threshold
        
        # Loss: push positive above threshold, negative below
        loss = torch.log(1 + torch.exp(torch.cat([
            -pos_logit,
            neg_logit
        ]))).mean()
        
        return loss


# ============================================================
# Layer Collaboration FF Network
# ============================================================

class CollabFFNetwork(nn.Module):
    """
    Multi-layer Forward-Forward Network with Layer Collaboration.
    
    FIXED: Uses correct goodness (MEAN) and proper training.
    """
    
    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        self.threshold = threshold
        
        for d in range(len(dims) - 1):
            self.layers.append(CollabFFLayer(dims[d], dims[d + 1], threshold, lr))
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return activations from all layers."""
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations
    
    def compute_all_goodness(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute goodness for all layers (detached for γ calculation)."""
        goodness_list = []
        h = x
        for layer in self.layers:
            h = layer(h)
            g = layer.goodness(h)
            goodness_list.append(g.detach())
            h = h.detach()
        return goodness_list
    
    def compute_gamma(self, goodness_list: List[torch.Tensor], current_layer: int, 
                      mode: str = 'all') -> torch.Tensor:
        """
        Compute γ for a specific layer.
        
        Args:
            goodness_list: List of goodness values from all layers (detached)
            current_layer: Index of current layer
            mode: 'all' (all other layers) or 'previous' (only previous layers)
        """
        gamma = torch.zeros_like(goodness_list[0])
        
        for i, g in enumerate(goodness_list):
            if mode == 'all' and i != current_layer:
                gamma = gamma + g
            elif mode == 'previous' and i < current_layer:
                gamma = gamma + g
        
        return gamma
    
    def train_layer_with_collab(self, layer_idx: int, 
                                 x_pos: torch.Tensor, x_neg: torch.Tensor,
                                 num_epochs: int = 1000,
                                 gamma_mode: str = 'all',
                                 verbose: bool = True):
        """Train a single layer with layer collaboration."""
        layer = self.layers[layer_idx]
        
        for epoch in range(num_epochs):
            # Compute all goodness values (detached)
            pos_goodness_all = self.compute_all_goodness(x_pos)
            neg_goodness_all = self.compute_all_goodness(x_neg)
            
            # Compute gamma for this layer
            gamma_pos = self.compute_gamma(pos_goodness_all, layer_idx, gamma_mode)
            gamma_neg = self.compute_gamma(neg_goodness_all, layer_idx, gamma_mode)
            
            # Forward to this layer's input
            h_pos = x_pos
            h_neg = x_neg
            for i in range(layer_idx):
                h_pos = self.layers[i](h_pos).detach()
                h_neg = self.layers[i](h_neg).detach()
            
            # Forward through current layer (with gradient)
            out_pos = layer(h_pos)
            out_neg = layer(h_neg)
            
            # Compute goodness and loss
            g_pos = layer.goodness(out_pos)
            g_neg = layer.goodness(out_neg)
            loss = layer.ff_loss(g_pos, g_neg, gamma_pos, gamma_neg)
            
            # Backward
            layer.opt.zero_grad()
            loss.backward()
            layer.opt.step()
            
            if verbose and (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, "
                      f"g_pos={g_pos.mean().item():.3f}, "
                      f"g_neg={g_neg.mean().item():.3f}")
    
    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 1000, verbose: bool = True):
        """Original FF greedy training (no collaboration)."""
        h_pos, h_neg = x_pos, x_neg
        
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\n  Training layer {i} (Original FF)...')
            
            for epoch in range(epochs_per_layer):
                out_pos = layer(h_pos)
                out_neg = layer(h_neg)
                
                g_pos = layer.goodness(out_pos)
                g_neg = layer.goodness(out_neg)
                
                loss = layer.ff_loss(g_pos, g_neg)
                
                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()
                
                if verbose and (epoch + 1) % 200 == 0:
                    print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, "
                          f"g_pos={g_pos.mean().item():.3f}, "
                          f"g_neg={g_neg.mean().item():.3f}")
            
            h_pos = layer(h_pos).detach()
            h_neg = layer(h_neg).detach()
    
    def train_collaborative(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                            epochs_per_layer: int = 1000,
                            gamma_mode: str = 'all',
                            verbose: bool = True):
        """Layer Collaboration FF training."""
        for i in range(len(self.layers)):
            if verbose:
                print(f'\n  Training layer {i} (Collaborative, γ={gamma_mode})...')
            self.train_layer_with_collab(i, x_pos, x_neg, epochs_per_layer, gamma_mode, verbose)
    
    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels and picking highest goodness."""
        batch_size = x.shape[0]
        goodness_per_label = []
        
        for label in range(num_classes):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))
            goodness_per_label.append(sum(goodness).unsqueeze(1))
        
        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)
    
    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute accuracy."""
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()


# ============================================================
# Training Functions  
# ============================================================

def get_mnist_loaders(train_batch_size: int = 50000, test_batch_size: int = 10000):
    """Get MNIST data loaders with large batch sizes."""
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


def run_comparison(num_epochs: int = 500, seed: int = 1234) -> Dict:
    """
    Compare Original FF vs Layer Collaboration FF on MNIST.
    
    Expected results (from paper):
    - Original FF: ~3.3% error
    - Layer Collab FF: ~2.1% error
    """
    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")
    
    # Data
    train_loader, test_loader = get_mnist_loaders()
    
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)
    
    # Create pos/neg samples
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    
    results = {}
    
    # ============================================================
    # Train Original FF
    # ============================================================
    print("\n" + "="*60)
    print("Training: Original Forward-Forward")
    print("="*60)
    
    torch.manual_seed(seed)
    model_orig = CollabFFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    
    start = time.time()
    model_orig.train_greedy(x_pos, x_neg, num_epochs)
    elapsed = time.time() - start
    
    train_acc = model_orig.get_accuracy(x, y)
    test_acc = model_orig.get_accuracy(x_te, y_te)
    
    results['original'] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'time': elapsed
    }
    
    print(f"\n  Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}%, Time: {elapsed:.1f}s")
    
    # ============================================================
    # Train Layer Collab FF (all)
    # ============================================================
    print("\n" + "="*60)
    print("Training: Layer Collaboration FF (γ = all)")
    print("="*60)
    
    torch.manual_seed(seed)
    model_collab = CollabFFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    
    start = time.time()
    model_collab.train_collaborative(x_pos, x_neg, num_epochs, 'all')
    elapsed = time.time() - start
    
    train_acc = model_collab.get_accuracy(x, y)
    test_acc = model_collab.get_accuracy(x_te, y_te)
    
    results['collab_all'] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'time': elapsed
    }
    
    print(f"\n  Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}%, Time: {elapsed:.1f}s")
    
    # ============================================================
    # Final Results
    # ============================================================
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print(f"\n{'Method':<35} {'Train Acc':>10} {'Test Acc':>10} {'Test Err':>10}")
    print("-" * 65)
    
    for name, res in results.items():
        print(f"{name:<35} {res['train_acc']*100:>9.2f}% {res['test_acc']*100:>9.2f}% "
              f"{(1-res['test_acc'])*100:>9.2f}%")
    
    print("\nPaper reference (MNIST):")
    print("  Original FF:     ~3.3% error (~96.7% accuracy)")
    print("  Layer Collab:    ~2.1% error (~97.9% accuracy)")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='Epochs per layer')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()
    
    results = run_comparison(num_epochs=args.epochs, seed=args.seed)
