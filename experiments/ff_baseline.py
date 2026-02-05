"""
Forward-Forward Algorithm Baseline Implementation
For RQ1 (Transfer Learning) and RQ2 (Negative Sample Comparison)

FIXED VERSION - Based on correct Hinton/mpezeshki implementation

Key fixes (vs original broken implementation):
1. Goodness = MEAN of squared activations (not sum!)
2. Layer-by-layer greedy training (train each layer to convergence)
3. Label embedding uses x.max() (not 1.0)
4. Full-batch training (large batch sizes)

Reference: https://github.com/mpezeshki/pytorch_forward_forward
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict, Optional
import time


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Correct Label Embedding (Hinton's method)
# ============================================================

def overlay_y_on_x(x: torch.Tensor, y) -> torch.Tensor:
    """
    Replace the first 10 pixels of data [x] with one-hot-encoded label [y].
    
    CRITICAL: Use x.max() as the label value, not 1.0!
    This ensures the label signal is comparable in magnitude to the image data.
    
    Args:
        x: Flattened image tensor [batch_size, 784]
        y: Labels - can be int (for prediction) or tensor (for training)
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0  # Zero out first 10 pixels
    x_[range(x.shape[0]), y] = x.max()  # Use x.max() not 1.0!
    return x_


# ============================================================
# FF Layer Implementation (Correct)
# ============================================================

class FFLayer(nn.Module):
    """
    Forward-Forward Layer - Correct Implementation.
    
    Key points:
    - Goodness = MEAN of squared activations (NOT sum!)
    - Threshold default = 2.0
    - Layer normalization before linear transformation
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
        """
        Compute goodness - MEAN of squared activations.
        
        CRITICAL: This must be MEAN, not SUM!
        Using sum makes the threshold scale with layer width.
        """
        return h.pow(2).mean(dim=1)
    
    def train_layer(self, x_pos: torch.Tensor, x_neg: torch.Tensor, 
                    num_epochs: int = 1000, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train this layer to convergence (greedy layer-wise training).
        
        This is THE KEY difference from incorrect implementations:
        - Train each layer for many epochs before moving to next
        - This allows proper greedy layer-wise optimization
        """
        for i in range(num_epochs):
            # Forward pass
            h_pos = self.forward(x_pos)
            h_neg = self.forward(x_neg)
            
            # Compute goodness (MEAN not sum!)
            g_pos = self.goodness(h_pos)
            g_neg = self.goodness(h_neg)
            
            # Loss: push positive above threshold, negative below
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold
            ]))).mean()
            
            # Backward (local to this layer only!)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            if verbose and (i + 1) % 200 == 0:
                print(f"    Epoch {i+1}: loss={loss.item():.4f}, "
                      f"g_pos={g_pos.mean().item():.3f}, "
                      f"g_neg={g_neg.mean().item():.3f}")
        
        # Return detached outputs for next layer
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


# ============================================================
# FF Network Implementation (Correct)
# ============================================================

class FFNetwork(nn.Module):
    """
    Forward-Forward Network - Correct Implementation.
    
    Key difference from incorrect implementations:
    - Uses greedy layer-by-layer training
    - NOT simultaneous mini-batch training of all layers
    """
    
    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(FFLayer(dims[d], dims[d + 1], threshold, lr))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get activations from all layers."""
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations
    
    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 1000, verbose: bool = True):
        """
        Greedy layer-by-layer training (CORRECT method).
        
        1. Train layer 0 for N epochs
        2. Freeze layer 0, train layer 1 for N epochs  
        3. And so on...
        """
        h_pos, h_neg = x_pos, x_neg
        
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\n  Training layer {i}...')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg, epochs_per_layer, verbose)
    
    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """
        Predict by trying all labels and picking highest total goodness.
        
        For each candidate label:
        1. Embed the label in the image
        2. Forward through all layers
        3. Sum goodness across all layers
        4. Pick label with highest total goodness
        """
        batch_size = x.shape[0]
        goodness_per_label = []
        
        for label in range(num_classes):
            # Create input with this candidate label
            h = overlay_y_on_x(x, label)
            
            # Compute goodness at each layer
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))
            
            # Sum goodness across layers
            total_goodness = sum(goodness)
            goodness_per_label.append(total_goodness.unsqueeze(1))
        
        # Return label with highest goodness
        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)
    
    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor, 
                     num_classes: int = 10) -> float:
        """Compute accuracy."""
        predictions = self.predict(x, num_classes)
        return (predictions == y).float().mean().item()


# ============================================================
# Data Loading
# ============================================================

def get_mnist_loaders(train_batch_size: int = 50000, test_batch_size: int = 10000):
    """
    Get MNIST data loaders.
    
    IMPORTANT: Use large batch sizes for FF training!
    The original implementation uses full training set (50000).
    """
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
# Training Functions
# ============================================================

def train_ff(model: FFNetwork, 
             train_loader: DataLoader,
             device: torch.device,
             epochs_per_layer: int = 1000,
             verbose: bool = True) -> Dict[str, float]:
    """
    Train FF network correctly using greedy layer-by-layer approach.
    
    This is the CORRECT way to train FF:
    1. Load entire training set (or large batch)
    2. Create positive samples (correct label embedded)
    3. Create negative samples (shuffled/wrong label embedded)
    4. Train each layer to convergence before moving to next
    """
    # Load all data
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    
    # Create positive samples (correct label)
    x_pos = overlay_y_on_x(x, y)
    
    # Create negative samples (shuffled labels - Hinton's method)
    # IMPORTANT: Shuffle labels across batch, don't generate random wrong labels
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    
    # Train layer-by-layer
    start_time = time.time()
    model.train_greedy(x_pos, x_neg, epochs_per_layer, verbose)
    train_time = time.time() - start_time
    
    # Compute final metrics
    train_acc = model.get_accuracy(x, y)
    
    return {
        'train_acc': train_acc,
        'train_error': 1.0 - train_acc,
        'train_time': train_time
    }


def evaluate_ff(model: FFNetwork, 
                test_loader: DataLoader,
                device: torch.device) -> Dict[str, float]:
    """Evaluate FF network on test set."""
    model.eval()
    
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)
    
    test_acc = model.get_accuracy(x_te, y_te)
    
    return {
        'test_acc': test_acc,
        'test_error': 1.0 - test_acc
    }


# ============================================================
# Main Experiment
# ============================================================

def main(epochs_per_layer: int = 1000, seed: int = 1234):
    """Run the correct FF implementation on MNIST."""
    # Setup
    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")
    
    # Data
    train_loader, test_loader = get_mnist_loaders()
    
    # Model
    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)
    
    print("\n" + "="*60)
    print("Training Forward-Forward Network (Correct Implementation)")
    print("="*60)
    print(f"Architecture: [784, 500, 500]")
    print(f"Epochs per layer: {epochs_per_layer}")
    print(f"Threshold: 2.0")
    print(f"Learning rate: 0.03")
    
    # Train
    train_results = train_ff(model, train_loader, device, epochs_per_layer)
    
    # Evaluate
    test_results = evaluate_ff(model, test_loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Train accuracy: {train_results['train_acc']*100:.2f}%")
    print(f"Train error:    {train_results['train_error']*100:.2f}%")
    print(f"Test accuracy:  {test_results['test_acc']*100:.2f}%")
    print(f"Test error:     {test_results['test_error']*100:.2f}%")
    print(f"Training time:  {train_results['train_time']:.1f}s")
    
    print("\n" + "-"*60)
    print("Reference (Hinton paper): ~1.4% test error (~98.6% accuracy)")
    print("Reference (mpezeshki repo): ~1.36% test error")
    print("-"*60)
    
    return model, train_results, test_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs per layer')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()
    
    model, train_res, test_res = main(epochs_per_layer=args.epochs, seed=args.seed)
