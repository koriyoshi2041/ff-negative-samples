"""
Layer Collaboration Forward-Forward Implementation

Based on: Lorberbom et al. (2024) "Layer Collaboration in the Forward-Forward Algorithm"
Paper: https://arxiv.org/abs/2305.12393

Key modifications from original FF:
1. Add γ (global goodness from other layers) to probability calculation
2. γ is detached from gradient computation
3. Train by alternating across all layers (not layer-by-layer to convergence)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict, Optional
import time


# ============================================================
# Layer Collaboration FF Layer
# ============================================================

class CollabFFLayer(nn.Module):
    """A Forward-Forward layer with layer collaboration support."""
    
    def __init__(self, in_features: int, out_features: int, threshold: float = 2.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.optimizer = None  # Set externally
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with layer normalization."""
        x_normalized = x / (x.norm(2, dim=1, keepdim=True) + 1e-8)
        return self.relu(self.linear(x_normalized))
    
    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute goodness (sum of squared activations)."""
        return (x ** 2).sum(dim=1)
    
    def ff_loss_with_gamma(self, 
                           pos_goodness: torch.Tensor, 
                           neg_goodness: torch.Tensor,
                           gamma_pos: torch.Tensor,
                           gamma_neg: torch.Tensor) -> torch.Tensor:
        """
        Layer Collaboration FF loss.
        
        Original FF:  p = sigmoid(goodness - θ)
        Collab FF:    p = sigmoid(goodness + γ - θ)
        
        where γ = sum of goodness from other layers (detached, no gradient)
        """
        # Collaborative probability calculation
        # p_pos = sigmoid(goodness_pos + gamma_pos - threshold)
        # loss_pos = -log(p_pos) = -log(sigmoid(x)) = log(1 + exp(-x)) = softplus(-x)
        # Since we want positive samples above threshold: minimize softplus(threshold - goodness - gamma)
        
        pos_logit = pos_goodness + gamma_pos - self.threshold
        neg_logit = neg_goodness + gamma_neg - self.threshold
        
        # Loss: push positive goodness up, negative down
        # For positive: we want sigmoid(logit) → 1, so minimize -log(sigmoid(logit)) = log(1 + exp(-logit))
        # For negative: we want sigmoid(logit) → 0, so minimize -log(1-sigmoid(logit)) = log(1 + exp(logit))
        loss_pos = torch.log(1 + torch.exp(-pos_logit)).mean()
        loss_neg = torch.log(1 + torch.exp(neg_logit)).mean()
        
        return loss_pos + loss_neg


# ============================================================
# Layer Collaboration FF Network
# ============================================================

class CollabFFNetwork(nn.Module):
    """
    Multi-layer Forward-Forward Network with Layer Collaboration.
    
    Key difference from original FF:
    - Each layer's loss includes γ = sum of other layers' goodness
    - γ is detached (constant for gradient computation)
    - Training alternates across layers instead of layer-by-layer convergence
    """
    
    def __init__(self, layer_sizes: List[int], threshold: float = 2.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.threshold = threshold
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(CollabFFLayer(layer_sizes[i], layer_sizes[i+1], threshold))
    
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
            goodness_list.append(g.detach())  # Detach for γ calculation
            h = h.detach()  # Detach between layers
        return goodness_list
    
    def compute_gamma(self, goodness_list: List[torch.Tensor], current_layer: int, 
                      mode: str = 'all') -> torch.Tensor:
        """
        Compute γ for a specific layer.
        
        Args:
            goodness_list: List of goodness values from all layers (detached)
            current_layer: Index of current layer
            mode: 'all' (all other layers) or 'previous' (only previous layers)
        
        Returns:
            γ: Sum of goodness from other layers
        """
        gamma = torch.zeros_like(goodness_list[0])
        
        for i, g in enumerate(goodness_list):
            if mode == 'all' and i != current_layer:
                gamma = gamma + g
            elif mode == 'previous' and i < current_layer:
                gamma = gamma + g
        
        return gamma
    
    def train_step_collaborative(self, 
                                  pos_data: torch.Tensor, 
                                  neg_data: torch.Tensor,
                                  gamma_mode: str = 'all') -> Dict[str, float]:
        """
        Train all layers with Layer Collaboration.
        
        Key differences from original FF:
        1. First compute all goodness values (detached)
        2. Then update each layer using γ from other layers
        3. Alternate across layers (one update per layer per step)
        """
        losses = {}
        
        # Step 1: Compute all goodness values (detached) for both pos and neg
        pos_goodness_list = self.compute_all_goodness(pos_data)
        neg_goodness_list = self.compute_all_goodness(neg_data)
        
        # Step 2: Update each layer with collaborative loss
        pos_input = pos_data
        neg_input = neg_data
        
        for i, layer in enumerate(self.layers):
            # Compute γ for this layer
            gamma_pos = self.compute_gamma(pos_goodness_list, i, gamma_mode)
            gamma_neg = self.compute_gamma(neg_goodness_list, i, gamma_mode)
            
            # Forward pass (with gradient)
            pos_output = layer(pos_input)
            neg_output = layer(neg_input)
            
            # Compute goodness (with gradient)
            pos_goodness = layer.goodness(pos_output)
            neg_goodness = layer.goodness(neg_output)
            
            # Collaborative loss
            loss = layer.ff_loss_with_gamma(pos_goodness, neg_goodness, gamma_pos, gamma_neg)
            
            # Backprop (local to this layer)
            layer.optimizer.zero_grad()
            loss.backward()
            layer.optimizer.step()
            
            losses[f'layer_{i}'] = loss.item()
            
            # Detach for next layer
            pos_input = pos_output.detach()
            neg_input = neg_output.detach()
        
        return losses


# ============================================================
# Original FF Network (for comparison)
# ============================================================

class OriginalFFNetwork(nn.Module):
    """Original Forward-Forward Network without layer collaboration."""
    
    def __init__(self, layer_sizes: List[int], threshold: float = 2.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.threshold = threshold
        
        for i in range(len(layer_sizes) - 1):
            layer = CollabFFLayer(layer_sizes[i], layer_sizes[i+1], threshold)
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations
    
    def train_step(self, pos_data: torch.Tensor, neg_data: torch.Tensor) -> Dict[str, float]:
        """Original FF training without collaboration."""
        losses = {}
        pos_input = pos_data
        neg_input = neg_data
        
        for i, layer in enumerate(self.layers):
            pos_output = layer(pos_input)
            neg_output = layer(neg_input)
            
            pos_goodness = layer.goodness(pos_output)
            neg_goodness = layer.goodness(neg_output)
            
            # Original loss (no gamma)
            loss_pos = torch.log(1 + torch.exp(self.threshold - pos_goodness)).mean()
            loss_neg = torch.log(1 + torch.exp(neg_goodness - self.threshold)).mean()
            loss = loss_pos + loss_neg
            
            layer.optimizer.zero_grad()
            loss.backward()
            layer.optimizer.step()
            
            losses[f'layer_{i}'] = loss.item()
            
            pos_input = pos_output.detach()
            neg_input = neg_output.detach()
        
        return losses


# ============================================================
# Negative Sample Strategy
# ============================================================

class LabelEmbeddingNegative:
    """Hinton's method: embed label in first pixels."""
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
    
    def create_positive(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)
        one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        result = flat.clone()
        result[:, :self.num_classes] = one_hot
        return result
    
    def generate(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)
        wrong_labels = torch.randint(0, self.num_classes, (batch_size,), device=images.device)
        mask = wrong_labels == labels
        wrong_labels[mask] = (wrong_labels[mask] + 1) % self.num_classes
        one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
        one_hot.scatter_(1, wrong_labels.unsqueeze(1), 1.0)
        result = flat.clone()
        result[:, :self.num_classes] = one_hot
        return result


# ============================================================
# Training Functions
# ============================================================

def train_epoch_collab(model: CollabFFNetwork,
                       train_loader: DataLoader,
                       neg_strategy: LabelEmbeddingNegative,
                       device: torch.device,
                       gamma_mode: str = 'all') -> Dict[str, float]:
    """Train collaborative FF for one epoch."""
    model.train()
    total_losses = {f'layer_{i}': 0.0 for i in range(len(model.layers))}
    num_batches = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        pos_data = neg_strategy.create_positive(images, labels)
        neg_data = neg_strategy.generate(images, labels)
        
        losses = model.train_step_collaborative(pos_data, neg_data, gamma_mode)
        
        for key in losses:
            total_losses[key] += losses[key]
        num_batches += 1
    
    return {k: v / num_batches for k, v in total_losses.items()}


def train_epoch_original(model: OriginalFFNetwork,
                         train_loader: DataLoader,
                         neg_strategy: LabelEmbeddingNegative,
                         device: torch.device) -> Dict[str, float]:
    """Train original FF for one epoch."""
    model.train()
    total_losses = {f'layer_{i}': 0.0 for i in range(len(model.layers))}
    num_batches = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        pos_data = neg_strategy.create_positive(images, labels)
        neg_data = neg_strategy.generate(images, labels)
        
        losses = model.train_step(pos_data, neg_data)
        
        for key in losses:
            total_losses[key] += losses[key]
        num_batches += 1
    
    return {k: v / num_batches for k, v in total_losses.items()}


def evaluate(model: nn.Module,
             test_loader: DataLoader,
             neg_strategy: LabelEmbeddingNegative,
             device: torch.device,
             num_classes: int = 10) -> float:
    """Evaluate FF network accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            best_goodness = torch.zeros(batch_size, device=device) - float('inf')
            predictions = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            for candidate_label in range(num_classes):
                candidate_labels = torch.full((batch_size,), candidate_label, device=device)
                pos_data = neg_strategy.create_positive(images, candidate_labels)
                
                activations = model(pos_data)
                # Sum goodness across all layers for prediction
                total_goodness = sum(model.layers[i].goodness(activations[i]) 
                                    for i in range(len(model.layers)))
                
                better = total_goodness > best_goodness
                predictions[better] = candidate_label
                best_goodness[better] = total_goodness[better]
            
            correct += (predictions == labels).sum().item()
            total += batch_size
    
    return correct / total


# ============================================================
# Main Comparison Experiment
# ============================================================

def run_comparison(num_epochs: int = 20, 
                   seed: int = 42,
                   threshold: float = 2.0,
                   lr: float = 0.03) -> Dict:
    """
    Compare Original FF vs Layer Collaboration FF on MNIST.
    
    Expected results (from paper):
    - Original FF: ~3.3% error
    - Layer Collab FF: ~2.1% error
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Architecture
    input_size = 28 * 28
    layer_sizes = [input_size, 500, 500]
    
    # Negative strategy
    neg_strategy = LabelEmbeddingNegative(num_classes=10)
    
    results = {
        'original': {'train_loss': [], 'test_acc': []},
        'collab_all': {'train_loss': [], 'test_acc': []},
        'collab_prev': {'train_loss': [], 'test_acc': []}
    }
    
    # ============================================================
    # Train Original FF
    # ============================================================
    print("\n" + "="*60)
    print("Training: Original Forward-Forward")
    print("="*60)
    
    torch.manual_seed(seed)  # Reset for fair comparison
    model_orig = OriginalFFNetwork(layer_sizes, threshold).to(device)
    for layer in model_orig.layers:
        layer.optimizer = optim.Adam(layer.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        start = time.time()
        losses = train_epoch_original(model_orig, train_loader, neg_strategy, device)
        acc = evaluate(model_orig, test_loader, neg_strategy, device)
        elapsed = time.time() - start
        
        total_loss = sum(losses.values())
        results['original']['train_loss'].append(total_loss)
        results['original']['test_acc'].append(acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {total_loss:.4f} | "
                  f"Acc: {acc*100:.2f}% | Time: {elapsed:.1f}s")
    
    # ============================================================
    # Train Layer Collab FF (all layers)
    # ============================================================
    print("\n" + "="*60)
    print("Training: Layer Collaboration FF (γ = all other layers)")
    print("="*60)
    
    torch.manual_seed(seed)
    model_collab = CollabFFNetwork(layer_sizes, threshold).to(device)
    for layer in model_collab.layers:
        layer.optimizer = optim.Adam(layer.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        start = time.time()
        losses = train_epoch_collab(model_collab, train_loader, neg_strategy, device, 'all')
        acc = evaluate(model_collab, test_loader, neg_strategy, device)
        elapsed = time.time() - start
        
        total_loss = sum(losses.values())
        results['collab_all']['train_loss'].append(total_loss)
        results['collab_all']['test_acc'].append(acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {total_loss:.4f} | "
                  f"Acc: {acc*100:.2f}% | Time: {elapsed:.1f}s")
    
    # ============================================================
    # Train Layer Collab FF (previous layers only)
    # ============================================================
    print("\n" + "="*60)
    print("Training: Layer Collaboration FF (γ = previous layers)")
    print("="*60)
    
    torch.manual_seed(seed)
    model_collab_prev = CollabFFNetwork(layer_sizes, threshold).to(device)
    for layer in model_collab_prev.layers:
        layer.optimizer = optim.Adam(layer.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        start = time.time()
        losses = train_epoch_collab(model_collab_prev, train_loader, neg_strategy, device, 'previous')
        acc = evaluate(model_collab_prev, test_loader, neg_strategy, device)
        elapsed = time.time() - start
        
        total_loss = sum(losses.values())
        results['collab_prev']['train_loss'].append(total_loss)
        results['collab_prev']['test_acc'].append(acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {total_loss:.4f} | "
                  f"Acc: {acc*100:.2f}% | Time: {elapsed:.1f}s")
    
    # ============================================================
    # Final Results
    # ============================================================
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    final_orig = results['original']['test_acc'][-1]
    final_collab_all = results['collab_all']['test_acc'][-1]
    final_collab_prev = results['collab_prev']['test_acc'][-1]
    
    print(f"\n{'Method':<35} {'Accuracy':>10} {'Error':>10}")
    print("-" * 60)
    print(f"{'Original FF':<35} {final_orig*100:>9.2f}% {(1-final_orig)*100:>9.2f}%")
    print(f"{'Layer Collab FF (γ=all)':<35} {final_collab_all*100:>9.2f}% {(1-final_collab_all)*100:>9.2f}%")
    print(f"{'Layer Collab FF (γ=previous)':<35} {final_collab_prev*100:>9.2f}% {(1-final_collab_prev)*100:>9.2f}%")
    
    print("\nPaper reference (MNIST):")
    print(f"  Original FF:     3.3% error")
    print(f"  Layer Collab:    2.1% error")
    
    improvement_all = (final_collab_all - final_orig) / final_orig * 100
    improvement_prev = (final_collab_prev - final_orig) / final_orig * 100
    print(f"\nImprovement over Original FF:")
    print(f"  γ=all:      {improvement_all:+.2f}% relative")
    print(f"  γ=previous: {improvement_prev:+.2f}% relative")
    
    return results


if __name__ == "__main__":
    results = run_comparison(num_epochs=20, seed=42)
