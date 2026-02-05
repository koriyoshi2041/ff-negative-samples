"""
Forward-Forward Algorithm Baseline Implementation
For RQ1 (Transfer Learning) and RQ2 (Negative Sample Comparison)

Based on: https://github.com/mpezeshki/pytorch_forward_forward
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List, Dict
import time

# ============================================================
# FF Layer Implementation
# ============================================================

class FFLayer(nn.Module):
    """A single Forward-Forward layer with local learning."""
    
    def __init__(self, in_features: int, out_features: int, threshold: float = 2.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.optimizer = None  # Set later
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = x / (x.norm(2, dim=1, keepdim=True) + 1e-8)
        return self.relu(self.linear(x_normalized))
    
    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute goodness (sum of squared activations)."""
        return (x ** 2).sum(dim=1)
    
    def ff_loss(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor) -> torch.Tensor:
        """FF loss: push positive goodness above threshold, negative below."""
        loss_pos = torch.log(1 + torch.exp(self.threshold - pos_goodness)).mean()
        loss_neg = torch.log(1 + torch.exp(neg_goodness - self.threshold)).mean()
        return loss_pos + loss_neg


class FFNetwork(nn.Module):
    """Multi-layer Forward-Forward Network."""
    
    def __init__(self, layer_sizes: List[int], threshold: float = 2.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(FFLayer(layer_sizes[i], layer_sizes[i+1], threshold))
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return activations from all layers."""
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations
    
    def train_step(self, pos_data: torch.Tensor, neg_data: torch.Tensor) -> Dict[str, float]:
        """Train all layers with FF algorithm."""
        losses = {}
        pos_input = pos_data
        neg_input = neg_data
        
        for i, layer in enumerate(self.layers):
            # Forward pass
            pos_output = layer(pos_input)
            neg_output = layer(neg_input)
            
            # Compute goodness
            pos_goodness = layer.goodness(pos_output)
            neg_goodness = layer.goodness(neg_output)
            
            # Compute and backprop loss (local!)
            loss = layer.ff_loss(pos_goodness, neg_goodness)
            
            layer.optimizer.zero_grad()
            loss.backward()
            layer.optimizer.step()
            
            losses[f'layer_{i}'] = loss.item()
            
            # Detach for next layer (no gradient flow between layers!)
            pos_input = pos_output.detach()
            neg_input = neg_output.detach()
        
        return losses


# ============================================================
# Negative Sample Strategies
# ============================================================

class NegativeSampleStrategy:
    """Base class for negative sample generation."""
    
    def generate(self, images: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        raise NotImplementedError


class LabelEmbeddingNegative(NegativeSampleStrategy):
    """Original Hinton method: embed wrong label in first pixels."""
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
    
    def create_positive(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Create positive samples with correct label embedded."""
        batch_size = images.size(0)
        # Flatten images
        flat = images.view(batch_size, -1)
        # Create one-hot labels
        one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        # Embed label in first pixels
        result = flat.clone()
        result[:, :self.num_classes] = one_hot
        return result
    
    def generate(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Create negative samples with wrong label embedded."""
        batch_size = images.size(0)
        # Flatten images
        flat = images.view(batch_size, -1)
        # Generate random wrong labels
        wrong_labels = torch.randint(0, self.num_classes, (batch_size,), device=images.device)
        # Make sure they're actually wrong
        mask = wrong_labels == labels
        wrong_labels[mask] = (wrong_labels[mask] + 1) % self.num_classes
        # Create one-hot
        one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
        one_hot.scatter_(1, wrong_labels.unsqueeze(1), 1.0)
        # Embed wrong label
        result = flat.clone()
        result[:, :self.num_classes] = one_hot
        return result


class HybridImageNegative(NegativeSampleStrategy):
    """Hinton's unsupervised method: mix two different images."""
    
    def generate(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)
        
        # Shuffle to get different images
        perm = torch.randperm(batch_size, device=images.device)
        other_images = flat[perm]
        
        # Create random mask
        mask = (torch.rand(flat.shape, device=images.device) > 0.5).float()
        
        # Mix images
        negative = flat * mask + other_images * (1 - mask)
        return negative


class SCFFNegative(NegativeSampleStrategy):
    """Self-Contrastive FF: concatenate different images."""
    
    def generate(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)
        
        # Shuffle to get different images
        perm = torch.randperm(batch_size, device=images.device)
        other_images = flat[perm]
        
        # For negative: combine different images (sum instead of concat for same dimension)
        negative = (flat + other_images) / 2
        return negative
    
    def create_positive(self, images: torch.Tensor) -> torch.Tensor:
        """Positive: same image combined with itself."""
        flat = images.view(images.size(0), -1)
        return flat  # Or (flat + flat) / 2 for consistency


# ============================================================
# Training and Evaluation
# ============================================================

def train_ff_epoch(model: FFNetwork, 
                   train_loader: DataLoader, 
                   neg_strategy: NegativeSampleStrategy,
                   device: torch.device) -> Dict[str, float]:
    """Train FF network for one epoch."""
    model.train()
    total_losses = {f'layer_{i}': 0.0 for i in range(len(model.layers))}
    num_batches = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate positive and negative samples
        if isinstance(neg_strategy, LabelEmbeddingNegative):
            pos_data = neg_strategy.create_positive(images, labels)
        else:
            pos_data = images.view(images.size(0), -1)
        
        neg_data = neg_strategy.generate(images, labels)
        
        # Train step
        losses = model.train_step(pos_data, neg_data)
        
        for key in losses:
            total_losses[key] += losses[key]
        num_batches += 1
    
    return {k: v / num_batches for k, v in total_losses.items()}


def evaluate_ff(model: FFNetwork, 
                test_loader: DataLoader, 
                neg_strategy: NegativeSampleStrategy,
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
            
            # For each sample, try all labels and pick the one with highest goodness
            best_goodness = torch.zeros(batch_size, device=device) - float('inf')
            predictions = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            for candidate_label in range(num_classes):
                # Create positive sample with this candidate label
                candidate_labels = torch.full((batch_size,), candidate_label, device=device)
                if isinstance(neg_strategy, LabelEmbeddingNegative):
                    pos_data = neg_strategy.create_positive(images, candidate_labels)
                else:
                    pos_data = images.view(batch_size, -1)
                
                # Get final layer activations
                activations = model(pos_data)
                goodness = model.layers[-1].goodness(activations[-1])
                
                # Update best predictions
                better = goodness > best_goodness
                predictions[better] = candidate_label
                best_goodness[better] = goodness[better]
            
            correct += (predictions == labels).sum().item()
            total += batch_size
    
    return correct / total


# ============================================================
# Main Experiment
# ============================================================

def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model
    input_size = 28 * 28  # MNIST
    layer_sizes = [input_size, 500, 500]
    model = FFNetwork(layer_sizes, threshold=2.0).to(device)
    
    # Setup optimizers for each layer
    for layer in model.layers:
        layer.optimizer = optim.Adam(layer.parameters(), lr=0.03)
    
    # Negative sample strategy
    neg_strategy = LabelEmbeddingNegative(num_classes=10)
    
    # Training
    num_epochs = 10
    print("\n" + "="*50)
    print("Training Forward-Forward Network on MNIST")
    print("="*50)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        losses = train_ff_epoch(model, train_loader, neg_strategy, device)
        epoch_time = time.time() - start_time
        
        # Evaluate
        accuracy = evaluate_ff(model, test_loader, neg_strategy, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {sum(losses.values()):.4f} | "
              f"Accuracy: {accuracy*100:.2f}% | "
              f"Time: {epoch_time:.1f}s")
    
    print("\n" + "="*50)
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    print("="*50)
    
    return model, accuracy


if __name__ == "__main__":
    main()
