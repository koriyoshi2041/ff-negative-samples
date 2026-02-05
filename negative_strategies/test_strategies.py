#!/usr/bin/env python3
"""
Test script for negative sample strategies.

Tests:
1. Basic instantiation and configuration
2. Output shape correctness
3. Integration with FF baseline training (quick validation)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

from negative_strategies import (
    StrategyRegistry,
    NegativeStrategy,
    LabelEmbeddingStrategy,
    ImageMixingStrategy,
    RandomNoiseStrategy,
    ClassConfusionStrategy,
    SelfContrastiveStrategy,
    MaskingStrategy,
    LayerWiseStrategy,
    AdversarialStrategy,
    HardMiningStrategy,
    MonoForwardStrategy,
    list_strategies,
    create_strategy,
)


# ============================================================
# Test Utilities
# ============================================================

def create_dummy_data(batch_size=32, num_classes=10, device='cpu'):
    """Create dummy MNIST-like data for testing."""
    images = torch.randn(batch_size, 1, 28, 28, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    return images, labels


def test_strategy_basic(strategy: NegativeStrategy, name: str, device='cpu'):
    """Test basic strategy functionality."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Test 1: Configuration
    print(f"  Config: {strategy.get_config()}")
    print(f"  Requires labels: {strategy.requires_labels}")
    
    # Test 2: Generate with image input
    images, labels = create_dummy_data(batch_size=16, device=device)
    
    try:
        negatives = strategy.generate(images, labels)
        print(f"  ✓ Generate from images: {images.shape} -> {negatives.shape}")
    except Exception as e:
        print(f"  ✗ Generate from images failed: {e}")
        return False
    
    # Test 3: Generate with flattened input
    flat_images = images.view(16, -1)
    try:
        negatives_flat = strategy.generate(flat_images, labels)
        print(f"  ✓ Generate from flat: {flat_images.shape} -> {negatives_flat.shape}")
    except Exception as e:
        print(f"  ✗ Generate from flat failed: {e}")
        return False
    
    # Test 4: Check output shape matches
    expected_shape = (16, 28 * 28)
    if negatives.shape != expected_shape:
        print(f"  ✗ Shape mismatch: expected {expected_shape}, got {negatives.shape}")
        return False
    print(f"  ✓ Output shape correct: {negatives.shape}")
    
    # Test 5: Check no NaN/Inf
    if torch.isnan(negatives).any() or torch.isinf(negatives).any():
        print(f"  ✗ Output contains NaN or Inf")
        return False
    print(f"  ✓ No NaN/Inf in output")
    
    # Test 6: create_positive if available
    try:
        positives = strategy.create_positive(images, labels)
        print(f"  ✓ Create positive: {positives.shape}")
    except Exception as e:
        print(f"  ○ Create positive not implemented (using default)")
    
    print(f"\n  ★ {name} passed all tests!")
    return True


# ============================================================
# FF Layer for Integration Testing
# ============================================================

class SimpleFFLayer(nn.Module):
    """Simplified FF layer for testing."""
    
    def __init__(self, in_features: int, out_features: int, threshold: float = 2.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / (x.norm(2, dim=1, keepdim=True) + 1e-8)
        return self.relu(self.linear(x_norm))
    
    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        return (x ** 2).sum(dim=1)
    
    def ff_loss(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor) -> torch.Tensor:
        loss_pos = torch.log(1 + torch.exp(self.threshold - pos_goodness)).mean()
        loss_neg = torch.log(1 + torch.exp(neg_goodness - self.threshold)).mean()
        return loss_pos + loss_neg


def test_strategy_integration(strategy: NegativeStrategy, name: str, device='cpu', num_steps=10):
    """Test strategy integration with FF training loop."""
    print(f"\n{'='*60}")
    print(f"Integration Test: {name}")
    print(f"{'='*60}")
    
    # Create simple FF model
    layer = SimpleFFLayer(784, 500).to(device)
    optimizer = optim.Adam(layer.parameters(), lr=0.01)
    
    # Create data
    images, labels = create_dummy_data(batch_size=32, device=device)
    
    try:
        losses = []
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Get positive and negative samples
            pos_data = strategy.create_positive(images, labels)
            neg_data = strategy.generate(images, labels)
            
            # Forward pass
            pos_out = layer(pos_data)
            neg_out = layer(neg_data)
            
            # Compute loss
            pos_goodness = layer.goodness(pos_out)
            neg_goodness = layer.goodness(neg_out)
            loss = layer.ff_loss(pos_goodness, neg_goodness)
            
            # Check for valid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ✗ NaN/Inf loss at step {step}")
                return False
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        print(f"  ✓ Training completed {num_steps} steps")
        print(f"  ✓ Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
        print(f"  ★ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# MNIST Integration Test
# ============================================================

def test_mnist_training(strategy: NegativeStrategy, name: str, device='cpu', num_batches=5):
    """Quick MNIST training validation."""
    print(f"\n{'='*60}")
    print(f"MNIST Training Test: {name}")
    print(f"{'='*60}")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
    except Exception as e:
        print(f"  ○ Skipping MNIST test (data not available): {e}")
        return True  # Not a failure, just skip
    
    # Create model
    layer = SimpleFFLayer(784, 500).to(device)
    optimizer = optim.Adam(layer.parameters(), lr=0.01)
    
    try:
        total_loss = 0
        for i, (images, labels) in enumerate(loader):
            if i >= num_batches:
                break
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            pos_data = strategy.create_positive(images, labels)
            neg_data = strategy.generate(images, labels)
            
            pos_out = layer(pos_data)
            neg_out = layer(neg_data)
            
            loss = layer.ff_loss(layer.goodness(pos_out), layer.goodness(neg_out))
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ✗ Invalid loss at batch {i}")
                return False
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"  ✓ Trained {num_batches} batches on MNIST")
        print(f"  ✓ Average loss: {avg_loss:.4f}")
        print(f"  ★ MNIST test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ MNIST test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Main Test Runner
# ============================================================

def run_all_tests():
    """Run all strategy tests."""
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # List available strategies
    print("\n" + "="*60)
    print("Available Strategies:")
    print("="*60)
    list_strategies()
    
    # Test configurations
    strategies_to_test = [
        ('label_embedding', {'num_classes': 10}),
        ('image_mixing', {'num_classes': 10, 'mixing_mode': 'interpolate'}),
        ('random_noise', {'num_classes': 10, 'noise_type': 'gaussian'}),
        ('class_confusion', {'num_classes': 10, 'confusion_mode': 'random'}),
        ('self_contrastive', {'num_classes': 10}),
        ('masking', {'num_classes': 10, 'mask_ratio': 0.3}),
        ('layer_wise', {'num_classes': 10, 'layer_dims': [784, 500]}),
        ('adversarial', {'num_classes': 10, 'epsilon': 0.1}),
        ('hard_mining', {'num_classes': 10, 'mining_mode': 'class'}),
        ('mono_forward', {'num_classes': 10, 'target_goodness': 2.0}),
    ]
    
    results = {}
    
    for name, kwargs in strategies_to_test:
        try:
            strategy = create_strategy(name, **kwargs)
            strategy = strategy.to(torch.device(device))
            
            # Basic tests
            basic_ok = test_strategy_basic(strategy, name, device)
            
            # Integration test
            integration_ok = test_strategy_integration(strategy, name, device)
            
            # MNIST test
            mnist_ok = test_mnist_training(strategy, name, device)
            
            results[name] = {
                'basic': basic_ok,
                'integration': integration_ok,
                'mnist': mnist_ok,
                'all_passed': basic_ok and integration_ok and mnist_ok
            }
            
        except Exception as e:
            print(f"\n✗ Failed to test {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(e), 'all_passed': False}
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, result in results.items():
        if result.get('all_passed'):
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_passed = False
        
        if 'error' in result:
            print(f"  {name:<25} {status} (Error: {result['error'][:30]}...)")
        else:
            basic = "✓" if result.get('basic') else "✗"
            integ = "✓" if result.get('integration') else "✗"
            mnist = "✓" if result.get('mnist') else "✗"
            print(f"  {name:<25} {status} [basic:{basic} integ:{integ} mnist:{mnist}]")
    
    print("="*60)
    if all_passed:
        print("★ ALL TESTS PASSED! ★")
    else:
        print("✗ Some tests failed")
    print("="*60)
    
    return all_passed


def test_single_strategy(name: str, **kwargs):
    """Test a single strategy."""
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    strategy = create_strategy(name, **kwargs)
    strategy = strategy.to(torch.device(device))
    
    test_strategy_basic(strategy, name, device)
    test_strategy_integration(strategy, name, device)
    test_mnist_training(strategy, name, device)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test negative sample strategies')
    parser.add_argument('--strategy', type=str, default=None,
                       help='Test a specific strategy (default: all)')
    parser.add_argument('--list', action='store_true',
                       help='List available strategies')
    
    args = parser.parse_args()
    
    if args.list:
        list_strategies()
    elif args.strategy:
        test_single_strategy(args.strategy)
    else:
        run_all_tests()
