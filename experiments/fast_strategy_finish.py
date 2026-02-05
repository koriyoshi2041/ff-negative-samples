"""
快速完成剩余策略实验
只运行: masking, layer_wise, adversarial, hard_mining, mono_forward
不包含 self_contrastive（太慢）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.strategy_comparison import (
    FFNetwork, train_epoch, evaluate, 
    run_single_experiment, find_convergence_epoch
)
from negative_strategies import (
    MaskingStrategy, LayerWiseStrategy, AdversarialStrategy,
    HardMiningStrategy, MonoForwardStrategy, LabelEmbeddingStrategy
)

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Strategies to test
    strategies = {
        'masking': MaskingStrategy(num_classes=10, mask_ratio=0.5, device=device),
        'layer_wise': LayerWiseStrategy(num_classes=10, perturbation_scale=0.5, device=device),
        'adversarial': AdversarialStrategy(num_classes=10, epsilon=0.1, num_steps=1, device=device),
        'hard_mining': HardMiningStrategy(num_classes=10, mining_mode='class', device=device),
        'mono_forward': MonoForwardStrategy(num_classes=10, device=device),
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\n[{name}] Starting...", flush=True)
        
        result = run_single_experiment(
            strategy_name=name,
            strategy=strategy,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=10,
            lr=0.03,
            seed=42
        )
        
        results[name] = {
            'mean_accuracy': result['final_accuracy'],
            'std_accuracy': 0.0,
            'mean_time': result['total_time'],
            'std_time': 0.0,
            'mean_convergence_epoch': result['convergence_epoch'],
            'accuracies': result['accuracies'],
        }
        
        print(f"  → Final: {result['final_accuracy']*100:.2f}%", flush=True)
    
    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'remaining_strategies.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

if __name__ == '__main__':
    main()
