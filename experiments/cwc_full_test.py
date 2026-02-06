"""
CwC-FF Full Test on MNIST - 20 Epochs
Complete performance evaluation of Channel-wise Competitive Forward-Forward

This script:
1. Uses create_cwc_mnist() from models/cwc_ff.py
2. Trains for full 20 epochs
3. Records per-epoch training loss and test accuracy
4. Saves results to JSON and generates learning curve
5. Compares with standard FF baseline
"""

import sys
import os
import json
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.cwc_ff import (
    create_cwc_mnist,
    get_mnist_loaders,
    get_device,
    CwCFFNetwork
)


def train_cwc_with_detailed_logging(
    model: CwCFFNetwork,
    train_loader,
    test_loader,
    num_epochs: int,
    device: torch.device,
    verbose: bool = True
) -> dict:
    """
    Train CwC-FF network with detailed per-epoch logging.

    Returns:
        Dict with detailed training history
    """
    model.to(device)

    history = {
        'epoch': [],
        'train_loss': [],          # Average loss across all layers
        'layer_losses': [],        # Per-layer losses
        'train_acc': [],
        'test_acc': [],
        'epoch_time': []
    }

    print(f"\n{'='*60}")
    print(f"CwC-FF Training - {num_epochs} Epochs")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Architecture: {[l.out_channels for l in model.layers]} channels")
    print(f"CFSE enabled: {model.use_cfse}")
    print(f"Loss type: {model.layers[0].loss_type}")
    print(f"{'='*60}\n")

    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.set_train_mode()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            h = x
            for layer_idx, layer in enumerate(model.layers):
                # Check ILT schedule
                start, end = model.ilt_schedule[layer_idx]

                if start <= epoch < end:
                    # Train this layer
                    h = layer.train_step(h, y)
                else:
                    # Just forward (frozen)
                    h = layer.infer(h)

        # Get layer losses for this epoch
        layer_losses = [layer.get_epoch_loss() for layer in model.layers]
        avg_loss = sum(layer_losses) / len(layer_losses) if layer_losses else 0.0

        # Evaluate
        train_acc = model.get_accuracy(train_loader, device, method='GA')
        test_acc = model.get_accuracy(test_loader, device, method='GA')

        epoch_time = time.time() - epoch_start

        # Record
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['layer_losses'].append(layer_losses)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)

        if verbose:
            layer_loss_str = ", ".join([f"L{i}:{l:.4f}" for i, l in enumerate(layer_losses)])
            print(f"Epoch {epoch+1:2d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc*100:.2f}% | "
                  f"Test Acc: {test_acc*100:.2f}% | "
                  f"Time: {epoch_time:.1f}s")
            if (epoch + 1) % 5 == 0:
                print(f"          Layer losses: {layer_loss_str}")

    total_time = time.time() - total_start

    # Final results
    results = {
        'model': 'CwC-FF',
        'dataset': 'MNIST',
        'architecture': [l.out_channels for l in model.layers],
        'num_epochs': num_epochs,
        'batch_size': train_loader.batch_size,
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'final_train_error': 1.0 - history['train_acc'][-1],
        'final_test_error': 1.0 - history['test_acc'][-1],
        'total_time': total_time,
        'history': history,
        'reference_error': 0.0058,  # Paper: 0.58% error
        'timestamp': datetime.now().isoformat()
    }

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Train Accuracy: {results['final_train_acc']*100:.2f}%")
    print(f"Test Accuracy:  {results['final_test_acc']*100:.2f}%")
    print(f"Test Error:     {results['final_test_error']*100:.2f}%")
    print(f"Total Time:     {total_time:.1f}s")
    print(f"Reference (paper): ~0.58% test error")
    print(f"{'='*60}")

    return results


def plot_learning_curve(history: dict, save_path: str):
    """Generate and save learning curve plot."""
    epochs = history['epoch']
    train_acc = [a * 100 for a in history['train_acc']]
    test_acc = [a * 100 for a in history['test_acc']]
    train_loss = history['train_loss']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1 = axes[0]
    ax1.plot(epochs, train_acc, 'b-o', label='Train Accuracy', markersize=4)
    ax1.plot(epochs, test_acc, 'r-o', label='Test Accuracy', markersize=4)
    ax1.axhline(y=99.42, color='g', linestyle='--', label='Reference (99.42%)', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('CwC-FF on MNIST: Accuracy', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(min(train_acc), min(test_acc)) - 5, 100])

    # Loss plot
    ax2 = axes[1]
    ax2.plot(epochs, train_loss, 'g-o', label='Training Loss', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('CwC-FF on MNIST: Training Loss', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Per-layer loss subplot
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    layer_losses = np.array(history['layer_losses'])
    for i in range(layer_losses.shape[1]):
        ax3.plot(epochs, layer_losses[:, i], '-o', label=f'Layer {i}', markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('CwC-FF on MNIST: Per-Layer Training Loss', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')

    # Save per-layer plot separately
    layer_save_path = save_path.replace('.png', '_per_layer.png')
    fig2.savefig(layer_save_path, dpi=150, bbox_inches='tight')

    plt.close(fig)
    plt.close(fig2)

    print(f"Learning curve saved to: {save_path}")
    print(f"Per-layer loss saved to: {layer_save_path}")


def run_standard_ff_baseline(train_loader, test_loader, device, num_epochs=20):
    """Run standard FF for comparison (simplified version)."""
    from models.ff_correct import FFNetwork, overlay_y_on_x

    print(f"\n{'='*60}")
    print("Standard FF Baseline (for comparison)")
    print(f"{'='*60}")

    # Get full batch for FF training
    x_train, y_train = [], []
    for x, y in train_loader:
        x_train.append(x)
        y_train.append(y)
    x_train = torch.cat(x_train, dim=0)
    y_train = torch.cat(y_train, dim=0)

    # Flatten for MLP
    x_train_flat = x_train.view(x_train.size(0), -1).to(device)
    y_train = y_train.to(device)

    # Create model
    model = FFNetwork([784, 500, 500], threshold=2.0, lr=0.03).to(device)

    # Create pos/neg samples
    x_pos = overlay_y_on_x(x_train_flat, y_train)
    rnd = torch.randperm(x_train_flat.size(0))
    x_neg = overlay_y_on_x(x_train_flat, y_train[rnd])

    # Train greedy (epochs_per_layer based on total epochs)
    epochs_per_layer = max(100, num_epochs * 50)  # Scale appropriately

    start_time = time.time()
    model.train_greedy(x_pos, x_neg, epochs_per_layer=epochs_per_layer, verbose=True)
    train_time = time.time() - start_time

    # Evaluate
    train_acc = model.get_accuracy(x_train_flat, y_train)

    # Test set
    x_test, y_test = [], []
    for x, y in test_loader:
        x_test.append(x)
        y_test.append(y)
    x_test = torch.cat(x_test, dim=0).view(-1, 784).to(device)
    y_test = torch.cat(y_test, dim=0).to(device)

    test_acc = model.get_accuracy(x_test, y_test)

    results = {
        'model': 'Standard-FF',
        'dataset': 'MNIST',
        'architecture': [784, 500, 500],
        'final_train_acc': train_acc,
        'final_test_acc': test_acc,
        'final_test_error': 1.0 - test_acc,
        'total_time': train_time,
        'reference_error': 0.014  # Paper: ~1.4% error
    }

    print(f"\nStandard FF Results:")
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    print(f"  Test Error:     {(1-test_acc)*100:.2f}%")
    print(f"  Reference: ~1.4% test error")

    return results


def main():
    # Setup
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = get_device()
    print(f"Using device: {device}")

    # Data
    batch_size = 64
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    # Create CwC-FF model
    model = create_cwc_mnist()

    # Train with detailed logging
    results = train_cwc_with_detailed_logging(
        model, train_loader, test_loader,
        num_epochs=20,
        device=device,
        verbose=True
    )

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save JSON (convert numpy to python types for JSON serialization)
    results_json = {
        'model': results['model'],
        'dataset': results['dataset'],
        'architecture': results['architecture'],
        'num_epochs': results['num_epochs'],
        'batch_size': results['batch_size'],
        'final_train_acc': float(results['final_train_acc']),
        'final_test_acc': float(results['final_test_acc']),
        'final_train_error': float(results['final_train_error']),
        'final_test_error': float(results['final_test_error']),
        'total_time': float(results['total_time']),
        'reference_error': results['reference_error'],
        'timestamp': results['timestamp'],
        'history': {
            'epoch': results['history']['epoch'],
            'train_loss': [float(x) for x in results['history']['train_loss']],
            'layer_losses': [[float(l) for l in epoch_losses]
                           for epoch_losses in results['history']['layer_losses']],
            'train_acc': [float(x) for x in results['history']['train_acc']],
            'test_acc': [float(x) for x in results['history']['test_acc']],
            'epoch_time': [float(x) for x in results['history']['epoch_time']]
        }
    }

    json_path = os.path.join(results_dir, 'cwc_ff_mnist_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate learning curve
    plot_path = os.path.join(results_dir, 'cwc_ff_learning_curve.png')
    plot_learning_curve(results['history'], plot_path)

    # Run standard FF baseline for comparison (optional - can be slow)
    print("\n" + "="*60)
    print("COMPARISON WITH STANDARD FF")
    print("="*60)

    try:
        ff_results = run_standard_ff_baseline(
            train_loader, test_loader, device, num_epochs=20
        )

        # Save comparison
        comparison = {
            'cwc_ff': {
                'test_acc': float(results['final_test_acc']),
                'test_error': float(results['final_test_error']),
                'time': float(results['total_time'])
            },
            'standard_ff': {
                'test_acc': float(ff_results['final_test_acc']),
                'test_error': float(ff_results['final_test_error']),
                'time': float(ff_results['total_time'])
            }
        }

        comparison_path = os.path.join(results_dir, 'cwc_ff_vs_standard_ff.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"\n{'='*60}")
        print("FINAL COMPARISON")
        print(f"{'='*60}")
        print(f"CwC-FF:      {results['final_test_acc']*100:.2f}% accuracy, "
              f"{results['final_test_error']*100:.2f}% error")
        print(f"Standard FF: {ff_results['final_test_acc']*100:.2f}% accuracy, "
              f"{ff_results['final_test_error']*100:.2f}% error")
        print(f"\nCwC-FF improvement: "
              f"{(ff_results['final_test_error'] - results['final_test_error'])*100:.2f}% error reduction")

    except Exception as e:
        print(f"Standard FF comparison skipped: {e}")

    return results


if __name__ == "__main__":
    results = main()
