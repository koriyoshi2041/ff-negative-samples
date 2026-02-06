"""
Predictive Forward-Forward (PFF) Full Performance Test

Tests the PFF model on MNIST with:
1. Representation circuit loss tracking
2. Generative circuit loss tracking
3. Test accuracy measurement
4. Generative sampling capability

Uses smaller model (n_units=1000) for faster training.

Based on: Ororbia & Mali (2022) "The Predictive Forward-Forward Algorithm"
"""

import sys
import os
import json
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pff import PFFNetwork, create_pos_neg_samples, get_device


# =============================================================================
# Data Loading
# =============================================================================

def get_mnist_loaders(train_batch_size: int = 128, test_batch_size: int = 1000):
    """
    Get MNIST data loaders.

    Uses smaller batch size than standard FF since PFF does K iterations per sample.
    """
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])

    # Download to ff-research/data directory
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    train_loader = DataLoader(
        datasets.MNIST(data_path, train=True, download=True, transform=transform),
        batch_size=train_batch_size, shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST(data_path, train=False, download=True, transform=transform),
        batch_size=test_batch_size, shuffle=False
    )

    return train_loader, test_loader


# =============================================================================
# Training Function
# =============================================================================

def train_pff_epoch(
    model: PFFNetwork,
    train_loader: DataLoader,
    rep_opt: Adam,
    gen_opt: Adam,
    device: torch.device,
    reg_lambda: float = 0.0,
    g_reg_lambda: float = 0.0001
):
    """
    Train PFF for one epoch.

    Returns:
        Dict with epoch metrics
    """
    model.train()

    total_rep_loss = 0.0
    total_gen_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y_onehot = F.one_hot(batch_y, 10).float().to(device)

        # Create pos/neg samples
        x, y, lab, _ = create_pos_neg_samples(batch_x, batch_y_onehot)

        # Inference with learning
        rep_loss, y_hat, gen_loss, x_hat = model.infer(
            x, y, lab,
            rep_opt=rep_opt, gen_opt=gen_opt,
            reg_lambda=reg_lambda, g_reg_lambda=g_reg_lambda
        )

        batch_size = batch_x.shape[0]
        total_rep_loss += rep_loss.item() * batch_size
        total_gen_loss += gen_loss.item() * batch_size

        # Accuracy on positive samples only
        y_hat_pos = y_hat[:batch_size]
        pred = y_hat_pos.argmax(dim=1)
        target = batch_y_onehot.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += batch_size

    return {
        'rep_loss': total_rep_loss / total,
        'gen_loss': total_gen_loss / total,
        'train_acc': correct / total
    }


def test_pff(
    model: PFFNetwork,
    test_loader: DataLoader,
    device: torch.device
):
    """
    Test PFF model on test set.

    Uses simple forward classification (not K-step inference).
    """
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Simple forward pass for classification
            y_hat = model.classify(batch_x)
            pred = y_hat.argmax(dim=1)

            correct += (pred == batch_y).sum().item()
            total += batch_x.shape[0]

    return correct / total


# =============================================================================
# Sampling Function
# =============================================================================

def generate_samples(
    model: PFFNetwork,
    n_samples: int,
    device: torch.device,
    sample_method: str = 'random'
):
    """
    Generate samples from the PFF generative circuit.

    Args:
        model: Trained PFF model
        n_samples: Number of samples to generate
        device: Target device
        sample_method: 'random' for random latent, 'interpolate' for latent interpolation

    Returns:
        Generated images of shape (n_samples, 784)
    """
    model.eval()

    with torch.no_grad():
        if sample_method == 'random':
            # Sample from random Gaussian latent
            samples = model.sample(n_samples=n_samples, device=device)
        else:
            # Interpolation in latent space (for visualization)
            z1 = torch.randn(1, model.g_units, device=device)
            z2 = torch.randn(1, model.g_units, device=device)

            alphas = torch.linspace(0, 1, n_samples, device=device).view(-1, 1)
            z_interp = z1 * (1 - alphas) + z2 * alphas

            samples = model.sample(z_g=z_interp)

    return samples


def save_samples_image(samples: torch.Tensor, filepath: str, nrow: int = 8):
    """
    Save generated samples as an image grid.

    Args:
        samples: Tensor of shape (n_samples, 784)
        filepath: Output file path
        nrow: Number of images per row
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        n_samples = samples.shape[0]
        ncol = (n_samples + nrow - 1) // nrow

        fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 1.5, ncol * 1.5))
        axes = axes.flatten() if ncol > 1 or nrow > 1 else [axes]

        for i, ax in enumerate(axes):
            if i < n_samples:
                img = samples[i].cpu().numpy().reshape(28, 28)
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved samples to {filepath}")
        return True

    except ImportError:
        print("matplotlib not available, skipping sample visualization")
        return False


# =============================================================================
# Main Experiment
# =============================================================================

def main(
    n_units: int = 1000,
    g_units: int = 20,
    K: int = 12,
    num_epochs: int = 10,
    batch_size: int = 128,
    lr_rep: float = 0.001,
    lr_gen: float = 0.001,
    seed: int = 42
):
    """
    Run full PFF performance test on MNIST.

    Args:
        n_units: Hidden units per layer (smaller for faster training)
        g_units: Generative latent dimension
        K: Number of inference steps
        num_epochs: Number of training epochs
        batch_size: Training batch size
        lr_rep: Learning rate for representation circuit
        lr_gen: Learning rate for generative circuit
        seed: Random seed
    """
    # Setup
    torch.manual_seed(seed)
    device = get_device()

    print("=" * 70)
    print("Predictive Forward-Forward (PFF) Full Performance Test")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: n_units={n_units}, g_units={g_units}, K={K}")
    print(f"Training: {num_epochs} epochs, batch_size={batch_size}")
    print(f"Learning rates: rep={lr_rep}, gen={lr_gen}")
    print("=" * 70)

    # Data
    train_loader, test_loader = get_mnist_loaders(
        train_batch_size=batch_size,
        test_batch_size=1000
    )

    # Model
    model = PFFNetwork(
        x_dim=784,
        y_dim=10,
        n_units=n_units,
        g_units=g_units,
        K=K,
        thr=10.0,
        alpha=0.3,
        beta=0.025,
        eps_r=0.01,
        eps_g=0.025,
        use_lateral=True,
        use_generative=True
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizers
    rep_opt = Adam(model.rep_circuit.get_parameters(model.use_lateral), lr=lr_rep)
    gen_opt = Adam(model.gen_circuit.get_parameters(), lr=lr_gen)

    # Training history
    history = {
        'rep_loss': [],
        'gen_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epoch_time': []
    }

    # Training loop
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train one epoch
        metrics = train_pff_epoch(
            model, train_loader, rep_opt, gen_opt, device
        )

        # Test on test set
        test_acc = test_pff(model, test_loader, device)

        epoch_time = time.time() - epoch_start

        # Record history
        history['rep_loss'].append(metrics['rep_loss'])
        history['gen_loss'].append(metrics['gen_loss'])
        history['train_acc'].append(metrics['train_acc'])
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)

        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"RepLoss={metrics['rep_loss']:.4f}, "
              f"GenLoss={metrics['gen_loss']:.4f}, "
              f"TrainAcc={metrics['train_acc']*100:.2f}%, "
              f"TestAcc={test_acc*100:.2f}%, "
              f"Time={epoch_time:.1f}s")

    total_time = time.time() - start_time

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_results = {
        'model_config': {
            'n_units': n_units,
            'g_units': g_units,
            'K': K,
            'thr': model.thr,
            'alpha': model.alpha,
            'beta': model.beta,
            'total_params': total_params
        },
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr_rep': lr_rep,
            'lr_gen': lr_gen,
            'seed': seed
        },
        'final_metrics': {
            'rep_loss': history['rep_loss'][-1],
            'gen_loss': history['gen_loss'][-1],
            'train_acc': history['train_acc'][-1],
            'test_acc': history['test_acc'][-1]
        },
        'best_metrics': {
            'best_test_acc': max(history['test_acc']),
            'best_test_epoch': history['test_acc'].index(max(history['test_acc'])) + 1,
            'min_rep_loss': min(history['rep_loss']),
            'min_gen_loss': min(history['gen_loss'])
        },
        'training_time': {
            'total_seconds': total_time,
            'avg_epoch_seconds': total_time / num_epochs
        },
        'history': history,
        'timestamp': datetime.now().isoformat()
    }

    print(f"Final Rep Loss:    {final_results['final_metrics']['rep_loss']:.4f}")
    print(f"Final Gen Loss:    {final_results['final_metrics']['gen_loss']:.4f}")
    print(f"Final Train Acc:   {final_results['final_metrics']['train_acc']*100:.2f}%")
    print(f"Final Test Acc:    {final_results['final_metrics']['test_acc']*100:.2f}%")
    print(f"Best Test Acc:     {final_results['best_metrics']['best_test_acc']*100:.2f}% "
          f"(epoch {final_results['best_metrics']['best_test_epoch']})")
    print(f"Total Time:        {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save results
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results'
    )
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, 'pff_mnist_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate samples
    print("\n" + "-" * 70)
    print("Testing Generative Capability")
    print("-" * 70)

    try:
        # Random samples
        print("Generating random samples from latent space...")
        random_samples = generate_samples(model, n_samples=16, device=device, sample_method='random')
        print(f"  Sample shape: {random_samples.shape}")
        print(f"  Sample range: [{random_samples.min().item():.3f}, {random_samples.max().item():.3f}]")

        # Interpolation samples
        print("Generating interpolated samples...")
        interp_samples = generate_samples(model, n_samples=8, device=device, sample_method='interpolate')

        # Save sample images
        samples_path = os.path.join(results_dir, 'pff_samples.png')

        # Combine random and interpolation samples
        all_samples = torch.cat([random_samples, interp_samples], dim=0)

        if save_samples_image(all_samples, samples_path, nrow=8):
            print(f"Samples saved to: {samples_path}")

        # Also save reconstruction samples
        print("\nTesting reconstruction capability...")
        test_batch_x, test_batch_y = next(iter(test_loader))
        test_batch_x = test_batch_x[:8].to(device)
        test_batch_y_onehot = F.one_hot(test_batch_y[:8], 10).float().to(device)

        with torch.no_grad():
            # Get latent representation
            z_g = model.get_latent(test_batch_x, test_batch_y_onehot, K=K)
            # Reconstruct
            reconstructed = model.sample(z_g=z_g)

        print(f"  Reconstruction shape: {reconstructed.shape}")
        recon_error = F.mse_loss(reconstructed, test_batch_x).item()
        print(f"  Reconstruction MSE: {recon_error:.4f}")

        # Save original vs reconstructed
        recon_path = os.path.join(results_dir, 'pff_reconstruction.png')
        combined = torch.cat([test_batch_x, reconstructed], dim=0)
        save_samples_image(combined, recon_path, nrow=8)

    except Exception as e:
        print(f"Error during sample generation: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)

    return model, final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PFF Full Performance Test')
    parser.add_argument('--n_units', type=int, default=1000, help='Hidden units per layer')
    parser.add_argument('--g_units', type=int, default=20, help='Generative latent dimension')
    parser.add_argument('--K', type=int, default=12, help='Inference steps')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr_rep', type=float, default=0.001, help='Rep circuit LR')
    parser.add_argument('--lr_gen', type=float, default=0.001, help='Gen circuit LR')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    model, results = main(
        n_units=args.n_units,
        g_units=args.g_units,
        K=args.K,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr_rep=args.lr_rep,
        lr_gen=args.lr_gen,
        seed=args.seed
    )
