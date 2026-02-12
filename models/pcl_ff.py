"""
Predictive Coding Light Forward-Forward (PCL-FF) Implementation

Based on Predictive Coding Light (PCL) from neuroscience (Nature Communications, 2025)

Key Innovation:
- Standard FF transmits "goodness" (sum of squared activations)
- PCL-FF transmits "surprise" (what CANNOT be predicted from current representation)
- This naturally produces compressed, information-rich representations

Core Mechanism:
1. Each layer tries to PREDICT its input
2. Predictable parts are SUPPRESSED (not transmitted)
3. Unpredictable parts (surprise/novelty) are transmitted
4. This naturally produces sparse, abstract representations

Biological Plausibility:
- Completely local learning (no backprop through layers)
- Each layer only needs its own input and output
- Predictors learn through local reconstruction error

Hypothesis:
PCL-FF representations should be more abstract and generalizable,
leading to better transfer learning performance.

Author: Clawd (for Parafee)
Date: 2026-02-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Replace the first 10 pixels of data [x] with one-hot-encoded label [y].

    CRITICAL: Use x.max() as the label value, not 1.0!
    This ensures the label signal is comparable in magnitude to the image data.
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


# ============================================================
# PCL-FF Layer Implementation
# ============================================================

class PCLFFLayer(nn.Module):
    """
    Predictive Coding Light Forward-Forward Layer.

    Key differences from standard FF:
    1. Has a predictor that tries to reconstruct the input
    2. Computes "surprise" - what cannot be predicted
    3. Suppresses predictable activations, transmits surprise
    4. Goodness = compression efficiency + reconstruction quality

    The core insight: instead of just pushing goodness up/down,
    we learn representations that are PREDICTIVELY USEFUL.
    """

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03,
                 alpha: float = 0.5, sparsity_weight: float = 0.1,
                 surprise_scale: float = 1.0):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension (hidden size)
            threshold: Goodness threshold for FF training
            lr: Learning rate
            alpha: Weight for reconstruction loss in goodness computation
            sparsity_weight: Weight for sparsity regularization
            surprise_scale: How much to scale the surprise mask (0=no suppression, 1=full)
        """
        super().__init__()

        # Forward encoder: input -> hidden
        self.encoder = nn.Linear(in_features, out_features)

        # Backward predictor: hidden -> input (predicts the input)
        self.predictor = nn.Linear(out_features, in_features)

        self.relu = nn.ReLU()
        self.threshold = threshold
        self.alpha = alpha
        self.sparsity_weight = sparsity_weight
        self.surprise_scale = surprise_scale

        # Separate optimizer for encoder and predictor
        self.opt = Adam(self.parameters(), lr=lr)

        # Tracking metrics
        self.training_history = {
            'loss': [],
            'g_pos': [],
            'g_neg': [],
            'reconstruction_loss': [],
            'sparsity': [],
            'surprise_ratio': []
        }

    def forward(self, x: torch.Tensor,
                apply_surprise_mask: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with predictive coding.

        Returns:
            h: Hidden representation (potentially masked by surprise)
            x_pred: Predicted input reconstruction
        """
        # L2 normalize input (standard FF preprocessing)
        x_normed = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)

        # Encode
        h_raw = self.relu(self.encoder(x_normed))

        # Predict input from hidden
        x_pred = self.predictor(h_raw)

        if apply_surprise_mask:
            # Compute per-sample prediction error (surprise signal)
            pred_error = F.mse_loss(x_pred, x_normed, reduction='none').mean(dim=1)

            # Normalize to [0, 1] range
            # Higher error = more surprise = more information to transmit
            # We use sigmoid to bound the surprise signal
            surprise = torch.sigmoid(pred_error * 10 - 5)  # Centered around 0.5

            # Create surprise mask: amplify when input is surprising
            # When surprise is high, we transmit more; when low, we suppress
            surprise_mask = 1 - self.surprise_scale * (1 - surprise.unsqueeze(1))

            # Apply mask to hidden representation
            h = h_raw * surprise_mask
        else:
            h = h_raw

        return h, x_pred

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Get hidden representation only (for downstream use)."""
        h, _ = self.forward(x, apply_surprise_mask=True)
        return h

    def goodness(self, h: torch.Tensor, x: torch.Tensor,
                 x_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute goodness with predictive coding components.

        Goodness = base_goodness - sparsity_penalty + reconstruction_bonus

        - base_goodness: Mean squared activations (standard FF)
        - sparsity_penalty: Encourages sparse representations
        - reconstruction_bonus: Rewards accurate predictions
        """
        # Base goodness (standard FF: mean of squared activations)
        base_goodness = h.pow(2).mean(dim=1)

        # Reconstruction quality (negative MSE, so higher is better)
        x_normed = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        reconstruction_bonus = -F.mse_loss(x_pred, x_normed, reduction='none').mean(dim=1)

        # Sparsity penalty (L1 norm, encourages sparse representations)
        sparsity_penalty = h.abs().mean(dim=1) * self.sparsity_weight

        # Combined goodness
        goodness = base_goodness + self.alpha * reconstruction_bonus - sparsity_penalty

        return goodness

    def simple_goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Simple goodness for prediction (compatible with standard FF)."""
        return h.pow(2).mean(dim=1)

    def train_layer(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                    num_epochs: int = 500, verbose: bool = True,
                    log_interval: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train this layer with PCL-FF objective.

        The training combines:
        1. Standard FF objective: positive goodness > threshold, negative < threshold
        2. Reconstruction objective: predictor learns to reconstruct input
        3. Sparsity regularization: encourages compressed representations
        """
        iterator = tqdm(range(num_epochs), desc="Training PCL-FF layer") if verbose else range(num_epochs)

        for epoch in iterator:
            # Forward pass for positive and negative samples
            h_pos, x_pos_pred = self.forward(x_pos, apply_surprise_mask=True)
            h_neg, x_neg_pred = self.forward(x_neg, apply_surprise_mask=True)

            # Compute PCL goodness
            g_pos = self.goodness(h_pos, x_pos, x_pos_pred)
            g_neg = self.goodness(h_neg, x_neg, x_neg_pred)

            # FF Loss: push positive above threshold, negative below
            ff_loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold
            ]))).mean()

            # Reconstruction loss (train the predictor)
            x_pos_normed = x_pos / (x_pos.norm(2, dim=1, keepdim=True) + 1e-4)
            x_neg_normed = x_neg / (x_neg.norm(2, dim=1, keepdim=True) + 1e-4)
            recon_loss = (F.mse_loss(x_pos_pred, x_pos_normed) +
                         F.mse_loss(x_neg_pred, x_neg_normed)) / 2

            # Combined loss
            total_loss = ff_loss + self.alpha * recon_loss

            # Backward and update
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()

            # Track metrics
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                with torch.no_grad():
                    sparsity = (h_pos.abs() < 0.1).float().mean().item()
                    h_pos_raw, _ = self.forward(x_pos, apply_surprise_mask=False)
                    surprise_ratio = (h_pos / (h_pos_raw + 1e-8)).mean().item()

                self.training_history['loss'].append(total_loss.item())
                self.training_history['g_pos'].append(g_pos.mean().item())
                self.training_history['g_neg'].append(g_neg.mean().item())
                self.training_history['reconstruction_loss'].append(recon_loss.item())
                self.training_history['sparsity'].append(sparsity)
                self.training_history['surprise_ratio'].append(surprise_ratio)

                if verbose and iterator is not None:
                    iterator.set_postfix({
                        'loss': total_loss.item(),
                        'g+': g_pos.mean().item(),
                        'g-': g_neg.mean().item(),
                        'recon': recon_loss.item()
                    })

        # Return detached outputs for next layer
        with torch.no_grad():
            h_pos_out, _ = self.forward(x_pos, apply_surprise_mask=True)
            h_neg_out, _ = self.forward(x_neg, apply_surprise_mask=True)

        return h_pos_out.detach(), h_neg_out.detach()


# ============================================================
# PCL-FF Network Implementation
# ============================================================

class PCLFFNetwork(nn.Module):
    """
    Predictive Coding Light Forward-Forward Network.

    A multi-layer network where each layer:
    1. Learns to encode its input to a hidden representation
    2. Learns to predict its input from that representation
    3. Suppresses predictable (redundant) information
    4. Transmits surprising (novel) information to next layer

    This should lead to hierarchical representations that are:
    - Increasingly abstract (each layer removes predictable details)
    - More information-dense (redundancy removed)
    - Better for transfer (captures essential structure, not superficial patterns)
    """

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03,
                 alpha: float = 0.5, sparsity_weight: float = 0.1,
                 surprise_scale: float = 1.0):
        """
        Args:
            dims: List of layer dimensions [input_dim, hidden1, hidden2, ...]
            threshold: Goodness threshold for FF training
            lr: Learning rate
            alpha: Weight for reconstruction loss
            sparsity_weight: Weight for sparsity regularization
            surprise_scale: How much to apply surprise masking
        """
        super().__init__()
        self.layers = nn.ModuleList()

        for d in range(len(dims) - 1):
            self.layers.append(PCLFFLayer(
                dims[d], dims[d + 1],
                threshold=threshold,
                lr=lr,
                alpha=alpha,
                sparsity_weight=sparsity_weight,
                surprise_scale=surprise_scale
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.get_hidden(x)
        return x

    def forward_with_predictions(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass returning all hidden states and predictions."""
        results = []
        for layer in self.layers:
            h, x_pred = layer.forward(x, apply_surprise_mask=True)
            results.append((h, x_pred))
            x = h
        return results

    def get_layer_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get activations from all layers."""
        activations = []
        for layer in self.layers:
            x = layer.get_hidden(x)
            activations.append(x)
        return activations

    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 500, verbose: bool = True):
        """
        Greedy layer-by-layer training.

        Each layer is trained to convergence before moving to the next.
        This allows proper hierarchical feature learning.
        """
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\n{"="*60}')
                print(f'Training PCL-FF Layer {i}')
                print(f'{"="*60}')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg, epochs_per_layer, verbose)

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """
        Predict by trying all labels and picking highest goodness.

        For each candidate label, embed it in the input and compute
        total goodness across all layers. Return label with highest goodness.
        """
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(num_classes):
            # Create input with this candidate label
            h = overlay_y_on_x(x, torch.full((batch_size,), label, device=x.device))

            # Compute goodness at each layer
            goodness = []
            for layer in self.layers:
                h = layer.get_hidden(h)
                goodness.append(layer.simple_goodness(h))

            # Sum goodness across layers
            total_goodness = sum(goodness)
            goodness_per_label.append(total_goodness.unsqueeze(1))

        # Return label with highest goodness
        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor,
                     num_classes: int = 10) -> float:
        """Compute classification accuracy."""
        predictions = self.predict(x, num_classes)
        return (predictions == y).float().mean().item()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features from last layer (for transfer learning).

        We use label=0 for consistent feature extraction.
        """
        with torch.no_grad():
            batch_size = x.shape[0]
            h = overlay_y_on_x(x, torch.zeros(batch_size, dtype=torch.long, device=x.device))
            for layer in self.layers:
                h = layer.get_hidden(h)
            return h

    def get_training_history(self) -> Dict[str, List]:
        """Get training history from all layers."""
        history = {}
        for i, layer in enumerate(self.layers):
            for key, values in layer.training_history.items():
                history[f'layer{i}_{key}'] = values
        return history

    def analyze_representations(self, x: torch.Tensor, y: torch.Tensor) -> Dict:
        """
        Analyze the learned representations.

        Returns metrics about sparsity, information content, etc.
        """
        with torch.no_grad():
            batch_size = x.shape[0]
            x_labeled = overlay_y_on_x(x, y)

            analysis = {
                'layer_sparsity': [],
                'layer_activation_mean': [],
                'layer_activation_std': [],
                'layer_dead_neurons': [],
                'reconstruction_errors': []
            }

            h = x_labeled
            for i, layer in enumerate(self.layers):
                h_raw, x_pred = layer.forward(h, apply_surprise_mask=False)
                h_masked = layer.get_hidden(h)

                # Sparsity: fraction of activations near zero
                sparsity = (h_masked.abs() < 0.1).float().mean().item()
                analysis['layer_sparsity'].append(sparsity)

                # Activation statistics
                analysis['layer_activation_mean'].append(h_masked.mean().item())
                analysis['layer_activation_std'].append(h_masked.std().item())

                # Dead neurons: neurons that never activate
                dead_neurons = (h_masked.abs().max(dim=0)[0] < 0.01).float().mean().item()
                analysis['layer_dead_neurons'].append(dead_neurons)

                # Reconstruction error
                h_normed = h / (h.norm(2, dim=1, keepdim=True) + 1e-4)
                recon_error = F.mse_loss(x_pred, h_normed).item()
                analysis['reconstruction_errors'].append(recon_error)

                h = h_masked

            return analysis


# ============================================================
# Standard FF Network (for comparison)
# ============================================================

class StandardFFLayer(nn.Module):
    """Standard Forward-Forward Layer for comparison."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.opt = Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        return h.pow(2).mean(dim=1)

    def train_layer(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                    num_epochs: int = 500, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        iterator = tqdm(range(num_epochs), desc="Training FF layer") if verbose else range(num_epochs)

        for _ in iterator:
            h_pos = self.forward(x_pos)
            h_neg = self.forward(x_neg)

            g_pos = self.goodness(h_pos)
            g_neg = self.goodness(h_neg)

            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold
            ]))).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if verbose and iterator is not None:
                iterator.set_postfix({
                    'loss': loss.item(),
                    'g+': g_pos.mean().item(),
                    'g-': g_neg.mean().item()
                })

        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class StandardFFNetwork(nn.Module):
    """Standard Forward-Forward Network for comparison."""

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(StandardFFLayer(dims[d], dims[d + 1], threshold, lr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     epochs_per_layer: int = 500, verbose: bool = True):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\n{"="*60}')
                print(f'Training Standard FF Layer {i}')
                print(f'{"="*60}')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg, epochs_per_layer, verbose)

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(num_classes):
            h = overlay_y_on_x(x, torch.full((batch_size,), label, device=x.device))
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
        predictions = self.predict(x, num_classes)
        return (predictions == y).float().mean().item()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_size = x.shape[0]
            h = overlay_y_on_x(x, torch.zeros(batch_size, dtype=torch.long, device=x.device))
            for layer in self.layers:
                h = layer(h)
            return h


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":
    print("Testing PCL-FF implementation...")

    device = get_device()
    print(f"Device: {device}")

    # Create dummy data
    batch_size = 1000
    x = torch.randn(batch_size, 784).to(device)
    y = torch.randint(0, 10, (batch_size,)).to(device)

    # Create positive and negative samples
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(batch_size)
    x_neg = overlay_y_on_x(x, y[rnd])

    # Test PCL-FF network
    print("\nCreating PCL-FF Network [784, 500, 500]...")
    pcl_net = PCLFFNetwork([784, 500, 500], threshold=2.0, lr=0.03, alpha=0.5).to(device)

    # Quick training test (just a few epochs)
    print("\nQuick training test (10 epochs per layer)...")
    pcl_net.train_greedy(x_pos, x_neg, epochs_per_layer=10, verbose=True)

    # Test prediction
    print("\nTesting prediction...")
    acc = pcl_net.get_accuracy(x, y)
    print(f"Accuracy on random data: {acc*100:.2f}%")

    # Test analysis
    print("\nAnalyzing representations...")
    analysis = pcl_net.analyze_representations(x, y)
    for key, values in analysis.items():
        print(f"  {key}: {values}")

    print("\nPCL-FF implementation test complete!")
