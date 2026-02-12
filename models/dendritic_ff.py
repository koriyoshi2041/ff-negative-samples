"""
Dendritic Forward-Forward Implementation
=========================================

Based on: Wright et al. (Science 2025) - Distinct learning rules in different dendritic compartments

Neuroscience Background:
------------------------
Single neurons have two distinct dendritic compartments with different learning rules:

1. Basal Dendrites:
   - Receive feedforward (bottom-up) inputs
   - Learning is synchronized with postsynaptic firing (Hebbian)
   - Goodness based on: basal activity × global spike signal
   - Standard FF's Hebbian learning applies here

2. Apical Dendrites:
   - Receive top-down contextual signals
   - Learning based on local co-activity prediction
   - Goodness based on: local co-activity (neighboring unit correlations)
   - Forms functional clusters through coordinated activation

Implementation:
---------------
- DendriticFFLayer: Each layer has basal and apical "compartments"
- Basal: Processes feedforward input from previous layer
- Apical: Processes top-down context (from higher layers or labels)
- Output: basal * sigmoid(apical) - apical gates basal activity
- Combined goodness: basal_goodness + alpha * apical_goodness

Author: Clawd (for Parafee)
Date: 2026-02-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Dict, Tuple, Optional
import math


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Embed label in first 10 pixels.
    CRITICAL: Use x.max() as the label value, not 1.0!
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


# ============================================================
# Dendritic FF Layer
# ============================================================

class DendriticFFLayer(nn.Module):
    """
    Forward-Forward Layer with Dendritic Compartments.

    Implements biologically-inspired dendritic computation:
    - Basal compartment: Receives feedforward input, Hebbian learning
    - Apical compartment: Receives top-down context, local co-activity learning

    Args:
        in_dim: Input dimension (feedforward from previous layer)
        out_dim: Output dimension (number of neurons)
        context_dim: Context dimension (top-down signal)
        threshold: Goodness threshold for FF learning
        alpha: Weight for apical goodness contribution (default: 0.3)
        coactivity_radius: Radius for local co-activity computation (default: 5)
        lr: Learning rate
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        context_dim: int,
        threshold: float = 2.0,
        alpha: float = 0.3,
        coactivity_radius: int = 5,
        lr: float = 0.03
    ):
        super().__init__()

        # Basal compartment: feedforward weights
        self.basal_weights = nn.Linear(in_dim, out_dim)

        # Apical compartment: context/top-down weights
        self.apical_weights = nn.Linear(context_dim, out_dim)

        # Activation
        self.relu = nn.ReLU()

        # Parameters
        self.threshold = threshold
        self.alpha = alpha
        self.coactivity_radius = coactivity_radius
        self.out_dim = out_dim

        # Optimizer for this layer
        self.opt = Adam(self.parameters(), lr=lr)

        # Pre-compute neighbor indices for local co-activity
        self._setup_neighbor_indices()

    def _setup_neighbor_indices(self):
        """Pre-compute neighbor indices for efficient local co-activity computation."""
        # For each neuron, find its neighbors within radius
        # Using circular topology for boundary handling
        indices = []
        for i in range(self.out_dim):
            neighbors = []
            for offset in range(-self.coactivity_radius, self.coactivity_radius + 1):
                if offset != 0:  # Exclude self
                    neighbor_idx = (i + offset) % self.out_dim
                    neighbors.append(neighbor_idx)
            indices.append(neighbors)

        # Store as tensor for efficient gathering
        self.register_buffer(
            'neighbor_indices',
            torch.tensor(indices, dtype=torch.long)  # [out_dim, 2*radius]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with dendritic computation.

        Args:
            x: Feedforward input [batch, in_dim]
            context: Top-down context [batch, context_dim]

        Returns:
            Output activations [batch, out_dim]
        """
        # Normalize feedforward input (L2 normalization per sample)
        x_norm = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)

        # Basal activity: feedforward processing
        basal = self.basal_weights(x_norm)

        if context is not None:
            # Normalize context
            context_norm = context / (context.norm(2, dim=1, keepdim=True) + 1e-4)

            # Apical activity: top-down modulation
            apical = self.apical_weights(context_norm)

            # Apical gates basal: multiplicative modulation
            # sigmoid(apical) gives gating signal between 0 and 1
            gate = torch.sigmoid(apical)
            output = basal * gate
        else:
            output = basal

        return self.relu(output)

    def goodness_basal(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute basal goodness - standard Hebbian (mean squared activations).

        This corresponds to the Hebbian learning in basal dendrites that
        synchronizes with postsynaptic firing.
        """
        return h.pow(2).mean(dim=1)

    def goodness_apical(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute apical goodness - local co-activity.

        This corresponds to the local co-activity based learning in apical
        dendrites that forms functional clusters.

        Implementation: For each neuron, compute correlation with neighbors.
        High co-activity = high goodness.
        """
        batch_size = h.shape[0]

        # Gather neighbor activations: [batch, out_dim, 2*radius]
        h_expanded = h.unsqueeze(2).expand(-1, -1, 2 * self.coactivity_radius)
        neighbor_acts = torch.gather(
            h.unsqueeze(1).expand(-1, self.out_dim, -1),
            dim=2,
            index=self.neighbor_indices.unsqueeze(0).expand(batch_size, -1, -1)
        )

        # Local co-activity: product of neuron with its neighbors
        # This measures coordinated activation (functional clustering)
        coactivity = h.unsqueeze(2) * neighbor_acts  # [batch, out_dim, 2*radius]

        # Average co-activity per neuron, then mean across neurons
        local_coact = coactivity.mean(dim=2)  # [batch, out_dim]

        return local_coact.mean(dim=1)  # [batch]

    def goodness(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined goodness from both dendritic compartments.

        Returns:
            total_goodness: Combined goodness for FF loss
            basal_goodness: Hebbian component
            apical_goodness: Local co-activity component
        """
        basal_g = self.goodness_basal(h)
        apical_g = self.goodness_apical(h)

        total_g = basal_g + self.alpha * apical_g

        return total_g, basal_g, apical_g

    def train_layer(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        context_pos: Optional[torch.Tensor] = None,
        context_neg: Optional[torch.Tensor] = None,
        num_epochs: int = 1000,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train this layer with dendritic learning.

        Args:
            x_pos: Positive feedforward inputs
            x_neg: Negative feedforward inputs
            context_pos: Positive context (optional)
            context_neg: Negative context (optional)
            num_epochs: Training epochs
            verbose: Print progress

        Returns:
            Tuple of (positive outputs, negative outputs) for next layer
        """
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(range(num_epochs), desc="Training layer")
        else:
            iterator = range(num_epochs)

        for epoch in iterator:
            # Forward pass
            h_pos = self.forward(x_pos, context_pos)
            h_neg = self.forward(x_neg, context_neg)

            # Compute goodness (combined)
            g_pos, basal_pos, apical_pos = self.goodness(h_pos)
            g_neg, basal_neg, apical_neg = self.goodness(h_neg)

            # FF loss: push positive above threshold, negative below
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold
            ]))).mean()

            # Backward (local to this layer only!)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if verbose and (epoch + 1) % 100 == 0:
                iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'g_pos': f'{g_pos.mean().item():.3f}',
                    'g_neg': f'{g_neg.mean().item():.3f}',
                    'basal': f'{basal_pos.mean().item():.3f}',
                    'apical': f'{apical_pos.mean().item():.3f}'
                })

        # Return detached outputs for next layer
        return self.forward(x_pos, context_pos).detach(), self.forward(x_neg, context_neg).detach()


# ============================================================
# Dendritic FF Network
# ============================================================

class DendriticFFNetwork(nn.Module):
    """
    Multi-layer Forward-Forward Network with Dendritic Compartments.

    Architecture:
    - Each layer has basal (feedforward) and apical (context) inputs
    - Context can be: label embedding, top-down from higher layers, or both
    - Supports multiple context modes for experimentation

    Args:
        dims: Layer dimensions [input, hidden1, hidden2, ...]
        context_dims: Context dimensions for each layer (or single value for all)
        threshold: Goodness threshold
        alpha: Apical goodness weight
        coactivity_radius: Neighbor radius for local co-activity
        context_mode: 'label' (embed label), 'topdown' (from higher layers), 'both'
        lr: Learning rate
    """

    def __init__(
        self,
        dims: List[int],
        context_dims: Optional[List[int]] = None,
        threshold: float = 2.0,
        alpha: float = 0.3,
        coactivity_radius: int = 5,
        context_mode: str = 'label',  # 'label', 'topdown', 'both'
        lr: float = 0.03
    ):
        super().__init__()

        self.dims = dims
        self.num_layers = len(dims) - 1
        self.context_mode = context_mode
        self.threshold = threshold
        self.alpha = alpha

        # Set up context dimensions
        if context_dims is None:
            # Default: label embedding (10 classes) for each layer
            context_dims = [10] * self.num_layers
        elif isinstance(context_dims, int):
            context_dims = [context_dims] * self.num_layers

        self.context_dims = context_dims

        # Create layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(DendriticFFLayer(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                context_dim=context_dims[i],
                threshold=threshold,
                alpha=alpha,
                coactivity_radius=coactivity_radius,
                lr=lr
            ))

        # Context encoder for label embedding
        if context_mode in ['label', 'both']:
            self.label_encoder = nn.Embedding(10, context_dims[0])
            # Initialize with orthogonal vectors for better separation
            nn.init.orthogonal_(self.label_encoder.weight)

    def _get_context(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        layer_idx: int = 0
    ) -> Optional[torch.Tensor]:
        """Get context for a specific layer based on context_mode."""

        if self.context_mode == 'label' and y is not None:
            # Use label embedding as context
            return self.label_encoder(y)

        elif self.context_mode == 'topdown':
            # Use activations from higher layers (requires forward pass first)
            # This is more complex and requires two-phase training
            return None  # Simplified for now

        elif self.context_mode == 'both' and y is not None:
            # Combine label and top-down
            return self.label_encoder(y)  # Simplified

        return None

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through all layers."""
        h = x
        for i, layer in enumerate(self.layers):
            context = self._get_context(h, y, i)
            h = layer(h, context)
        return h

    def train_greedy(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
        epochs_per_layer: int = 500,
        verbose: bool = True
    ):
        """
        Greedy layer-by-layer training with dendritic learning.

        Unlike standard FF, we also pass context (labels) to each layer.
        """
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\nTraining layer {i}...')

            # Get context for this layer
            context_pos = self._get_context(h_pos, y_pos, i)
            context_neg = self._get_context(h_neg, y_neg, i)

            # Train layer
            h_pos, h_neg = layer.train_layer(
                h_pos, h_neg,
                context_pos, context_neg,
                epochs_per_layer, verbose
            )

    def predict(
        self,
        x: torch.Tensor,
        num_classes: int = 10
    ) -> torch.Tensor:
        """
        Predict by trying all labels and picking highest goodness.

        For dendritic FF, we also use the label as context for each layer.
        """
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(num_classes):
            # Label tensor for context
            y = torch.full((batch_size,), label, device=x.device, dtype=torch.long)

            # Also embed label in input for compatibility with standard FF
            h = overlay_y_on_x(x, y)

            # Compute goodness at each layer
            goodness = []
            for i, layer in enumerate(self.layers):
                context = self._get_context(h, y, i)
                h = layer(h, context)
                total_g, _, _ = layer.goodness(h)
                goodness.append(total_g)

            # Sum goodness across layers
            total_goodness = sum(goodness)
            goodness_per_label.append(total_goodness.unsqueeze(1))

        # Return label with highest goodness
        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)

    def get_accuracy(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_classes: int = 10
    ) -> float:
        """Compute accuracy."""
        predictions = self.predict(x, num_classes)
        return (predictions == y).float().mean().item()

    def get_features(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract features from final layer.

        For transfer learning, use neutral context (label=0 or average).
        """
        with torch.no_grad():
            batch_size = x.shape[0]

            # Use label=0 for consistent feature extraction
            if y is None:
                y = torch.zeros(batch_size, dtype=torch.long, device=x.device)

            # Embed label in input
            h = overlay_y_on_x(x, y)

            # Forward through all layers
            for i, layer in enumerate(self.layers):
                context = self._get_context(h, y, i)
                h = layer(h, context)

            return h

    def get_layer_statistics(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, List[float]]:
        """Get detailed statistics per layer for analysis."""
        stats = {
            'basal_goodness': [],
            'apical_goodness': [],
            'total_goodness': [],
            'activation_mean': [],
            'activation_std': []
        }

        with torch.no_grad():
            h = overlay_y_on_x(x, y)

            for i, layer in enumerate(self.layers):
                context = self._get_context(h, y, i)
                h = layer(h, context)

                total_g, basal_g, apical_g = layer.goodness(h)

                stats['basal_goodness'].append(basal_g.mean().item())
                stats['apical_goodness'].append(apical_g.mean().item())
                stats['total_goodness'].append(total_g.mean().item())
                stats['activation_mean'].append(h.mean().item())
                stats['activation_std'].append(h.std().item())

        return stats


# ============================================================
# Standard FF Network (for comparison)
# ============================================================

class StandardFFLayer(nn.Module):
    """Standard Forward-Forward Layer (no dendritic computation)."""

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
                    num_epochs: int = 1000, verbose: bool = True):
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(range(num_epochs), desc="Training layer")
        else:
            iterator = range(num_epochs)

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
                     epochs_per_layer: int = 1000, verbose: bool = True):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'\nTraining layer {i}...')
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

def quick_test():
    """Quick test of Dendritic FF implementation."""
    print("="*60)
    print("Quick Test: Dendritic FF Layer")
    print("="*60)

    device = get_device()
    print(f"Device: {device}")

    # Create layer
    layer = DendriticFFLayer(
        in_dim=784,
        out_dim=500,
        context_dim=10,
        threshold=2.0,
        alpha=0.3,
        coactivity_radius=5
    ).to(device)

    # Test data
    batch_size = 100
    x = torch.randn(batch_size, 784, device=device)
    context = torch.randn(batch_size, 10, device=device)

    # Forward pass
    output = layer(x, context)
    print(f"Input shape: {x.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Output shape: {output.shape}")

    # Goodness
    total_g, basal_g, apical_g = layer.goodness(output)
    print(f"\nGoodness:")
    print(f"  Basal (Hebbian): {basal_g.mean().item():.4f}")
    print(f"  Apical (co-activity): {apical_g.mean().item():.4f}")
    print(f"  Total: {total_g.mean().item():.4f}")

    print("\nQuick test passed!")


if __name__ == "__main__":
    quick_test()
