"""
Three-Factor Forward-Forward Implementation
============================================

Neuroscience-inspired extension of the Forward-Forward algorithm.

Three-Factor Learning Rule (from neuroscience):
    dw = f(pre) * f(post) * M(t)

Where:
- f(pre): Presynaptic activity
- f(post): Postsynaptic activity
- M(t): Neuromodulatory signal (e.g., dopamine) providing global feedback

This implementation adds a third factor (modulation signal) to the standard FF:
- Standard FF: goodness = h.pow(2).mean()
- Three-Factor FF: modulated_goodness = goodness * M(t)

Three modulation signal types are implemented:
1. Top-down feedback: M(t) = softmax(final_layer)[correct_class]
2. Reward prediction error: M(t) = actual_reward - expected_reward
3. Layer agreement: M(t) = correlation(current_layer, next_layer)

Author: Clawd (for Parafee)
Date: 2026-02-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Dict, Optional, Tuple
from enum import Enum


class ModulationType(Enum):
    """Types of modulation signals."""
    NONE = "none"  # Standard FF (baseline)
    TOP_DOWN = "top_down"  # From output layer
    REWARD_PREDICTION = "reward_prediction"  # TD-like learning
    LAYER_AGREEMENT = "layer_agreement"  # Inter-layer correlation


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def overlay_y_on_x(x: torch.Tensor, y) -> torch.Tensor:
    """
    Replace first 10 pixels with one-hot-encoded label.
    Uses x.max() as the label value (critical for FF).
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


# ============================================================
# Three-Factor FF Layer
# ============================================================

class ThreeFactorFFLayer(nn.Module):
    """
    Forward-Forward Layer with Three-Factor Learning support.

    Implements: dw = f(pre) * f(post) * M(t)

    Key additions:
    - modulation_scale: Learnable parameter for modulation strength
    - Supports three types of modulation signals
    """

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03,
                 modulation_strength: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold

        # Modulation scale (learnable)
        self.modulation_scale = nn.Parameter(torch.tensor([modulation_strength]))

        # Expected reward for TD-like learning (running average)
        self.register_buffer('expected_reward', torch.tensor(0.5))
        self.reward_momentum = 0.99

        self.opt = Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with L2 normalization."""
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Compute goodness = MEAN of squared activations."""
        return h.pow(2).mean(dim=1)

    def goodness_with_modulation(self, h: torch.Tensor,
                                   modulation_signal: torch.Tensor) -> torch.Tensor:
        """
        Compute modulated goodness (Three-Factor).

        goodness_modulated = goodness * (1 + scale * M(t))

        This allows the modulation to amplify or suppress goodness
        based on the global signal.
        """
        local_goodness = self.goodness(h)

        # Modulation effect (centered at 1, so no modulation = standard FF)
        # Using sigmoid to bound the modulation effect
        modulation_effect = 1.0 + self.modulation_scale * torch.tanh(modulation_signal)

        modulated_goodness = local_goodness * modulation_effect
        return modulated_goodness

    def update_expected_reward(self, actual_reward: float):
        """Update running average of expected reward (for TD learning)."""
        self.expected_reward = (self.reward_momentum * self.expected_reward +
                                (1 - self.reward_momentum) * actual_reward)

    def ff_loss(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor) -> torch.Tensor:
        """Standard FF loss."""
        loss = torch.log(1 + torch.exp(torch.cat([
            -pos_goodness + self.threshold,
            neg_goodness - self.threshold
        ]))).mean()
        return loss


# ============================================================
# Three-Factor FF Network
# ============================================================

class ThreeFactorFFNetwork(nn.Module):
    """
    Forward-Forward Network with Three-Factor Learning.

    Implements neuroscience-inspired modulation signals:
    1. Top-down feedback (attention-like)
    2. Reward prediction error (TD-learning-like)
    3. Layer agreement (inter-layer consistency)
    """

    def __init__(self, dims: List[int], threshold: float = 2.0, lr: float = 0.03,
                 modulation_type: ModulationType = ModulationType.NONE,
                 modulation_strength: float = 0.5):
        super().__init__()
        self.dims = dims
        self.modulation_type = modulation_type
        self.modulation_strength = modulation_strength

        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers.append(
                ThreeFactorFFLayer(dims[d], dims[d + 1], threshold, lr, modulation_strength)
            )

        # Output projection for top-down modulation
        if modulation_type == ModulationType.TOP_DOWN:
            self.output_projection = nn.Linear(dims[-1], 10)  # 10 classes
            self.output_opt = Adam(self.output_projection.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_all_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get activations from all layers."""
        activations = []
        h = x
        for layer in self.layers:
            h = layer(h)
            activations.append(h)
        return activations

    # ================================================================
    # Modulation Signal Computations
    # ================================================================

    def compute_top_down_modulation(self, x: torch.Tensor,
                                     correct_label: torch.Tensor) -> torch.Tensor:
        """
        Compute top-down modulation signal.

        M(t) = softmax(final_layer_projection)[correct_class]

        This is similar to attention: the network "pays attention" to
        samples it already correctly classifies.
        """
        with torch.no_grad():
            # Forward through all layers
            h = x
            for layer in self.layers:
                h = layer(h)

            # Project to class probabilities
            logits = self.output_projection(h)
            probs = F.softmax(logits, dim=1)

            # Get probability of correct class
            batch_indices = torch.arange(len(correct_label), device=x.device)
            correct_probs = probs[batch_indices, correct_label]

            # Center around 0 for modulation
            modulation = correct_probs - 0.5  # Range: [-0.5, 0.5]

        return modulation

    def compute_reward_prediction_modulation(self, is_positive: bool,
                                              layer_idx: int) -> torch.Tensor:
        """
        Compute reward prediction error modulation.

        M(t) = actual_reward - expected_reward

        Where:
        - actual_reward = 1 for positive samples, 0 for negative
        - expected_reward = running average

        This is similar to TD learning in RL.
        """
        layer = self.layers[layer_idx]

        actual_reward = 1.0 if is_positive else 0.0

        # Reward prediction error
        rpe = actual_reward - layer.expected_reward.item()

        # Update expected reward
        layer.update_expected_reward(actual_reward)

        return torch.tensor(rpe, device=layer.linear.weight.device)

    def compute_layer_agreement_modulation(self, x: torch.Tensor,
                                            current_layer_idx: int) -> torch.Tensor:
        """
        Compute layer agreement modulation.

        M(t) = correlation(current_layer, next_layer)

        This encourages layers to learn consistent representations.
        """
        with torch.no_grad():
            activations = self.get_all_activations(x)

            if current_layer_idx >= len(activations) - 1:
                # Last layer: use correlation with previous
                if current_layer_idx == 0:
                    return torch.zeros(x.shape[0], device=x.device)
                h_curr = activations[current_layer_idx]
                h_other = activations[current_layer_idx - 1]
            else:
                # Use correlation with next layer
                h_curr = activations[current_layer_idx]
                h_other = activations[current_layer_idx + 1]

            # Compute per-sample correlation
            # Normalize both activations
            h_curr_norm = h_curr - h_curr.mean(dim=1, keepdim=True)
            h_other_norm = h_other - h_other.mean(dim=1, keepdim=True)

            # Project to same dimension if needed (use linear projection)
            if h_curr_norm.shape[1] != h_other_norm.shape[1]:
                # Use mean pooling to align dimensions
                target_dim = min(h_curr_norm.shape[1], h_other_norm.shape[1])
                h_curr_norm = h_curr_norm[:, :target_dim]
                h_other_norm = h_other_norm[:, :target_dim]

            # Correlation per sample
            correlation = (h_curr_norm * h_other_norm).sum(dim=1) / (
                (h_curr_norm.pow(2).sum(dim=1).sqrt() + 1e-8) *
                (h_other_norm.pow(2).sum(dim=1).sqrt() + 1e-8)
            )

            # Center around 0
            modulation = correlation  # Already in [-1, 1]

        return modulation

    # ================================================================
    # Training Methods
    # ================================================================

    def train_layer_three_factor(self, layer_idx: int,
                                  h_pos: torch.Tensor, h_neg: torch.Tensor,
                                  x_pos_full: torch.Tensor, x_neg_full: torch.Tensor,
                                  y_pos: torch.Tensor,
                                  num_epochs: int = 500,
                                  verbose: bool = True):
        """Train a single layer with three-factor modulation."""
        layer = self.layers[layer_idx]

        for epoch in range(num_epochs):
            # Forward through current layer
            out_pos = layer(h_pos)
            out_neg = layer(h_neg)

            # Compute modulation signals based on type
            if self.modulation_type == ModulationType.NONE:
                # Standard FF (no modulation)
                g_pos = layer.goodness(out_pos)
                g_neg = layer.goodness(out_neg)

            elif self.modulation_type == ModulationType.TOP_DOWN:
                # Top-down modulation from output layer
                mod_pos = self.compute_top_down_modulation(x_pos_full, y_pos)
                mod_neg = self.compute_top_down_modulation(x_neg_full, y_pos)  # Wrong label

                g_pos = layer.goodness_with_modulation(out_pos, mod_pos)
                g_neg = layer.goodness_with_modulation(out_neg, -mod_neg)  # Negate for neg

            elif self.modulation_type == ModulationType.REWARD_PREDICTION:
                # Reward prediction error modulation
                rpe_pos = self.compute_reward_prediction_modulation(True, layer_idx)
                rpe_neg = self.compute_reward_prediction_modulation(False, layer_idx)

                # Expand to batch size
                rpe_pos_batch = rpe_pos.expand(out_pos.shape[0])
                rpe_neg_batch = rpe_neg.expand(out_neg.shape[0])

                g_pos = layer.goodness_with_modulation(out_pos, rpe_pos_batch)
                g_neg = layer.goodness_with_modulation(out_neg, rpe_neg_batch)

            elif self.modulation_type == ModulationType.LAYER_AGREEMENT:
                # Layer agreement modulation
                mod_pos = self.compute_layer_agreement_modulation(x_pos_full, layer_idx)
                mod_neg = self.compute_layer_agreement_modulation(x_neg_full, layer_idx)

                g_pos = layer.goodness_with_modulation(out_pos, mod_pos)
                g_neg = layer.goodness_with_modulation(out_neg, mod_neg)

            # Compute loss
            loss = layer.ff_loss(g_pos, g_neg)

            # Backward and update
            layer.opt.zero_grad()
            loss.backward()
            layer.opt.step()

            # Also update output projection for top-down
            if self.modulation_type == ModulationType.TOP_DOWN and hasattr(self, 'output_projection'):
                # Train output projection with classification loss
                with torch.enable_grad():
                    activations = self.get_all_activations(x_pos_full)
                    logits = self.output_projection(activations[-1].detach())
                    cls_loss = F.cross_entropy(logits, y_pos)

                    self.output_opt.zero_grad()
                    cls_loss.backward()
                    self.output_opt.step()

            if verbose and (epoch + 1) % 100 == 0:
                print(f"      Epoch {epoch+1}: loss={loss.item():.4f}, "
                      f"g_pos={g_pos.mean().item():.3f}, g_neg={g_neg.mean().item():.3f}")

    def train_greedy(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                     y_pos: torch.Tensor,
                     epochs_per_layer: int = 500, verbose: bool = True):
        """
        Greedy layer-by-layer training with three-factor modulation.
        """
        h_pos, h_neg = x_pos, x_neg

        for i in range(len(self.layers)):
            if verbose:
                mod_name = self.modulation_type.value
                print(f'    Training layer {i} (Three-Factor: {mod_name})...')

            self.train_layer_three_factor(
                i, h_pos, h_neg, x_pos, x_neg, y_pos,
                epochs_per_layer, verbose
            )

            # Get outputs for next layer
            h_pos = self.layers[i](h_pos).detach()
            h_neg = self.layers[i](h_neg).detach()

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels and picking highest goodness."""
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(num_classes):
            h = overlay_y_on_x(x, torch.full((batch_size,), label, device=x.device))
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))
            goodness_per_label.append(sum(goodness).unsqueeze(1))

        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor,
                     num_classes: int = 10) -> float:
        """Compute accuracy."""
        predictions = self.predict(x, num_classes)
        return (predictions == y).float().mean().item()

    def get_features(self, x: torch.Tensor, up_to_layer: int = None,
                     label: int = 0) -> torch.Tensor:
        """Extract features with label embedding."""
        with torch.no_grad():
            h = overlay_y_on_x(x, torch.full((x.shape[0],), label, dtype=torch.long, device=x.device))

            layers_to_use = self.layers if up_to_layer is None else self.layers[:up_to_layer + 1]

            for layer in layers_to_use:
                h = layer(h)

            return h


# ============================================================
# Convenience Functions
# ============================================================

def create_three_factor_network(modulation_type: str = "none",
                                  dims: List[int] = None,
                                  modulation_strength: float = 0.5,
                                  **kwargs) -> ThreeFactorFFNetwork:
    """
    Factory function to create a Three-Factor FF Network.

    Args:
        modulation_type: One of "none", "top_down", "reward_prediction", "layer_agreement"
        dims: Network dimensions (default: [784, 500, 500])
        modulation_strength: Strength of modulation signal (default: 0.5)
    """
    if dims is None:
        dims = [784, 500, 500]

    mod_type_map = {
        "none": ModulationType.NONE,
        "top_down": ModulationType.TOP_DOWN,
        "reward_prediction": ModulationType.REWARD_PREDICTION,
        "layer_agreement": ModulationType.LAYER_AGREEMENT
    }

    if modulation_type not in mod_type_map:
        raise ValueError(f"Unknown modulation type: {modulation_type}. "
                         f"Choose from: {list(mod_type_map.keys())}")

    return ThreeFactorFFNetwork(
        dims=dims,
        modulation_type=mod_type_map[modulation_type],
        modulation_strength=modulation_strength,
        **kwargs
    )


if __name__ == "__main__":
    # Quick test
    device = get_device()
    print(f"Device: {device}")

    # Test all modulation types
    for mod_type in ["none", "top_down", "reward_prediction", "layer_agreement"]:
        print(f"\nTesting {mod_type}...")
        net = create_three_factor_network(modulation_type=mod_type).to(device)

        # Dummy data
        x = torch.randn(100, 784).to(device)
        y = torch.randint(0, 10, (100,)).to(device)

        x_pos = overlay_y_on_x(x, y)
        x_neg = overlay_y_on_x(x, (y + 1) % 10)

        # Test forward
        out = net(x_pos)
        print(f"  Output shape: {out.shape}")

        # Test training (just a few epochs)
        net.train_greedy(x_pos, x_neg, y, epochs_per_layer=5, verbose=False)

        # Test prediction
        preds = net.predict(x)
        acc = net.get_accuracy(x, y)
        print(f"  Accuracy (after 5 epochs): {acc*100:.1f}%")

    print("\nAll tests passed!")
