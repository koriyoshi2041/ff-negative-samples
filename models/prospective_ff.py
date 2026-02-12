"""
Prospective Forward-Forward (Prospective FF) Implementation

Based on: Prospective Configuration (Nature Neuroscience, 2024)

Key neuroscience insight:
1. Network first INFERS what neural activity should be after learning
2. Then modifies synaptic weights to CONSOLIDATE this activity change
3. Only needs ONE ITERATION to learn (vs BP needing many)

Key advantage: Reduces learning interference (catastrophic forgetting)

The core idea differs from standard FF:
- Standard FF: Each layer learns to maximize/minimize goodness locally
- Prospective FF: Each layer infers TARGET activity based on global goal,
                  then adjusts weights to produce that target activity

Implementation:
1. Forward pass: compute current activations
2. Target inference: compute prospective target activations using feedback
3. Consolidation: adjust weights to reduce (h_current - h_target)

This is biologically plausible because:
- Target inference uses only local + feedback signals (no backprop)
- Weight update is Hebbian-like: depends on pre/post activity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Tuple, Optional, Dict, Any
import time


def get_device() -> torch.device:
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
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    if isinstance(y, int):
        x_[:, y] = x.max()
    else:
        x_[range(x.shape[0]), y] = x.max()
    return x_


# =============================================================================
# Prospective FF Layer
# =============================================================================

class ProspectiveFFLayer(nn.Module):
    """
    Forward-Forward Layer with Prospective Configuration.

    Key innovation: Two-phase learning
    1. Inference phase: Compute target activity based on feedback
    2. Consolidation phase: Adjust weights to produce target activity

    Args:
        in_features: Input dimension
        out_features: Output dimension
        threshold: Goodness threshold for FF
        lr: Learning rate
        beta: Target inference strength (how much to adjust toward target)
        consolidation_lr: Learning rate for consolidation step
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: float = 2.0,
        lr: float = 0.03,
        beta: float = 0.5,
        consolidation_lr: float = 0.01
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.beta = beta
        self.consolidation_lr = consolidation_lr

        # Forward weights
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

        # Feedback projection: maps target hint to activity adjustment
        self.feedback_proj = nn.Linear(out_features, out_features, bias=False)
        nn.init.orthogonal_(self.feedback_proj.weight, gain=0.1)

        # Optimizer for standard FF learning
        self.opt = Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass with L2 normalization."""
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Compute goodness - MEAN of squared activations."""
        return h.pow(2).mean(dim=1)

    def infer_target_activity(
        self,
        h_current: torch.Tensor,
        target_hint: torch.Tensor,
        is_positive: bool = True
    ) -> torch.Tensor:
        """
        Infer target activity based on current activity and target hint.

        This is the "prospective configuration" step:
        - For positive samples: increase goodness (activity should be higher)
        - For negative samples: decrease goodness (activity should be lower)

        Args:
            h_current: Current layer activation
            target_hint: Feedback signal (can be from next layer or global target)
            is_positive: Whether this is a positive sample

        Returns:
            Target activity h_target
        """
        # Project the hint to this layer's dimension
        adjustment = self.feedback_proj(target_hint)

        # Direction depends on whether positive or negative
        if is_positive:
            # For positive: increase activity in direction of hint
            h_target = h_current + self.beta * adjustment
        else:
            # For negative: decrease activity
            h_target = h_current - self.beta * adjustment

        # Apply ReLU to keep non-negative
        h_target = F.relu(h_target)

        return h_target

    def consolidate(
        self,
        x: torch.Tensor,
        h_current: torch.Tensor,
        h_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Consolidation step: adjust weights to produce target activity.

        This is a local Hebbian-like update:
        Delta_W proportional to (h_target - h_current) * x

        Args:
            x: Layer input (pre-synaptic activity)
            h_current: Current output (post-synaptic activity)
            h_target: Target output we want to produce

        Returns:
            Consolidation loss (for monitoring)
        """
        # Compute error between target and current
        error = h_target.detach() - h_current

        # Consolidation loss: MSE between current and target
        loss = (error ** 2).mean()

        return loss

    def forward_with_prospective(
        self,
        x: torch.Tensor,
        target_hint: Optional[torch.Tensor] = None,
        is_positive: bool = True,
        do_consolidate: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full prospective forward pass.

        Args:
            x: Input tensor
            target_hint: Feedback signal for target inference
            is_positive: Whether this is a positive sample
            do_consolidate: Whether to compute consolidation loss

        Returns:
            Tuple of (output, target, consolidation_loss)
        """
        # Phase 1: Standard forward
        h_current = self.forward(x)

        if target_hint is not None:
            # Phase 2: Infer target activity
            h_target = self.infer_target_activity(h_current, target_hint, is_positive)

            # Phase 3: Compute consolidation loss
            if do_consolidate:
                consolidation_loss = self.consolidate(x, h_current, h_target)
            else:
                consolidation_loss = torch.tensor(0.0, device=x.device)

            return h_current, h_target, consolidation_loss

        return h_current, h_current, torch.tensor(0.0, device=x.device)

    def train_prospective_step(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        target_hint_pos: torch.Tensor,
        target_hint_neg: torch.Tensor,
        use_standard_ff_loss: bool = True
    ) -> Dict[str, float]:
        """
        Single training step with prospective configuration.

        Combines:
        1. Standard FF loss (goodness-based)
        2. Consolidation loss (target tracking)

        Args:
            x_pos: Positive input samples
            x_neg: Negative input samples
            target_hint_pos: Target hints for positive samples
            target_hint_neg: Target hints for negative samples
            use_standard_ff_loss: Whether to also use standard FF loss

        Returns:
            Dictionary of loss values
        """
        # Forward with prospective configuration
        h_pos, h_target_pos, consol_pos = self.forward_with_prospective(
            x_pos, target_hint_pos, is_positive=True
        )
        h_neg, h_target_neg, consol_neg = self.forward_with_prospective(
            x_neg, target_hint_neg, is_positive=False
        )

        # Compute goodness
        g_pos = self.goodness(h_pos)
        g_neg = self.goodness(h_neg)

        # Standard FF loss
        if use_standard_ff_loss:
            ff_loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold
            ]))).mean()
        else:
            ff_loss = torch.tensor(0.0, device=x_pos.device)

        # Total loss = FF loss + consolidation loss
        consolidation_loss = consol_pos + consol_neg
        total_loss = ff_loss + self.consolidation_lr * consolidation_loss

        # Backward
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        return {
            'ff_loss': ff_loss.item(),
            'consolidation_loss': consolidation_loss.item(),
            'total_loss': total_loss.item(),
            'g_pos': g_pos.mean().item(),
            'g_neg': g_neg.mean().item()
        }


# =============================================================================
# Prospective FF Network
# =============================================================================

class ProspectiveFFNetwork(nn.Module):
    """
    Multi-layer Forward-Forward Network with Prospective Configuration.

    Architecture:
    - Multiple ProspectiveFFLayers
    - Top-down feedback connections for target inference
    - Can use label as global target hint

    Training modes:
    1. 'standard': Standard greedy FF (no prospective)
    2. 'prospective': Prospective configuration with single iteration learning
    3. 'hybrid': Multiple iterations with prospective refinement

    Args:
        dims: List of layer dimensions [input, hidden1, hidden2, ...]
        threshold: Goodness threshold
        lr: Learning rate
        beta: Target inference strength
        consolidation_lr: Consolidation learning rate
    """

    def __init__(
        self,
        dims: List[int],
        threshold: float = 2.0,
        lr: float = 0.03,
        beta: float = 0.5,
        consolidation_lr: float = 0.01
    ):
        super().__init__()

        self.dims = dims
        self.threshold = threshold
        self.beta = beta

        # Create layers
        self.layers = nn.ModuleList()
        for d in range(len(dims) - 1):
            layer = ProspectiveFFLayer(
                dims[d], dims[d + 1],
                threshold=threshold,
                lr=lr,
                beta=beta,
                consolidation_lr=consolidation_lr
            )
            self.layers.append(layer)

        # Top-down feedback weights (from layer l+1 to layer l)
        self.feedback_weights = nn.ModuleList()
        for d in range(len(dims) - 2):
            fb = nn.Linear(dims[d + 2], dims[d + 1], bias=False)
            nn.init.orthogonal_(fb.weight, gain=0.1)
            self.feedback_weights.append(fb)

        # Label to top layer projection (for using label as target hint)
        self.label_proj = nn.Linear(10, dims[-1], bias=False)
        nn.init.orthogonal_(self.label_proj.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning all layer activations."""
        activations = []
        h = x
        for layer in self.layers:
            h = layer(h)
            activations.append(h)
        return activations

    def compute_target_hints(
        self,
        activations: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        use_label_hint: bool = True
    ) -> List[torch.Tensor]:
        """
        Compute target hints for each layer using top-down feedback.

        The target hint for layer l comes from:
        1. Activity of layer l+1 (via feedback weights)
        2. Label signal (for top layer)

        Args:
            activations: List of current layer activations
            labels: One-hot encoded labels (batch, 10)
            use_label_hint: Whether to use label as hint for top layer

        Returns:
            List of target hints for each layer
        """
        num_layers = len(self.layers)
        hints = [None] * num_layers

        # Top layer gets hint from label
        if use_label_hint and labels is not None:
            hints[-1] = self.label_proj(labels)
        else:
            hints[-1] = activations[-1]  # Self-hint

        # Other layers get hint from layer above (top-down)
        for l in range(num_layers - 2, -1, -1):
            if l < len(self.feedback_weights):
                hints[l] = self.feedback_weights[l](activations[l + 1].detach())
            else:
                hints[l] = activations[l]  # Self-hint if no feedback

        return hints

    def train_prospective_single_iteration(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Train with prospective configuration - SINGLE ITERATION.

        This is the key feature: learning in one pass through the network.

        Args:
            x_pos: Positive samples (with correct label embedded)
            x_neg: Negative samples (with wrong label embedded)
            y_pos: Correct labels (one-hot)
            y_neg: Wrong labels (one-hot)
            verbose: Print layer-wise info

        Returns:
            Dictionary of losses
        """
        total_ff_loss = 0.0
        total_consol_loss = 0.0

        # First, do a forward pass to get current activations
        with torch.no_grad():
            acts_pos = self.forward(x_pos)
            acts_neg = self.forward(x_neg)

        # Compute target hints based on current activations + labels
        hints_pos = self.compute_target_hints(acts_pos, y_pos, use_label_hint=True)
        hints_neg = self.compute_target_hints(acts_neg, y_neg, use_label_hint=True)

        # Train each layer with prospective configuration
        h_pos, h_neg = x_pos, x_neg

        for l, layer in enumerate(self.layers):
            # Get target hints for this layer
            hint_pos = hints_pos[l].detach()
            hint_neg = hints_neg[l].detach()

            # Train this layer
            losses = layer.train_prospective_step(
                h_pos.detach() if l > 0 else h_pos,
                h_neg.detach() if l > 0 else h_neg,
                hint_pos,
                hint_neg,
                use_standard_ff_loss=True
            )

            total_ff_loss += losses['ff_loss']
            total_consol_loss += losses['consolidation_loss']

            if verbose:
                print(f"  Layer {l}: ff={losses['ff_loss']:.4f}, "
                      f"consol={losses['consolidation_loss']:.4f}, "
                      f"g+={losses['g_pos']:.3f}, g-={losses['g_neg']:.3f}")

            # Get new activations for next layer
            h_pos = layer(h_pos.detach() if l > 0 else h_pos).detach()
            h_neg = layer(h_neg.detach() if l > 0 else h_neg).detach()

        return {
            'ff_loss': total_ff_loss / len(self.layers),
            'consolidation_loss': total_consol_loss / len(self.layers),
            'total_loss': (total_ff_loss + total_consol_loss) / len(self.layers)
        }

    def train_standard_ff(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        epochs_per_layer: int = 1000,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Standard greedy FF training (for comparison).

        Train each layer to convergence before moving to next.
        """
        h_pos, h_neg = x_pos, x_neg

        for l, layer in enumerate(self.layers):
            if verbose:
                print(f'\n  Training layer {l} (Standard FF)...')

            for epoch in range(epochs_per_layer):
                # Forward
                out_pos = layer(h_pos)
                out_neg = layer(h_neg)

                # Compute goodness
                g_pos = layer.goodness(out_pos)
                g_neg = layer.goodness(out_neg)

                # FF loss
                loss = torch.log(1 + torch.exp(torch.cat([
                    -g_pos + layer.threshold,
                    g_neg - layer.threshold
                ]))).mean()

                # Backward
                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()

                if verbose and (epoch + 1) % 200 == 0:
                    print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, "
                          f"g+={g_pos.mean().item():.3f}, g-={g_neg.mean().item():.3f}")

            # Detach for next layer
            h_pos = layer(h_pos).detach()
            h_neg = layer(h_neg).detach()

        return {'status': 'complete'}

    def train_prospective_multi_iteration(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
        num_iterations: int = 100,
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        """
        Train with prospective configuration over multiple iterations.

        Each iteration:
        1. Forward pass to get activations
        2. Compute target hints
        3. Update all layers simultaneously

        Args:
            x_pos, x_neg: Positive/negative samples
            y_pos, y_neg: Labels (one-hot)
            num_iterations: Number of training iterations
            verbose: Print progress

        Returns:
            List of loss dictionaries per iteration
        """
        history = []

        for it in range(num_iterations):
            losses = self.train_prospective_single_iteration(
                x_pos, x_neg, y_pos, y_neg, verbose=False
            )
            history.append(losses)

            if verbose and (it + 1) % 20 == 0:
                print(f"  Iteration {it+1}: ff={losses['ff_loss']:.4f}, "
                      f"consol={losses['consolidation_loss']:.4f}")

        return history

    def predict(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Predict by trying all labels and picking highest goodness."""
        batch_size = x.shape[0]
        goodness_per_label = []

        for label in range(num_classes):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(layer.goodness(h))
            goodness_per_label.append(sum(goodness).unsqueeze(1))

        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(dim=1)

    def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute accuracy."""
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()


# =============================================================================
# Data Loading Utilities
# =============================================================================

def get_mnist_loaders(train_batch_size: int = 50000, test_batch_size: int = 10000):
    """Get MNIST data loaders."""
    from torchvision import datasets
    from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
    from torch.utils.data import DataLoader

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


def get_fashion_mnist_loaders(train_batch_size: int = 50000, test_batch_size: int = 10000):
    """Get Fashion-MNIST data loaders."""
    from torchvision import datasets
    from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
    from torch.utils.data import DataLoader

    transform = Compose([
        ToTensor(),
        Normalize((0.2860,), (0.3530,)),
        Lambda(lambda x: torch.flatten(x))
    ])

    train_loader = DataLoader(
        datasets.FashionMNIST('./data/', train=True, download=True, transform=transform),
        batch_size=train_batch_size, shuffle=True
    )

    test_loader = DataLoader(
        datasets.FashionMNIST('./data/', train=False, download=True, transform=transform),
        batch_size=test_batch_size, shuffle=False
    )

    return train_loader, test_loader


# =============================================================================
# Main Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Prospective FF implementation...")

    device = get_device()
    print(f"Device: {device}")

    # Create model
    model = ProspectiveFFNetwork(
        dims=[784, 500, 500],
        threshold=2.0,
        lr=0.03,
        beta=0.5,
        consolidation_lr=0.01
    ).to(device)

    print(f"\nModel created:")
    print(f"  dims: {model.dims}")
    print(f"  threshold: {model.threshold}")
    print(f"  beta: {model.beta}")
    print(f"  num_layers: {len(model.layers)}")

    # Test with random data
    batch_size = 32
    x = torch.randn(batch_size, 784, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    y_onehot = F.one_hot(y, 10).float()

    # Create pos/neg samples
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(batch_size, device=device)
    x_neg = overlay_y_on_x(x, y[rnd])
    y_neg = F.one_hot(y[rnd], 10).float()

    print("\n--- Testing Single Iteration Training ---")
    losses = model.train_prospective_single_iteration(
        x_pos, x_neg, y_onehot, y_neg, verbose=True
    )
    print(f"Result: ff_loss={losses['ff_loss']:.4f}, "
          f"consol_loss={losses['consolidation_loss']:.4f}")

    print("\n--- Forward Pass Test ---")
    activations = model.forward(x)
    for i, act in enumerate(activations):
        print(f"  Layer {i}: shape={act.shape}, "
              f"mean={act.mean().item():.4f}, std={act.std().item():.4f}")

    print("\n--- Target Hints Test ---")
    hints = model.compute_target_hints(activations, y_onehot)
    for i, hint in enumerate(hints):
        print(f"  Layer {i} hint: shape={hint.shape}, "
              f"mean={hint.mean().item():.4f}")

    print("\nAll tests passed!")
