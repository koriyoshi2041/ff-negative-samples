"""
Predictive Forward-Forward (PFF) Implementation in PyTorch

Based on: Ororbia & Mali (2022) "The Predictive Forward-Forward Algorithm"
Reference: https://github.com/ago109/predictive-forward-forward

This implementation ports the official TensorFlow code to PyTorch, preserving:
1. Dual-circuit architecture (Representation + Generative circuits)
2. Competition matrix with lateral inhibition
3. Modified ReLU with full gradient flow
4. L2 normalization before weight multiplication
5. K-step iterative inference loop
6. Goodness calculation (sum of squared activations)

Key differences from standard FF:
- Recurrent dynamics with dampening (alpha)
- Top-down feedback from labels
- Lateral competition within layers
- Generative circuit for reconstruction
- Intertwined inference and learning

Default hyperparameters (from paper):
- n_units=2000, K=12, thr=10.0
- alpha=0.3 (dampening), beta=0.025 (latent update)
- eps_r=0.01, eps_g=0.025 (noise injection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Tuple, Optional, Dict, Any
import math


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# =============================================================================
# Core Functions
# =============================================================================

def create_competition_matrix(
    z_dim: int,
    n_group: int = 10,
    beta_scale: float = 1.0,
    alpha_scale: float = 1.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create competition matrix for lateral inhibition.

    Creates a block-diagonal structure where neurons within the same group
    compete with each other (lateral inhibition), while neurons in different
    groups do not interact.

    Adapted from Ororbia & Kifer 2022 (Nature Communications).

    Args:
        z_dim: Dimension of the layer (number of neurons)
        n_group: Size of each competition group
        beta_scale: Scale factor for off-diagonal (inhibitory) connections
        alpha_scale: Scale factor for diagonal (self) connections
        device: Target device for the tensor

    Returns:
        Competition matrix of shape (z_dim, z_dim)

    Example:
        >>> M = create_competition_matrix(100, n_group=10)
        >>> M.shape
        torch.Size([100, 100])
    """
    if device is None:
        device = get_device()

    diag = torch.eye(z_dim, device=device)
    V_l = None
    g_shift = 0

    while (z_dim - (n_group + g_shift)) >= 0:
        if g_shift > 0:
            left = torch.zeros(1, g_shift, device=device)
            middle = torch.ones(1, n_group, device=device)
            right = torch.zeros(1, z_dim - (n_group + g_shift), device=device)
            slice_row = torch.cat([left, middle, right], dim=1)
            for _ in range(n_group):
                V_l = torch.cat([V_l, slice_row], dim=0)
        else:
            middle = torch.ones(1, n_group, device=device)
            right = torch.zeros(1, z_dim - n_group, device=device)
            slice_row = torch.cat([middle, right], dim=1)
            for _ in range(n_group):
                if V_l is not None:
                    V_l = torch.cat([V_l, slice_row], dim=0)
                else:
                    V_l = slice_row
        g_shift += n_group

    # Apply scaling: off-diagonal gets beta_scale, diagonal gets alpha_scale
    V_l = V_l * (1.0 - diag) * beta_scale + diag * alpha_scale
    return V_l


def temperature_softmax(x: torch.Tensor, tau: float = 0.0) -> torch.Tensor:
    """
    Temperature-controlled softmax activation.

    Args:
        x: Input tensor of shape (batch, features)
        tau: Temperature parameter. If tau > 0, divides logits by tau.
             Higher tau -> softer distribution, lower tau -> harder distribution.

    Returns:
        Softmax probabilities of same shape as input
    """
    if tau > 0.0:
        x = x / tau
    # Numerical stability: subtract max
    max_x = x.max(dim=1, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    return exp_x / exp_x.sum(dim=1, keepdim=True)


class ModifiedReLU(torch.autograd.Function):
    """
    Modified ReLU activation for Forward-Forward.

    Forward: Standard ReLU (max(0, x))
    Backward: Full gradient flow (gradient = 1 everywhere, not 0 for x < 0)

    This allows gradients to flow even through "dead" neurons, which is
    important for the FF learning rule where we want to adjust weights
    even when the neuron isn't currently active.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return F.relu(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Full gradient flow: pretend derivatives exist everywhere
        # This is equivalent to: dx = torch.ones_like(x)
        return grad_output


def modified_relu(x: torch.Tensor) -> torch.Tensor:
    """Apply modified ReLU with full gradient flow."""
    return ModifiedReLU.apply(x)


def clip_activation(x: torch.Tensor) -> torch.Tensor:
    """Hard-clip activation to [0, 1] range."""
    return torch.clamp(x, 0.0, 1.0)


def l2_normalize(z: torch.Tensor, eps: float = 1e-8, scale: float = 1.0) -> torch.Tensor:
    """
    L2 normalize vectors along the feature dimension.

    Args:
        z: Input tensor of shape (batch, features)
        eps: Small constant for numerical stability
        scale: Optional scaling factor after normalization

    Returns:
        L2 normalized tensor of same shape
    """
    l2_norm = torch.norm(z, p=2, dim=1, keepdim=True)
    return (z / (l2_norm + eps)) * scale


# =============================================================================
# PFF Layer
# =============================================================================

class PFFLayer(nn.Module):
    """
    Single layer in the Predictive Forward-Forward network.

    Features:
    - Lateral inhibition via competition matrix
    - L2 normalization of inputs
    - Modified ReLU activation with full gradient flow
    - Dampening for temporal smoothing

    Args:
        in_features: Input dimension
        out_features: Output dimension (number of neurons)
        n_group: Size of competition groups for lateral inhibition
        use_lateral: Whether to use lateral inhibition (default: True)

    Example:
        >>> layer = PFFLayer(784, 2000)
        >>> x = torch.randn(32, 784)
        >>> z_prev = torch.zeros(32, 2000)
        >>> z_out = layer(x, z_prev, alpha=0.3)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_group: int = 10,
        use_lateral: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_group = n_group
        self.use_lateral = use_lateral

        # Main weight and bias (orthogonal initialization)
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(1, out_features))
        nn.init.orthogonal_(self.weight)
        nn.init.orthogonal_(self.bias)

        # Lateral inhibition weights (only if enabled)
        if use_lateral:
            self.lateral = nn.Parameter(torch.empty(out_features, out_features))
            nn.init.uniform_(self.lateral, 0.0, 0.05)
            # Competition matrix is fixed, not learned
            self.register_buffer('competition_matrix',
                                 create_competition_matrix(out_features, n_group))
        else:
            self.lateral = None
            self.competition_matrix = None

    def forward(
        self,
        x: torch.Tensor,
        z_prev: Optional[torch.Tensor] = None,
        feedback: Optional[torch.Tensor] = None,
        alpha: float = 0.0,
        eps: float = 0.0,
        rec_gamma: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass with recurrent dynamics.

        Args:
            x: Input tensor of shape (batch, in_features)
            z_prev: Previous state of this layer (batch, out_features)
            feedback: Top-down feedback signal (batch, out_features)
            alpha: Dampening factor for temporal smoothing (0 = no dampening)
            eps: Noise standard deviation for regularization
            rec_gamma: Scaling factor for lateral inhibition

        Returns:
            Output activations of shape (batch, out_features)
        """
        batch_size = x.shape[0]

        # L2 normalize input
        x_norm = l2_normalize(x)

        # Linear transformation
        z = torch.matmul(x_norm, self.weight) + self.bias

        # Add feedback if provided
        if feedback is not None:
            z = z + feedback

        # Add noise for regularization
        if eps > 0.0 and self.training:
            noise = torch.randn_like(z) * eps
            z = z + noise

        # Apply lateral inhibition
        if self.use_lateral and z_prev is not None and rec_gamma > 0.0:
            # L = ReLU(lateral) * competition_matrix * (1 - I) - ReLU(lateral) * I
            L = F.relu(self.lateral)
            eye = torch.eye(self.out_features, device=z.device)
            L = L * self.competition_matrix * (1.0 - eye) - L * eye
            z = z - torch.matmul(z_prev, L) * rec_gamma

        # Apply modified ReLU
        z = modified_relu(z)

        # Apply dampening (temporal smoothing)
        if alpha > 0.0 and z_prev is not None:
            z = z * (1.0 - alpha) + z_prev * alpha

        return z

    def forward_simple(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass without recurrence (for initialization)."""
        x_norm = l2_normalize(x)
        return modified_relu(torch.matmul(x_norm, self.weight) + self.bias)


# =============================================================================
# Representation Circuit
# =============================================================================

class RepresentationCircuit(nn.Module):
    """
    Representation (encoding) circuit of the PFF network.

    Two-layer architecture with:
    - Bottom-up sensory processing
    - Top-down label feedback
    - Lateral competition within layers
    - Cross-layer feedback (z2 -> z1)

    Args:
        x_dim: Input dimension (e.g., 784 for MNIST)
        y_dim: Number of classes (e.g., 10 for MNIST)
        n_units: Number of hidden units per layer (default: 2000)
        n_group: Competition group size (default: 10)
        use_lateral: Enable lateral inhibition (default: True)

    Example:
        >>> circuit = RepresentationCircuit(784, 10, n_units=2000)
        >>> x = torch.randn(32, 784)
        >>> y = F.one_hot(torch.randint(0, 10, (32,)), 10).float()
        >>> z1, z2, y_hat = circuit(x, y, K=12)
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_units: int = 2000,
        n_group: int = 10,
        use_lateral: bool = True
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_units = n_units

        # Layer 1: x -> z1 with feedback from z2
        self.W1 = nn.Parameter(torch.empty(x_dim, n_units))
        self.V2 = nn.Parameter(torch.empty(n_units, n_units))  # z2 -> z1 feedback
        self.b1 = nn.Parameter(torch.empty(1, n_units))

        # Layer 2: z1 -> z2 with feedback from y
        self.W2 = nn.Parameter(torch.empty(n_units, n_units))
        self.V = nn.Parameter(torch.empty(y_dim, n_units))  # y -> z2 feedback
        self.b2 = nn.Parameter(torch.empty(1, n_units))

        # Output layer: z2 -> y_hat (softmax)
        self.W = nn.Parameter(torch.empty(n_units, y_dim))
        self.b = nn.Parameter(torch.zeros(1, y_dim))

        # Lateral inhibition
        self.use_lateral = use_lateral
        if use_lateral:
            self.L1 = nn.Parameter(torch.empty(n_units, n_units))
            self.L2 = nn.Parameter(torch.empty(n_units, n_units))
            nn.init.uniform_(self.L1, 0.0, 0.05)
            nn.init.uniform_(self.L2, 0.0, 0.05)
            self.register_buffer('M1', create_competition_matrix(n_units, n_group))
            self.register_buffer('M2', create_competition_matrix(n_units, n_group))

        # Initialize weights (orthogonal)
        for param in [self.W1, self.W2, self.V2, self.V, self.W, self.b1, self.b2]:
            nn.init.orthogonal_(param)

    def forward_simple(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Simple forward pass (no recurrence) for initialization.

        Args:
            x: Input of shape (batch, x_dim)

        Returns:
            List of [z1, z2, y_hat]
        """
        z1 = modified_relu(torch.matmul(l2_normalize(x), self.W1) + self.b1)
        z2 = modified_relu(torch.matmul(l2_normalize(z1), self.W2) + self.b2)
        y_hat = temperature_softmax(torch.matmul(l2_normalize(z2), self.W) + self.b)
        return [z1, z2, y_hat]

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z1_prev: torch.Tensor,
        z2_prev: torch.Tensor,
        alpha: float = 0.3,
        eps_r: float = 0.01,
        rec_gamma: float = 1.0,
        y_scale: float = 5.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single step of recurrent inference.

        Args:
            x: Input of shape (batch, x_dim)
            y: Label (one-hot) of shape (batch, y_dim)
            z1_prev: Previous z1 state
            z2_prev: Previous z2 state
            alpha: Dampening factor
            eps_r: Noise standard deviation
            rec_gamma: Lateral inhibition strength
            y_scale: Scale factor for label signal

        Returns:
            Tuple of (z1, z2, y_hat)
        """
        batch_size = x.shape[0]
        y_scaled = y * y_scale

        # Add noise
        eps1 = torch.randn_like(z1_prev) * eps_r if self.training else 0
        eps2 = torch.randn_like(z2_prev) * eps_r if self.training else 0

        # Layer 1: input + feedback from z2
        z1 = (torch.matmul(l2_normalize(x), self.W1) +
              torch.matmul(l2_normalize(z2_prev), self.V2) +
              self.b1 + eps1)

        # Apply lateral inhibition to z1
        if self.use_lateral and rec_gamma > 0.0:
            L1 = F.relu(self.L1)
            eye = torch.eye(self.n_units, device=z1.device)
            L1 = L1 * self.M1 * (1.0 - eye) - L1 * eye
            z1 = z1 - torch.matmul(z1_prev, L1) * rec_gamma

        # Activation + dampening for z1
        z1 = modified_relu(z1) * (1.0 - alpha) + z1_prev * alpha

        # Layer 2: z1 (previous) + feedback from y
        z2 = (torch.matmul(l2_normalize(z1_prev), self.W2) +
              torch.matmul(y_scaled, self.V) +
              self.b2 + eps2)

        # Apply lateral inhibition to z2
        if self.use_lateral and rec_gamma > 0.0:
            L2 = F.relu(self.L2)
            eye = torch.eye(self.n_units, device=z2.device)
            L2 = L2 * self.M2 * (1.0 - eye) - L2 * eye
            z2 = z2 - torch.matmul(z2_prev, L2) * rec_gamma

        # Activation + dampening for z2
        z2 = modified_relu(z2) * (1.0 - alpha) + z2_prev * alpha

        # Output prediction (uses z2_prev for temporal consistency)
        y_hat = temperature_softmax(torch.matmul(l2_normalize(z2_prev), self.W) + self.b)

        return z1, z2, y_hat

    def get_parameters(self, include_lateral: bool = True) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        params = [self.b1, self.b2, self.W1, self.W2, self.V2, self.V, self.W, self.b]
        if include_lateral and self.use_lateral:
            params = [self.L1, self.L2] + params
        return params


# =============================================================================
# Generative Circuit
# =============================================================================

class GenerativeCircuit(nn.Module):
    """
    Generative (decoding) circuit of the PFF network.

    Reconstructs input from latent representations:
    z_g -> z2_hat -> z1_hat -> x_hat

    Args:
        x_dim: Output dimension (reconstructed input)
        n_units: Number of hidden units
        g_units: Dimension of top-most generative latent

    Example:
        >>> gen = GenerativeCircuit(784, 2000, 20)
        >>> z_g = torch.randn(32, 20)
        >>> x_hat = gen.sample(z_g)
    """

    def __init__(
        self,
        x_dim: int,
        n_units: int = 2000,
        g_units: int = 20
    ):
        super().__init__()

        self.x_dim = x_dim
        self.n_units = n_units
        self.g_units = g_units

        # Generative weights: top -> middle -> bottom
        self.Gy = nn.Parameter(torch.empty(g_units, n_units))  # z_g -> z2
        self.G2 = nn.Parameter(torch.empty(n_units, n_units))  # z2 -> z1
        self.G1 = nn.Parameter(torch.empty(n_units, x_dim))    # z1 -> x

        # Initialize
        nn.init.uniform_(self.Gy, 0.0, 0.05)
        nn.init.uniform_(self.G2, 0.0, 0.05)
        nn.init.uniform_(self.G1, 0.0, 0.05)

    def forward(
        self,
        z_g: torch.Tensor,
        z2_target: Optional[torch.Tensor] = None,
        z1_target: Optional[torch.Tensor] = None,
        eps_g: float = 0.025
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through generative circuit.

        When targets are provided, uses them with added noise instead of
        predictions (teacher forcing for reconstruction).

        Args:
            z_g: Top-most latent of shape (batch, g_units)
            z2_target: Target z2 from representation circuit
            z1_target: Target z1 from representation circuit
            eps_g: Noise standard deviation

        Returns:
            Tuple of (z2_hat, z1_hat, x_hat)
        """
        # z_g -> z2_hat
        z3_bar = modified_relu(z_g)
        z2_hat = torch.matmul(l2_normalize(z3_bar), self.Gy)

        # z2 -> z1_hat (use target with noise if provided)
        if z2_target is not None and self.training:
            eps2 = torch.randn_like(z2_target) * eps_g
            z2_input = modified_relu(z2_target + eps2)
        else:
            z2_input = modified_relu(z2_hat)
        z1_hat = torch.matmul(l2_normalize(z2_input), self.G2)

        # z1 -> x_hat (use target with noise if provided)
        if z1_target is not None and self.training:
            eps1 = torch.randn_like(z1_target) * eps_g
            z1_input = modified_relu(z1_target + eps1)
        else:
            z1_input = modified_relu(z1_hat)
        x_hat = clip_activation(torch.matmul(l2_normalize(z1_input), self.G1))

        return z2_hat, z1_hat, x_hat

    def sample(
        self,
        z_g: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample from the generative circuit.

        Args:
            z_g: Top-most latent of shape (batch, g_units). If provided, decodes from this.
            latent: Alternative latent input of shape (batch, g_units). Used if z_g is None.
                    In the original paper, this could come from a learned GMM prior.

        Returns:
            Reconstructed/sampled x of shape (batch, x_dim)

        Note:
            At least one of z_g or latent must be provided. The latent should be
            of dimension g_units, NOT y_dim (class labels).
        """
        # Use provided latent or z_g
        if z_g is not None:
            latent_input = z_g
        else:
            assert latent is not None, "Either z_g or latent must be provided"
            latent_input = latent

        # z_g (g_units) -> z2 (n_units) via Gy
        z_in = l2_normalize(modified_relu(latent_input))
        z2 = modified_relu(torch.matmul(z_in, self.Gy))

        # z2 (n_units) -> z1 (n_units) via G2
        z1 = modified_relu(torch.matmul(l2_normalize(z2), self.G2))

        # z1 (n_units) -> x (x_dim) via G1
        x_hat = clip_activation(torch.matmul(l2_normalize(z1), self.G1))

        return x_hat

    def get_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [self.Gy, self.G1, self.G2]


# =============================================================================
# Full PFF Network
# =============================================================================

class PFFNetwork(nn.Module):
    """
    Predictive Forward-Forward Network.

    Dual-circuit architecture combining:
    - Representation circuit: encodes input to latent
    - Generative circuit: reconstructs input from latent

    Training uses intertwined inference and learning with:
    - Local goodness-based loss for representation circuit
    - Reconstruction loss for generative circuit
    - K-step iterative settling

    Args:
        x_dim: Input dimension (e.g., 784 for MNIST)
        y_dim: Number of classes (e.g., 10 for MNIST)
        n_units: Hidden units per layer (default: 2000)
        g_units: Top generative latent dimension (default: 20)
        K: Number of inference steps (default: 12)
        thr: Goodness threshold (default: 10.0)
        alpha: Dampening factor (default: 0.3)
        beta: Latent update rate for z_g (default: 0.025)
        eps_r: Noise for representation circuit (default: 0.01)
        eps_g: Noise for generative circuit (default: 0.025)
        y_scale: Label signal scaling (default: 5.0)
        use_lateral: Enable lateral inhibition (default: True)
        use_generative: Enable generative circuit (default: True)

    Example:
        >>> model = PFFNetwork(784, 10)
        >>> x = torch.randn(32, 784)
        >>> y = F.one_hot(torch.randint(0, 10, (32,)), 10).float()
        >>> loss, y_hat, gen_loss, x_hat = model.infer(x, y, lab=torch.ones(32, 1))
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_units: int = 2000,
        g_units: int = 20,
        K: int = 12,
        thr: float = 10.0,
        alpha: float = 0.3,
        beta: float = 0.025,
        eps_r: float = 0.01,
        eps_g: float = 0.025,
        y_scale: float = 5.0,
        use_lateral: bool = True,
        use_generative: bool = True
    ):
        super().__init__()

        # Store hyperparameters
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_units = n_units
        self.g_units = g_units
        self.K = K
        self.thr = thr
        self.alpha = alpha
        self.beta = beta
        self.eps_r = eps_r
        self.eps_g = eps_g
        self.y_scale = y_scale
        self.use_lateral = use_lateral
        self.use_generative = use_generative

        # Representation circuit
        self.rep_circuit = RepresentationCircuit(
            x_dim, y_dim, n_units,
            n_group=10, use_lateral=use_lateral
        )

        # Generative circuit
        if use_generative:
            self.gen_circuit = GenerativeCircuit(x_dim, n_units, g_units)
        else:
            self.gen_circuit = None

        # Top-most generative latent (set during inference)
        self.z_g = None

    def calc_goodness(
        self,
        z: torch.Tensor,
        thr: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate goodness of activation vector.

        Goodness = sum of squared activations
        P(positive) = sigmoid(thr - goodness)

        Note: Original PFF uses SUM (not mean like standard FF)

        Args:
            z: Activation tensor of shape (batch, features)
            thr: Goodness threshold (default: self.thr)

        Returns:
            Tuple of (probability, logit)
        """
        if thr is None:
            thr = self.thr

        z_sqr = z.pow(2)
        delta = z_sqr.sum(dim=1, keepdim=True)
        # Negate: minimize for positive, maximize for negative
        delta = -delta + thr

        # P(positive)
        p = torch.sigmoid(delta)
        eps = 1e-5
        p = torch.clamp(p, eps, 1.0 - eps)

        return p, delta

    def calc_loss(
        self,
        z: torch.Tensor,
        lab: torch.Tensor,
        thr: Optional[float] = None,
        keep_batch: bool = False
    ) -> torch.Tensor:
        """
        Calculate local FF loss for an activation vector.

        Uses binary cross-entropy: CE = max(logit, 0) - logit * lab + log(1 + exp(-|logit|))

        Args:
            z: Activation tensor of shape (batch, features)
            lab: Binary label (1 for positive, 0 for negative) of shape (batch, 1)
            thr: Goodness threshold
            keep_batch: If True, return per-sample loss

        Returns:
            Loss scalar (or per-sample if keep_batch=True)
        """
        _, logit = self.calc_goodness(z, thr)

        # Numerically stable binary cross-entropy
        CE = torch.maximum(logit, torch.zeros_like(logit)) - logit * lab + \
             torch.log(1.0 + torch.exp(-torch.abs(logit)))

        L = CE.sum(dim=1, keepdim=True)

        if keep_batch:
            return L
        return L.mean()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Simple forward pass (no recurrence).

        Args:
            x: Input of shape (batch, x_dim)

        Returns:
            List of [z1, z2, y_hat]
        """
        return self.rep_circuit.forward_simple(x)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input by forward propagation.

        Args:
            x: Input of shape (batch, x_dim)

        Returns:
            Predicted probability distribution over classes
        """
        z = self.forward(x)
        return z[-1]  # y_hat

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lab: torch.Tensor,
        z_lat: List[torch.Tensor],
        zero_y: bool = False,
        reg_lambda: float = 0.0
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Single inference step with gradient computation.

        Args:
            x: Input of shape (batch, x_dim)
            y: Label (one-hot) of shape (batch, y_dim)
            lab: Binary label (1=positive, 0=negative) of shape (batch, 1)
            z_lat: Current latent states [z1, z2, y_hat]
            zero_y: If True, zero out label loss
            reg_lambda: Weight decay coefficient

        Returns:
            Tuple of (new z_lat, goodness loss, gradients)
        """
        N_pos = lab.sum()

        # Run one step of representation circuit
        z1, z2, y_hat = self.rep_circuit.step(
            x, y, z_lat[0], z_lat[1],
            alpha=self.alpha, eps_r=self.eps_r,
            y_scale=self.y_scale
        )

        # Calculate losses
        L1 = self.calc_loss(z1, lab, self.thr)
        L2 = self.calc_loss(z2, lab, self.thr)

        # Label prediction loss (only for positive samples)
        if zero_y:
            L3 = torch.tensor(0.0, device=x.device)
        else:
            # Cross-entropy loss weighted by lab
            y_scaled = y * self.y_scale
            ce = -(y_scaled * torch.log(y_hat + 1e-8)).sum(dim=1, keepdim=True)
            L3 = (ce * lab).sum() / (N_pos + 1e-8)

        # Regularization
        reg = torch.tensor(0.0, device=x.device)
        if reg_lambda > 0.0:
            reg = (torch.norm(self.rep_circuit.W1) +
                   torch.norm(self.rep_circuit.W2) +
                   torch.norm(self.rep_circuit.V2) +
                   torch.norm(self.rep_circuit.V) +
                   torch.norm(self.rep_circuit.W)) * reg_lambda

        L = L1 + L2 + L3 + reg
        L_goodness = L1 + L2

        return [z1, z2, y_hat], L_goodness, L

    def update_generator(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z_lat: List[torch.Tensor],
        opt: Optional[torch.optim.Optimizer] = None,
        reg_lambda: float = 0.0001
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update generative circuit.

        Args:
            x: Input of shape (batch, x_dim)
            y: Label of shape (batch, y_dim)
            z_lat: Current latent states [z1, z2, y_hat]
            opt: Optimizer for generative circuit
            reg_lambda: Weight decay coefficient

        Returns:
            Tuple of (generative loss, reconstructed x)
        """
        if self.gen_circuit is None:
            return torch.tensor(0.0, device=x.device), x * 0

        z1, z2, _ = z_lat

        # Forward through generative circuit
        z2_hat, z1_hat, x_hat = self.gen_circuit(
            self.z_g, z2, z1, self.eps_g
        )

        # Reconstruction losses
        z2_bar = modified_relu(z2_hat)
        z1_bar = modified_relu(z1_hat)

        e2 = z2_bar - z2
        L2 = (e2.pow(2).sum(dim=1, keepdim=True)).mean()

        e1 = z1_bar - z1
        L1 = (e1.pow(2).sum(dim=1, keepdim=True)).mean()

        e0 = x_hat - x
        L0 = (e0.pow(2).sum(dim=1, keepdim=True)).mean()

        # Regularization
        reg = torch.tensor(0.0, device=x.device)
        if reg_lambda > 0.0:
            reg = (torch.norm(self.gen_circuit.G1) +
                   torch.norm(self.gen_circuit.G2) +
                   torch.norm(self.gen_circuit.Gy)) * reg_lambda

        L = L2 + L1 + L0 + reg
        L_gen = L2 + L1 + L0

        # Update generative weights
        if opt is not None:
            opt.zero_grad()
            L.backward(retain_graph=True)
            # Clip gradients
            for p in self.gen_circuit.get_parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-1.0, 1.0)
            opt.step()

        # Update z_g via gradient descent on L2
        if self.z_g.grad is not None:
            self.z_g.grad.zero_()
        L2_for_z = (z2_bar - z2.detach()).pow(2).sum(dim=1, keepdim=True).mean()
        L2_for_z.backward(retain_graph=True)
        if self.z_g.grad is not None:
            self.z_g.data = self.z_g.data - self.z_g.grad.data * self.beta

        return L_gen, x_hat.detach()

    def infer(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lab: torch.Tensor,
        K: Optional[int] = None,
        rep_opt: Optional[torch.optim.Optimizer] = None,
        gen_opt: Optional[torch.optim.Optimizer] = None,
        reg_lambda: float = 0.0,
        g_reg_lambda: float = 0.0,
        zero_y: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run K-step inference with intertwined learning.

        This is the main training/inference function. For each of K steps:
        1. Update representation circuit (one step)
        2. Apply gradients to representation weights
        3. Update generative circuit

        Args:
            x: Input of shape (batch, x_dim)
            y: Label (one-hot) of shape (batch, y_dim)
            lab: Binary label (1=positive, 0=negative) of shape (batch, 1)
            K: Number of steps (default: self.K)
            rep_opt: Optimizer for representation circuit
            gen_opt: Optimizer for generative circuit
            reg_lambda: Weight decay for representation circuit
            g_reg_lambda: Weight decay for generative circuit
            zero_y: If True, zero out label loss

        Returns:
            Tuple of (goodness loss, y_hat, generative loss, x_hat)
        """
        if K is None:
            K = self.K

        # Initialize latents with forward pass
        z_lat = self.forward(x)

        # Initialize generative latent
        self.z_g = torch.zeros(x.shape[0], self.g_units,
                               device=x.device, requires_grad=True)

        L_total = torch.tensor(0.0, device=x.device)
        L_gen_total = torch.tensor(0.0, device=x.device)
        x_hat = x * 0

        for k in range(K):
            # Representation step
            if rep_opt is not None:
                rep_opt.zero_grad()

            z_lat, L_goodness, L_full = self.step(
                x, y, lab, z_lat, zero_y, reg_lambda
            )

            # Update representation weights
            if rep_opt is not None:
                L_full.backward(retain_graph=True)
                # Clip gradients
                for p in self.rep_circuit.get_parameters(include_lateral=self.use_lateral):
                    if p.grad is not None:
                        p.grad.data.clamp_(-1.0, 1.0)
                rep_opt.step()

            # Update generative circuit
            if self.use_generative:
                L_gen, x_hat = self.update_generator(
                    x, y, [z.detach() for z in z_lat],
                    gen_opt, g_reg_lambda
                )
                L_gen_total = L_gen_total + L_gen

            L_total = L_total + L_goodness.detach()

            # Detach for next iteration
            z_lat = [z.detach().requires_grad_(True) for z in z_lat]

        y_hat = z_lat[-1]
        return L_total / K, y_hat, L_gen_total / K, x_hat

    def sample(
        self,
        n_samples: int = 0,
        z_g: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Sample from the generative circuit.

        Args:
            n_samples: Number of samples to generate (if z_g and latent are None,
                       generates random samples)
            z_g: Top-most latent of shape (batch, g_units) to decode
            latent: Alternative latent input of shape (batch, g_units)
            device: Device for random samples

        Returns:
            Generated samples of shape (batch, x_dim)

        Example:
            >>> # Sample from random latent
            >>> samples = model.sample(n_samples=10)
            >>> # Sample from inferred latent
            >>> z_g = model.get_latent(x, y)
            >>> samples = model.sample(z_g=z_g)
        """
        if self.gen_circuit is None:
            raise ValueError("Generative circuit not enabled")

        # Generate random latent if none provided
        if z_g is None and latent is None:
            if device is None:
                device = next(self.parameters()).device
            latent = torch.randn(n_samples, self.g_units, device=device)

        return self.gen_circuit.sample(z_g, latent)

    def get_latent(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        K: Optional[int] = None,
        use_y_hat: bool = False
    ) -> torch.Tensor:
        """
        Infer the top-most generative latent z_g.

        Args:
            x: Input of shape (batch, x_dim)
            y: Label (one-hot) of shape (batch, y_dim)
            K: Number of inference steps
            use_y_hat: If True, use predicted y instead of given y

        Returns:
            Inferred z_g of shape (batch, g_units)
        """
        if K is None:
            K = self.K

        # Initialize with no gradient tracking for latents
        with torch.no_grad():
            z_lat = self.forward(x)
            z_lat = [z.detach() for z in z_lat]
        y_hat = z_lat[-1]

        # z_g needs gradients for the optimization loop
        self.z_g = torch.zeros(x.shape[0], self.g_units,
                               device=x.device, requires_grad=True)

        y_input = y_hat.detach() if use_y_hat else y

        # Run inference
        for k in range(K):
            # Step through representation circuit (no gradient tracking)
            with torch.no_grad():
                z1, z2, _ = self.rep_circuit.step(
                    x, y_input, z_lat[0], z_lat[1],
                    alpha=self.alpha, eps_r=0.0,  # No noise during inference
                    y_scale=self.y_scale
                )
                z_lat = [z1.detach(), z2.detach(), z_lat[2]]

            # Update z_g to match z2 (with gradient for z_g only)
            if self.gen_circuit is not None:
                z3_bar = modified_relu(self.z_g)
                z2_hat = torch.matmul(l2_normalize(z3_bar), self.gen_circuit.Gy)
                z2_bar = modified_relu(z2_hat)

                # Use detached z2 as target
                e2 = z2_bar - z2.detach()
                L2 = (e2.pow(2).sum(dim=1, keepdim=True)).mean()

                # Manual gradient descent on z_g
                if self.z_g.grad is not None:
                    self.z_g.grad.zero_()
                L2.backward()
                if self.z_g.grad is not None:
                    # Update z_g in place
                    with torch.no_grad():
                        self.z_g.data = self.z_g.data - self.z_g.grad.data * self.beta
                    # Re-create z_g with requires_grad for next iteration
                    self.z_g = self.z_g.detach().requires_grad_(True)

        return self.z_g.detach()


# =============================================================================
# Training Utilities
# =============================================================================

def create_pos_neg_samples(
    x: torch.Tensor,
    y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create positive and negative samples for PFF training.

    Positive samples: original (x, y) pairs with lab=1
    Negative samples: shuffled y with lab=0

    Args:
        x: Input data of shape (batch, features)
        y: Labels (one-hot) of shape (batch, num_classes)

    Returns:
        Tuple of (x_combined, y_combined, lab_combined, shuffle_idx)
    """
    batch_size = x.shape[0]

    # Shuffle indices for negative samples
    shuffle_idx = torch.randperm(batch_size, device=x.device)
    y_neg = y[shuffle_idx]

    # Combine positive and negative
    x_combined = torch.cat([x, x], dim=0)
    y_combined = torch.cat([y, y_neg], dim=0)

    # Labels: 1 for positive, 0 for negative
    lab_pos = torch.ones(batch_size, 1, device=x.device)
    lab_neg = torch.zeros(batch_size, 1, device=x.device)
    lab_combined = torch.cat([lab_pos, lab_neg], dim=0)

    return x_combined, y_combined, lab_combined, shuffle_idx


def train_pff(
    model: PFFNetwork,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    lr_rep: float = 0.001,
    lr_gen: float = 0.001,
    reg_lambda: float = 0.0,
    g_reg_lambda: float = 0.0001,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train a PFF network.

    Args:
        model: PFFNetwork to train
        train_loader: DataLoader with (x, y) batches
        num_epochs: Number of training epochs
        lr_rep: Learning rate for representation circuit
        lr_gen: Learning rate for generative circuit
        reg_lambda: Weight decay for representation
        g_reg_lambda: Weight decay for generative
        device: Target device
        verbose: Print progress

    Returns:
        Dictionary with training history
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    # Optimizers
    rep_opt = Adam(model.rep_circuit.get_parameters(model.use_lateral), lr=lr_rep)
    gen_opt = Adam(model.gen_circuit.get_parameters(), lr=lr_gen) if model.use_generative else None

    history = {'loss': [], 'gen_loss': [], 'accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_gen_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            # Convert to one-hot
            if batch_y.dim() == 1:
                batch_y = F.one_hot(batch_y, model.y_dim).float()
            batch_y = batch_y.to(device)

            # Create pos/neg samples
            x, y, lab, _ = create_pos_neg_samples(batch_x, batch_y)

            # Inference with learning
            loss, y_hat, gen_loss, _ = model.infer(
                x, y, lab,
                rep_opt=rep_opt, gen_opt=gen_opt,
                reg_lambda=reg_lambda, g_reg_lambda=g_reg_lambda
            )

            epoch_loss += loss.item() * batch_x.shape[0]
            epoch_gen_loss += gen_loss.item() * batch_x.shape[0]

            # Accuracy on positive samples only
            y_hat_pos = y_hat[:batch_x.shape[0]]
            pred = y_hat_pos.argmax(dim=1)
            target = batch_y.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += batch_x.shape[0]

        avg_loss = epoch_loss / total
        avg_gen_loss = epoch_gen_loss / total
        accuracy = correct / total

        history['loss'].append(avg_loss)
        history['gen_loss'].append(avg_gen_loss)
        history['accuracy'].append(accuracy)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Loss={avg_loss:.4f}, GenLoss={avg_gen_loss:.4f}, "
                  f"Acc={accuracy*100:.2f}%")

    return history


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Quick test
    print("Testing PFF implementation...")

    device = get_device()
    print(f"Device: {device}")

    # Create model
    model = PFFNetwork(
        x_dim=784,
        y_dim=10,
        n_units=2000,
        g_units=20,
        K=12,
        thr=10.0,
        alpha=0.3,
        beta=0.025
    ).to(device)

    print(f"\nModel created:")
    print(f"  x_dim: {model.x_dim}")
    print(f"  y_dim: {model.y_dim}")
    print(f"  n_units: {model.n_units}")
    print(f"  K: {model.K}")
    print(f"  thr: {model.thr}")

    # Test forward pass
    x = torch.randn(32, 784, device=device)
    y = F.one_hot(torch.randint(0, 10, (32,), device=device), 10).float()

    z_lat = model.forward(x)
    print(f"\nForward pass shapes:")
    print(f"  z1: {z_lat[0].shape}")
    print(f"  z2: {z_lat[1].shape}")
    print(f"  y_hat: {z_lat[2].shape}")

    # Test inference
    x_comb, y_comb, lab, _ = create_pos_neg_samples(x, y)
    loss, y_hat, gen_loss, x_hat = model.infer(x_comb, y_comb, lab, K=5)

    print(f"\nInference (K=5):")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gen Loss: {gen_loss.item():.4f}")
    print(f"  y_hat shape: {y_hat.shape}")
    print(f"  x_hat shape: {x_hat.shape}")

    # Test sampling from random latent
    samples = model.sample(n_samples=5)
    print(f"\nSampling (random latent):")
    print(f"  samples shape: {samples.shape}")

    # Test sampling from inferred latent
    z_g = model.get_latent(x[:5], y[:5], K=5)
    samples_from_z = model.sample(z_g=z_g)
    print(f"\nSampling (from inferred z_g):")
    print(f"  z_g shape: {z_g.shape}")
    print(f"  samples shape: {samples_from_z.shape}")

    # Test competition matrix
    M = create_competition_matrix(100, n_group=10, device=device)
    print(f"\nCompetition matrix:")
    print(f"  shape: {M.shape}")
    print(f"  block structure: {M[:10, :10].sum().item():.0f} ones in first block")

    print("\nAll tests passed!")
