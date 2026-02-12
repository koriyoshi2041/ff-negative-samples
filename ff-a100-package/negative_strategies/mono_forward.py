"""
Mono-Forward Strategy: Forward-Forward without explicit negative samples.

This module implements strategies for training FF networks without traditional
negative samples, using alternative loss functions instead.

Based on research suggesting that negative samples may not be strictly necessary
for learning, these strategies explore:
1. Contrastive loss within positive samples (self-supervised)
2. Variance maximization (prevent collapse)
3. Feature decorrelation (Barlow Twins-style)
4. Direct goodness targets

References:
- "Mono-Forward: Backpropagation-Free Algorithm for Efficient Neural Network Training"
  (arXiv:2501.08756)
- "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" (Zbontar et al., 2021)
- "VICReg: Variance-Invariance-Covariance Regularization" (Bardes et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any, Literal
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('mono_forward')
class MonoForwardStrategy(NegativeStrategy):
    """
    Mono-Forward: Training Forward-Forward networks without explicit negative samples.

    This strategy explores whether FF can work without the traditional positive vs
    negative contrast by using alternative objectives:

    Loss Types:
    -----------
    1. 'contrastive': Contrastive loss within positive samples
       - Creates two augmented views of each sample
       - Pulls together representations of same sample
       - Pushes apart representations of different samples (within batch)

    2. 'variance': Variance maximization loss
       - Ensures activations have high variance across the batch
       - Prevents representation collapse

    3. 'decorrelation': Feature decorrelation (Barlow Twins-style)
       - Encourages different features to be uncorrelated
       - Reduces redundancy in learned representations

    4. 'vicreg': Combined VICReg-style loss
       - Variance: prevent collapse
       - Invariance: bring views together
       - Covariance: decorrelate features

    5. 'direct_goodness': Direct goodness target
       - Simply targets a specific goodness value
       - Most similar to original FF but without negatives

    Key Insight:
    ------------
    Traditional FF uses "goodness" contrast between positive and negative samples.
    Mono-Forward achieves similar learning by:
    - Using self-supervised objectives on positive samples only
    - Preventing collapse through variance/decorrelation terms
    - Creating implicit "negatives" through within-batch contrasts

    Example Usage:
    --------------
    >>> strategy = MonoForwardStrategy(
    ...     num_classes=10,
    ...     loss_type='vicreg',
    ...     augmentation_noise=0.1
    ... )
    >>>
    >>> # In training loop:
    >>> positive = strategy.create_positive(images, labels)
    >>> # No negative samples needed!
    >>> loss = strategy.compute_mono_loss(activations, labels)

    Attributes:
        loss_type: Type of alternative loss function
        augmentation_noise: Noise level for creating augmented views
        target_goodness: Target goodness value (for direct_goodness loss)
        lambda_variance: Weight for variance loss term
        lambda_covariance: Weight for covariance/decorrelation term
        lambda_invariance: Weight for invariance term
        use_label_embedding: Whether to embed labels in input
    """

    # Type alias for supported loss types
    LossType = Literal['contrastive', 'variance', 'decorrelation', 'vicreg', 'direct_goodness']

    def __init__(
        self,
        num_classes: int = 10,
        loss_type: LossType = 'vicreg',
        augmentation_noise: float = 0.1,
        target_goodness: float = 2.0,
        lambda_variance: float = 25.0,
        lambda_covariance: float = 1.0,
        lambda_invariance: float = 25.0,
        use_label_embedding: bool = True,
        temperature: float = 0.5,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize the Mono-Forward strategy.

        Args:
            num_classes: Number of classes in the dataset (default: 10)
            loss_type: Type of alternative loss function to use
                - 'contrastive': InfoNCE-style contrastive loss within positives
                - 'variance': Variance maximization to prevent collapse
                - 'decorrelation': Barlow Twins-style feature decorrelation
                - 'vicreg': Combined variance-invariance-covariance loss
                - 'direct_goodness': Simple target goodness matching
            augmentation_noise: Standard deviation of Gaussian noise for
                creating augmented views (default: 0.1)
            target_goodness: Target goodness value for direct_goodness loss
                (default: 2.0)
            lambda_variance: Weight for variance loss term in vicreg
                (default: 25.0)
            lambda_covariance: Weight for covariance regularization
                (default: 1.0)
            lambda_invariance: Weight for invariance term in vicreg
                (default: 25.0)
            use_label_embedding: Whether to embed class labels in input
                (default: True, helps with classification)
            temperature: Temperature for contrastive loss (default: 0.5)
            device: Device to place tensors on
        """
        super().__init__(num_classes=num_classes, device=device)

        self.loss_type = loss_type
        self.augmentation_noise = augmentation_noise
        self.target_goodness = target_goodness
        self.lambda_variance = lambda_variance
        self.lambda_covariance = lambda_covariance
        self.lambda_invariance = lambda_invariance
        self.use_label_embedding = use_label_embedding
        self.temperature = temperature

        # Cache for augmented views (used in loss computation)
        self._view1: Optional[torch.Tensor] = None
        self._view2: Optional[torch.Tensor] = None

    def create_positive(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Create positive samples with optional label embedding and augmentation.

        For contrastive/vicreg losses, this method also caches augmented views
        that will be used in compute_mono_loss().

        Args:
            images: Input images, shape (B, C, H, W) or (B, D)
            labels: Class labels, shape (B,)

        Returns:
            Positive samples with shape (B, D) where D is flattened dimension
        """
        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()

        # Create two augmented views for contrastive-style losses
        if self.loss_type in ['contrastive', 'vicreg']:
            self._view1 = flat + torch.randn_like(flat) * self.augmentation_noise
            self._view2 = flat + torch.randn_like(flat) * self.augmentation_noise
        else:
            self._view1 = flat
            self._view2 = None

        # Optionally embed label information for supervised signal
        result = flat.clone()
        if self.use_label_embedding:
            one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            result[:, :self.num_classes] = one_hot

            # Also embed in cached views
            if self._view1 is not None:
                self._view1[:, :self.num_classes] = one_hot
            if self._view2 is not None:
                self._view2[:, :self.num_classes] = one_hot

        return result

    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate 'negative' samples - returns positive samples since mono-forward
        doesn't use explicit negatives.

        This method exists for API compatibility with other strategies.
        The returned tensor should be ignored in the training loop when
        using mono-forward.

        Args:
            images: Input images, shape (B, ...)
            labels: Labels, shape (B,)

        Returns:
            Same as positive samples (dummy return for compatibility)
        """
        # Return positive samples - no explicit negatives in mono-forward
        return self.create_positive(images, labels, **kwargs)

    def compute_mono_loss(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        activations_view2: Optional[torch.Tensor] = None,
        goodness_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute the mono-forward loss without negative samples.

        This is the core method that replaces the traditional FF contrastive loss.

        Args:
            activations: Layer activations from view 1, shape (B, D)
            labels: Class labels, shape (B,)
            activations_view2: Optional activations from second view (for vicreg)
            goodness_fn: Optional custom function to compute goodness
                Default: sum of squared activations

        Returns:
            Scalar loss value

        Raises:
            ValueError: If loss_type is unknown
        """
        if self.loss_type == 'contrastive':
            return self._contrastive_loss(activations, labels)
        elif self.loss_type == 'variance':
            return self._variance_loss(activations)
        elif self.loss_type == 'decorrelation':
            return self._decorrelation_loss(activations)
        elif self.loss_type == 'vicreg':
            return self._vicreg_loss(activations, activations_view2)
        elif self.loss_type == 'direct_goodness':
            return self._direct_goodness_loss(activations, goodness_fn)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _contrastive_loss(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss within positive samples (InfoNCE-style).

        Creates implicit negatives by treating other samples in the batch
        as negatives. This provides a learning signal without explicit
        negative sample generation.

        The loss encourages:
        - Same sample representations (across augmentations) to be similar
        - Different sample representations to be dissimilar

        Args:
            activations: Normalized activations, shape (B, D)
            labels: Class labels (used for supervised variant)

        Returns:
            Contrastive loss value
        """
        batch_size = activations.size(0)

        # Normalize activations for cosine similarity
        z = F.normalize(activations, dim=1)

        # Compute similarity matrix
        similarity = torch.mm(z, z.t()) / self.temperature  # (B, B)

        # Create mask for same-class samples (positives)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)

        # Self-mask (diagonal)
        self_mask = torch.eye(batch_size, device=activations.device, dtype=torch.bool)

        # Positives: same class, different sample
        positive_mask = labels_equal & ~self_mask

        # Negatives: different class
        negative_mask = ~labels_equal

        # InfoNCE-style loss
        # For each sample, pull together same-class samples, push apart different-class

        # Compute log-sum-exp over negatives
        neg_sim = similarity.masked_fill(~negative_mask, float('-inf'))
        neg_logsumexp = torch.logsumexp(neg_sim, dim=1)  # (B,)

        # Compute mean of positive similarities
        pos_sim = similarity.masked_fill(~positive_mask, 0)
        pos_count = positive_mask.sum(dim=1).clamp(min=1)
        pos_mean = pos_sim.sum(dim=1) / pos_count  # (B,)

        # Loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = -pos + log(exp(pos) + sum(exp(neg)))
        # Simplified: -pos + logsumexp([pos, negs])
        loss = -pos_mean + neg_logsumexp

        return loss.mean()

    def _variance_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute variance maximization loss to prevent representation collapse.

        This loss ensures that activations have high variance across the batch,
        which is crucial for learning meaningful representations without negatives.

        Without this regularization, the network might collapse to outputting
        constant activations for all inputs.

        Implementation:
        - Compute std of each feature across batch
        - Apply hinge loss to encourage std above threshold (1.0)
        - Average across all features

        Args:
            activations: Layer activations, shape (B, D)

        Returns:
            Variance loss value (to minimize = maximize variance)
        """
        # Compute standard deviation of each feature across batch
        std = activations.std(dim=0)  # (D,)

        # Hinge loss: penalize if std < 1
        # We want std >= 1, so loss = max(0, 1 - std)
        variance_loss = F.relu(1.0 - std).mean()

        return variance_loss

    def _decorrelation_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute decorrelation loss (Barlow Twins-style).

        This loss encourages different features to be uncorrelated, reducing
        redundancy in the learned representations. Each feature should capture
        unique information about the input.

        Implementation:
        - Normalize activations (zero mean, unit variance per feature)
        - Compute correlation matrix
        - Penalize off-diagonal elements (feature correlations)

        Args:
            activations: Layer activations, shape (B, D)

        Returns:
            Decorrelation loss value
        """
        batch_size = activations.size(0)

        # Normalize: zero mean, unit variance per feature
        z = activations - activations.mean(dim=0, keepdim=True)
        std = z.std(dim=0, keepdim=True) + 1e-8
        z = z / std

        # Compute correlation matrix: (D, D)
        correlation = torch.mm(z.t(), z) / batch_size

        # Create mask for off-diagonal elements
        eye = torch.eye(correlation.size(0), device=activations.device)
        off_diagonal = 1 - eye

        # Penalize off-diagonal correlations (want them to be 0)
        decorrelation_loss = (correlation * off_diagonal).pow(2).sum() / correlation.size(0)

        return decorrelation_loss

    def _vicreg_loss(
        self,
        z1: torch.Tensor,
        z2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute VICReg-style combined loss.

        VICReg (Variance-Invariance-Covariance Regularization) combines three
        objectives that together enable self-supervised learning:

        1. Variance: Prevent collapse by ensuring high variance
        2. Invariance: Pull together representations of augmented views
        3. Covariance: Decorrelate features to reduce redundancy

        This is the most comprehensive mono-forward loss and typically works
        best when properly tuned.

        Args:
            z1: Activations from first view, shape (B, D)
            z2: Activations from second view (optional), shape (B, D)
                If None, only variance and covariance terms are computed

        Returns:
            Combined VICReg loss value
        """
        loss = 0.0

        # Variance term: prevent collapse
        variance_loss = self._variance_loss(z1)
        if z2 is not None:
            variance_loss = (variance_loss + self._variance_loss(z2)) / 2
        loss = loss + self.lambda_variance * variance_loss

        # Invariance term: pull views together (if two views provided)
        if z2 is not None:
            invariance_loss = F.mse_loss(z1, z2)
            loss = loss + self.lambda_invariance * invariance_loss

        # Covariance term: decorrelate features
        covariance_loss = self._decorrelation_loss(z1)
        if z2 is not None:
            covariance_loss = (covariance_loss + self._decorrelation_loss(z2)) / 2
        loss = loss + self.lambda_covariance * covariance_loss

        return loss

    def _direct_goodness_loss(
        self,
        activations: torch.Tensor,
        goodness_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute direct goodness target loss.

        This is the simplest mono-forward loss: directly target a specific
        goodness value. Unlike traditional FF which contrasts positive vs
        negative goodness, this just pushes positive goodness toward a target.

        The intuition is that if all real data has similar goodness, the
        network learns features that produce consistent goodness for valid inputs.

        Args:
            activations: Layer activations, shape (B, D)
            goodness_fn: Function to compute goodness from activations
                Default: sum of squared activations

        Returns:
            MSE loss between computed and target goodness
        """
        if goodness_fn is None:
            goodness_fn = lambda x: (x ** 2).sum(dim=1)

        goodness = goodness_fn(activations)  # (B,)
        target = torch.full_like(goodness, self.target_goodness)

        return F.mse_loss(goodness, target)

    def get_augmented_views(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get the cached augmented views created in create_positive().

        Useful for training loops that need to forward both views through
        the network separately (e.g., for vicreg loss).

        Returns:
            Tuple of (view1, view2) tensors, or (None, None) if not available
        """
        return self._view1, self._view2

    @property
    def requires_labels(self) -> bool:
        """Whether this strategy requires labels."""
        # Contrastive loss uses labels for supervised contrast
        # Label embedding also needs labels
        return self.use_label_embedding or self.loss_type == 'contrastive'

    @property
    def uses_negatives(self) -> bool:
        """Mono-forward does not use explicit negative samples."""
        return False

    def get_training_config(self) -> Dict[str, Any]:
        """
        Get configuration for the training loop.

        Returns a dict that tells the training loop:
        - That this strategy doesn't use negatives
        - What loss function to use
        - Whether two views are needed

        Returns:
            Dictionary with training configuration
        """
        return {
            'use_negatives': False,
            'loss_fn': self.compute_mono_loss,
            'loss_type': self.loss_type,
            'needs_two_views': self.loss_type in ['contrastive', 'vicreg'],
            'target_goodness': self.target_goodness,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return complete strategy configuration for logging."""
        config = super().get_config()
        config.update({
            'loss_type': self.loss_type,
            'augmentation_noise': self.augmentation_noise,
            'target_goodness': self.target_goodness,
            'lambda_variance': self.lambda_variance,
            'lambda_covariance': self.lambda_covariance,
            'lambda_invariance': self.lambda_invariance,
            'use_label_embedding': self.use_label_embedding,
            'temperature': self.temperature,
            'uses_negatives': False,
        })
        return config


@StrategyRegistry.register('variance_only')
class VarianceOnlyStrategy(MonoForwardStrategy):
    """
    Simplified mono-forward using only variance loss.

    This is the most minimal mono-forward strategy: just prevent collapse
    by maximizing activation variance. Good for ablation studies.

    Example:
        >>> strategy = VarianceOnlyStrategy(num_classes=10)
        >>> loss = strategy.compute_mono_loss(activations, labels)
    """

    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__(
            num_classes=num_classes,
            loss_type='variance',
            **kwargs
        )


@StrategyRegistry.register('decorrelation_only')
class DecorrelationOnlyStrategy(MonoForwardStrategy):
    """
    Simplified mono-forward using only decorrelation loss.

    Uses Barlow Twins-style decorrelation without variance term.
    May be prone to collapse without variance regularization.

    Example:
        >>> strategy = DecorrelationOnlyStrategy(num_classes=10)
        >>> loss = strategy.compute_mono_loss(activations, labels)
    """

    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__(
            num_classes=num_classes,
            loss_type='decorrelation',
            **kwargs
        )


@StrategyRegistry.register('energy_minimization')
class EnergyMinimizationStrategy(MonoForwardStrategy):
    """
    Energy-based mono-forward strategy.

    Instead of goodness contrast, uses energy-based objectives:
    - 'entropy': Minimize activation entropy (encourage confident outputs)
    - 'sparsity': Encourage sparse activations (L1 regularization)
    - 'reconstruction': Auto-encoder style reconstruction (requires decoder)

    These objectives provide implicit learning signals without explicit negatives.

    Attributes:
        energy_type: Type of energy function ('entropy', 'sparsity')
        sparsity_target: Target sparsity level for sparsity energy
    """

    EnergyType = Literal['entropy', 'sparsity']

    def __init__(
        self,
        num_classes: int = 10,
        energy_type: EnergyType = 'entropy',
        sparsity_target: float = 0.1,
        **kwargs
    ):
        """
        Initialize the energy minimization strategy.

        Args:
            num_classes: Number of classes
            energy_type: Type of energy function
                - 'entropy': Minimize entropy of activations
                - 'sparsity': Encourage sparse activations
            sparsity_target: Target fraction of active units (default: 0.1)
        """
        super().__init__(num_classes=num_classes, loss_type='direct_goodness', **kwargs)
        self.energy_type = energy_type
        self.sparsity_target = sparsity_target

    def compute_mono_loss(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        activations_view2: Optional[torch.Tensor] = None,
        goodness_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute energy-based loss.

        Args:
            activations: Layer activations, shape (B, D)
            labels: Class labels (unused for unsupervised energy)
            activations_view2: Unused (for API compatibility)
            goodness_fn: Unused (energy function is determined by energy_type)

        Returns:
            Energy loss value
        """
        if self.energy_type == 'entropy':
            return self._entropy_loss(activations)
        elif self.energy_type == 'sparsity':
            return self._sparsity_loss(activations)
        else:
            raise ValueError(f"Unknown energy type: {self.energy_type}")

    def _entropy_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy minimization loss.

        Encourages confident (low entropy) activations by treating
        activations as probabilities and minimizing their entropy.

        Args:
            activations: Layer activations, shape (B, D)

        Returns:
            Entropy loss value
        """
        # Convert to probabilities
        probs = F.softmax(activations, dim=1)

        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        return entropy.mean()

    def _sparsity_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity loss.

        Encourages sparse activations through a KL divergence term
        between actual and target sparsity.

        Args:
            activations: Layer activations, shape (B, D)

        Returns:
            Sparsity loss value
        """
        # Average activation per unit
        avg_activation = torch.sigmoid(activations).mean(dim=0)  # (D,)

        # KL divergence from target sparsity
        # KL(target || actual) for each unit
        target = self.sparsity_target
        kl = (
            target * torch.log(target / (avg_activation + 1e-8)) +
            (1 - target) * torch.log((1 - target) / (1 - avg_activation + 1e-8))
        )

        return kl.sum()

    def get_config(self) -> Dict[str, Any]:
        """Return complete strategy configuration."""
        config = super().get_config()
        config.update({
            'energy_type': self.energy_type,
            'sparsity_target': self.sparsity_target,
        })
        return config
