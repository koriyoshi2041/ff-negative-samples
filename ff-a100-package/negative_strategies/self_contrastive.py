"""
Self-Contrastive Forward-Forward Strategy (SCFF).

Based on: "Self-Contrastive Forward-Forward Algorithm" (Nature 2025, arXiv:2409.11593)
Official implementation: https://github.com/Toffooo/contrastive-forward-forward

CRITICAL IMPLEMENTATION DETAIL:
The official paper uses CONCATENATION, not addition:
- Positive sample: [x || x] (concatenate same image with itself)
- Negative sample: [x || x'] (concatenate different images)

The network layer then splits the input in half and applies the same weights
to both halves before summing: W(x1) + W(x2) + 2*bias

This creates a 2x wider input representation that the layer processes specially.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base import NegativeStrategy, StrategyRegistry


class SCFFLayer(nn.Linear):
    """
    SCFF-compatible layer that handles concatenated inputs.

    Based on the official implementation, this layer:
    1. Takes input of shape (B, 2*D) where D is the original feature dimension
    2. Splits input in half: x1, x2 = x[:, :D], x[:, D:]
    3. Applies same weights to both halves: W(x1) + W(x2) + 2*bias

    This is mathematically equivalent to having twin networks with shared weights.

    Args:
        in_features: Original feature dimension (D, not 2*D)
        out_features: Output dimension
        norm: Normalization type ('L2norm' or 'stdnorm')
        activation: Activation function (0=ReLU, 1=triangle)
        concat: If True, use concat mode (split and sum). If False, standard linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: str = 'stdnorm',
        activation: int = 0,
        concat: bool = True,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        # Note: in_features is the ORIGINAL dimension, not 2x
        super().__init__(in_features, out_features, bias, device, dtype)

        self.concat = concat
        self.norm_type = norm

        # Activation
        if activation == 0:
            self.act = nn.ReLU()
        else:
            self.act = self._triangle_activation

        self.relu = nn.ReLU()

    def _triangle_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Triangle activation: ReLU(x - mean(x))"""
        x = x - torch.mean(x, dim=1, keepdim=True)
        return F.relu(x)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization."""
        if self.norm_type == 'L2norm':
            return x / (x.norm(p=2, dim=1, keepdim=True) + 1e-10)
        else:  # stdnorm
            x = x - torch.mean(x, dim=1, keepdim=True)
            return x / (torch.std(x, dim=1, keepdim=True) + 1e-10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional concat mode.

        Args:
            x: Input tensor of shape (B, 2*D) if concat=True, else (B, D)

        Returns:
            Output tensor of shape (B, out_features)
        """
        x_normalized = self._normalize(x)

        if self.concat:
            # Split input in half and apply same weights to both
            half_dim = x.size(1) // 2
            x1 = x_normalized[:, :half_dim]
            x2 = x_normalized[:, half_dim:]
            # Apply same weights to both halves, sum results
            output = torch.mm(x1, self.weight.T) + torch.mm(x2, self.weight.T) + 2 * self.bias
        else:
            output = torch.mm(x_normalized, self.weight.T) + self.bias

        return output

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute goodness (sum of squared activations)."""
        return self.relu(x).pow(2).mean(dim=1)


@StrategyRegistry.register('self_contrastive')
class SelfContrastiveStrategy(NegativeStrategy):
    """
    Self-Contrastive Forward-Forward (SCFF).

    Based on: "Self-Contrastive Forward-Forward Algorithm" (Nature 2025, arXiv:2409.11593)

    CRITICAL: This implementation uses CONCATENATION (not addition) as per the official paper.

    Key mechanism:
    - Positive: Concatenate same image with itself: [x || x] or [x || augment(x)]
    - Negative: Concatenate different images: [x || x'] where x' is from different sample

    The output dimension is 2x the input dimension. Layers using SCFF must handle
    this by splitting the input and applying shared weights (see SCFFLayer).

    For classification/inference, uses label embedding (like Hinton's method) to enable
    fair comparison with other strategies.

    Args:
        num_classes: Number of classes (for classification mode)
        use_augmentation: If True, use augmented version for positive pair
        noise_std: Noise level for augmentation
        label_scale: Scale factor for label embedding (used in classification)
        num_negatives: Number of negative pairs per positive (default 1)
        eval_every: Evaluate every N epochs (0 = only at end, -1 = never during training)
    """

    def __init__(
        self,
        num_classes: int = 10,
        use_augmentation: bool = True,
        noise_std: float = 0.1,
        label_scale: float = 1.0,
        num_negatives: int = 1,
        eval_every: int = 0,
        **kwargs
    ):
        super().__init__(num_classes=num_classes, **kwargs)
        self.use_augmentation = use_augmentation
        self.noise_std = noise_std
        self.label_scale = label_scale
        self.num_negatives = num_negatives
        self.eval_every = eval_every
        self._training_mode = True

    def set_training_mode(self, mode: bool):
        """Set training/classification mode."""
        self._training_mode = mode
        return self

    def train(self):
        """Set to training mode (use SCFF concatenation)."""
        self._training_mode = True
        return self

    def set_inference_mode(self):
        """Set to classification mode (use label embedding)."""
        self._training_mode = False
        return self
    
    @property
    def output_dim_multiplier(self) -> int:
        """
        SCFF outputs 2x the input dimension due to concatenation.

        Layers should be aware of this when using SCFF strategy.
        """
        return 2 if self._training_mode else 1

    def create_positive(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        augmented_images: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Create positive samples using CONCATENATION.

        Training mode (SCFF): [x || x] or [x || augment(x)]
        - Concatenates same image with itself (or its augmentation)
        - Output shape: (B, 2*D)

        Classification mode: Standard flattened image with label embedding
        - Output shape: (B, D)

        Args:
            images: Input images (B, C, H, W) or (B, D)
            labels: Labels (B,) - used in classification mode for embedding
            augmented_images: Optional pre-augmented images (B, C, H, W) or (B, D)

        Returns:
            Positive samples - (B, 2*D) in training, (B, D) in classification
        """
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)

        if self._training_mode:
            # SCFF: Concatenate [x || x] or [x || augment(x)]
            if augmented_images is not None:
                # Use provided augmented images
                aug_flat = augmented_images.view(batch_size, -1)
            elif self.use_augmentation:
                # Apply simple noise augmentation
                aug_flat = flat + torch.randn_like(flat) * self.noise_std
            else:
                # No augmentation - use same image twice
                aug_flat = flat

            # CONCATENATION (not addition!)
            return torch.cat([flat, aug_flat], dim=1)
        else:
            # Classification: use label embedding
            result = flat.clone()
            one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
            one_hot.scatter_(1, labels.unsqueeze(1), self.label_scale)
            result[:, :self.num_classes] = one_hot
            return result

    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        augmented_images: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate negative samples using CONCATENATION.

        Training mode (SCFF): [x || x'] where x' is from different sample
        - Concatenates image with a DIFFERENT image from the batch
        - Output shape: (B * num_negatives, 2*D)

        Classification mode: Wrong label embedding
        - Output shape: (B, D)

        Args:
            images: Input images (B, C, H, W) or (B, D)
            labels: Labels (B,) - used in classification mode
            augmented_images: Optional augmented images for negative pairs

        Returns:
            Negative samples - (B*num_negatives, 2*D) in training, (B, D) in classification
        """
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)

        if self._training_mode:
            # Use augmented images if provided, else use original
            if augmented_images is not None:
                pair_source = augmented_images.view(batch_size, -1)
            else:
                pair_source = flat

            # Generate multiple negative pairs
            all_negatives = []
            for i in range(self.num_negatives):
                # Generate random shift to ensure different samples
                # Use modular arithmetic to ensure different samples
                shift = (torch.randperm(batch_size - 1, device=flat.device) + 1)[0].item()
                shifted = torch.roll(pair_source, shifts=int(shift), dims=0)

                # CONCATENATION (not addition!)
                neg_pair = torch.cat([flat, shifted], dim=1)
                all_negatives.append(neg_pair)

            return torch.cat(all_negatives, dim=0)
        else:
            # Classification: wrong label embedding
            result = flat.clone()
            wrong_labels = torch.randint(
                0, self.num_classes, (batch_size,),
                device=images.device
            )
            mask = wrong_labels == labels
            wrong_labels[mask] = (wrong_labels[mask] + 1) % self.num_classes

            one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
            one_hot.scatter_(1, wrong_labels.unsqueeze(1), self.label_scale)
            result[:, :self.num_classes] = one_hot
            return result

    def get_positive_negative_batch(
        self,
        images: torch.Tensor,
        augmented_images_1: Optional[torch.Tensor] = None,
        augmented_images_2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate both positive and negative batches efficiently (matching official impl).

        This method mirrors the official `get_pos_neg_batch_imgcats` function.

        Args:
            images: Original images (B, D) - already flattened
            augmented_images_1: First augmentation (B, D), or None to use images
            augmented_images_2: Second augmentation (B, D), or None to use images

        Returns:
            Tuple of (positive_batch, negative_batch):
            - positive_batch: (B, 2*D) - [x1 || x2] for same-identity pairs
            - negative_batch: (B * num_negatives, 2*D) - [x1 || x2'] for different pairs
        """
        batch_size = images.size(0)

        # Default to using original images if no augmentations provided
        x1 = augmented_images_1 if augmented_images_1 is not None else images
        x2 = augmented_images_2 if augmented_images_2 is not None else images

        # Positive: concatenate corresponding pairs
        positive_batch = torch.cat([x1, x2], dim=1)

        # Negative: concatenate mismatched pairs
        all_negatives = []
        indices = torch.arange(batch_size, device=images.device)

        for _ in range(self.num_negatives):
            # Random shift to get different samples
            shift = torch.randint(1, batch_size, (1,), device=images.device).item()
            shifted_indices = (indices + int(shift)) % batch_size
            x2_shifted = x2[shifted_indices]

            # Concatenate x1 with shifted x2
            neg_batch = torch.cat([x1, x2_shifted], dim=1)
            all_negatives.append(neg_batch)

        negative_batch = torch.cat(all_negatives, dim=0)

        return positive_batch, negative_batch

    def should_evaluate(self, epoch: int) -> bool:
        """Check if we should run linear probe evaluation this epoch."""
        if self.eval_every < 0:
            return False  # Never evaluate during training
        if self.eval_every == 0:
            return False  # Only evaluate at end (caller handles this)
        return epoch % self.eval_every == 0

    @property
    def requires_labels(self) -> bool:
        return False  # Self-supervised - no labels needed for training

    @property
    def uses_negatives(self) -> bool:
        """SCFF uses negatives (different from mono-forward)."""
        return True

    @property
    def requires_concat_layer(self) -> bool:
        """SCFF requires layers that handle concatenated input."""
        return True

    def get_config(self):
        config = super().get_config()
        config.update({
            'use_augmentation': self.use_augmentation,
            'noise_std': self.noise_std,
            'label_scale': self.label_scale,
            'num_negatives': self.num_negatives,
            'eval_every': self.eval_every,
            'output_dim_multiplier': self.output_dim_multiplier,
            'requires_concat_layer': self.requires_concat_layer,
        })
        return config

    @staticmethod
    def create_layer(
        in_features: int,
        out_features: int,
        norm: str = 'stdnorm',
        activation: int = 0,
        concat: bool = True
    ) -> SCFFLayer:
        """
        Factory method to create an SCFF-compatible layer.

        Args:
            in_features: Original feature dimension (will be split in concat mode)
            out_features: Output dimension
            norm: Normalization type
            activation: Activation function
            concat: Whether to use concat mode

        Returns:
            SCFFLayer configured for SCFF training
        """
        return SCFFLayer(
            in_features=in_features,
            out_features=out_features,
            norm=norm,
            activation=activation,
            concat=concat
        )
