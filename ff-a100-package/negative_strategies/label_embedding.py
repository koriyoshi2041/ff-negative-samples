"""
Label Embedding Strategy - Hinton's original method.
Embeds wrong labels into the first pixels of the image.
"""

import torch
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('label_embedding')
class LabelEmbeddingStrategy(NegativeStrategy):
    """
    Hinton's original negative sample strategy.
    
    Embeds a one-hot encoded wrong label into the first N pixels
    of the flattened image, where N = num_classes.
    
    Positive: correct label embedded
    Negative: random wrong label embedded
    """
    
    def __init__(self, num_classes: int = 10, label_scale: float = 1.0, **kwargs):
        """
        Args:
            num_classes: Number of classes
            label_scale: Scale factor for embedded labels (default 1.0)
        """
        super().__init__(num_classes=num_classes, **kwargs)
        self.label_scale = label_scale
    
    def create_positive(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Create positive samples with correct label embedded."""
        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()

        # Use x.max() as the label value (like Hinton's original implementation)
        label_value = flat.abs().max() if flat.numel() > 0 else 1.0

        # Zero out label positions and embed correct label
        flat[:, :self.num_classes] = 0.0
        flat[range(batch_size), labels] = label_value
        return flat
    
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate negative samples with wrong labels embedded.

        Args:
            images: Input images (B, C, H, W) or (B, D)
            labels: Correct labels (B,)

        Returns:
            Negative samples with wrong labels embedded (B, D)
        """
        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()

        # Use x.max() as the label value (like Hinton's original implementation)
        label_value = flat.abs().max() if flat.numel() > 0 else 1.0

        # Generate random wrong labels
        wrong_labels = torch.randint(
            0, self.num_classes, (batch_size,),
            device=images.device
        )
        # Ensure labels are actually wrong
        mask = wrong_labels == labels
        wrong_labels[mask] = (wrong_labels[mask] + 1) % self.num_classes

        # Zero out label positions and embed wrong label
        flat[:, :self.num_classes] = 0.0
        flat[range(batch_size), wrong_labels] = label_value
        return flat
    
    @property
    def requires_labels(self) -> bool:
        return True
    
    def get_config(self):
        config = super().get_config()
        config['label_scale'] = self.label_scale
        return config
