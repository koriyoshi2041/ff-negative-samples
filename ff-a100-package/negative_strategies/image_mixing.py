"""
Image Mixing Strategy - Hybrid/Mixing approach.
Creates negative samples by mixing two different images.
"""

import torch
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('image_mixing')
class ImageMixingStrategy(NegativeStrategy):
    """
    Hinton's unsupervised method: mix two different images.
    
    Creates negative samples by:
    1. Pixel-wise mixing: neg = alpha * img1 + (1-alpha) * img2
    2. Or random masking: take pixels from img1 or img2 randomly
    
    Intuition: Mixed images don't belong to any real class,
    so they serve as good negative examples.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        alpha_range: tuple = (0.3, 0.7),
        mixing_mode: str = 'interpolate',  # 'interpolate' or 'mask'
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes
            alpha_range: Range for mixing coefficient (min, max)
            mixing_mode: 'interpolate' for weighted sum, 'mask' for random selection
        """
        super().__init__(num_classes=num_classes, **kwargs)
        self.alpha_range = alpha_range
        self.mixing_mode = mixing_mode
    
    def generate(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate negative samples by mixing pairs of images.
        
        Args:
            images: Input images (B, C, H, W) or (B, D)
            labels: Labels (B,) - used to ensure different classes are mixed
            
        Returns:
            Mixed negative samples (B, D)
        """
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)
        
        # Shuffle to get different images
        perm = torch.randperm(batch_size, device=images.device)
        other_images = flat[perm]
        
        if self.mixing_mode == 'interpolate':
            # Random mixing coefficients
            alpha = torch.rand(batch_size, 1, device=images.device)
            alpha = alpha * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]
            negative = alpha * flat + (1 - alpha) * other_images
            
        elif self.mixing_mode == 'mask':
            # Random binary mask
            mask = (torch.rand_like(flat) > 0.5).float()
            negative = flat * mask + other_images * (1 - mask)
            
        else:
            raise ValueError(f"Unknown mixing mode: {self.mixing_mode}")
        
        return negative
    
    @property
    def requires_labels(self) -> bool:
        return False  # Can work without labels
    
    def get_config(self):
        config = super().get_config()
        config['alpha_range'] = self.alpha_range
        config['mixing_mode'] = self.mixing_mode
        return config


@StrategyRegistry.register('class_aware_mixing')
class ClassAwareMixingStrategy(ImageMixingStrategy):
    """
    Variant that ensures mixed images come from different classes.
    """
    
    def generate(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Mix images from different classes."""
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)
        
        # Find pairs from different classes
        perm = torch.randperm(batch_size, device=images.device)
        other_images = flat[perm]
        other_labels = labels[perm]
        
        # For samples with same class after shuffle, rotate more
        same_class = labels == other_labels
        attempts = 0
        max_attempts = 5
        
        while same_class.any() and attempts < max_attempts:
            # Re-shuffle those with same class
            n_same = same_class.sum().item()
            new_perm = torch.randperm(batch_size, device=images.device)[:n_same]
            other_images[same_class] = flat[new_perm]
            other_labels[same_class] = labels[new_perm]
            same_class = labels == other_labels
            attempts += 1
        
        # Mix
        alpha = torch.rand(batch_size, 1, device=images.device)
        alpha = alpha * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]
        negative = alpha * flat + (1 - alpha) * other_images
        
        return negative
    
    @property
    def requires_labels(self) -> bool:
        return True  # Needs labels for class-aware mixing
