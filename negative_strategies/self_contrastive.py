"""
Self-Contrastive Strategy.
Uses augmented versions of the same image as negatives (SCFF).
"""

import torch
import torch.nn.functional as F
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('self_contrastive')
class SelfContrastiveStrategy(NegativeStrategy):
    """
    Self-Contrastive Forward-Forward (SCFF).
    
    Based on: "Self-Contrastive Forward-Forward Algorithm" (arXiv:2409.12184)
    
    Key idea: Use different augmentations of the same image as positive/negative pairs.
    - Positive: Weak augmentation (or original)
    - Negative: Strong augmentation that makes the image hard to recognize
    
    This enables self-supervised learning without labels.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        strong_aug_prob: float = 0.8,
        noise_std: float = 0.3,
        blur_kernel: int = 3,
        cutout_ratio: float = 0.3,
        color_jitter: float = 0.5,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes
            strong_aug_prob: Probability of applying each augmentation
            noise_std: Std of Gaussian noise to add
            blur_kernel: Size of blur kernel (for 2D images)
            cutout_ratio: Ratio of image to cut out
            color_jitter: Strength of color jittering
        """
        super().__init__(num_classes=num_classes, **kwargs)
        self.strong_aug_prob = strong_aug_prob
        self.noise_std = noise_std
        self.blur_kernel = blur_kernel
        self.cutout_ratio = cutout_ratio
        self.color_jitter = color_jitter
    
    def generate(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate strongly augmented negative samples.
        
        Args:
            images: Input images (B, C, H, W) or (B, D)
            labels: Labels (B,) - not used (self-supervised)
            
        Returns:
            Strongly augmented versions (B, D)
        """
        batch_size = images.size(0)
        
        # Handle both flattened and image formats
        if images.dim() == 2:
            # Already flattened, assume MNIST-like shape
            # Reshape to image for augmentation
            side = int(images.size(1) ** 0.5)
            if side * side == images.size(1):
                img = images.view(batch_size, 1, side, side)
            else:
                # Can't reshape, apply noise augmentation only
                return self._augment_flat(images)
        else:
            img = images
        
        # Apply strong augmentations
        neg = img.clone()
        
        # 1. Add Gaussian noise
        if torch.rand(1).item() < self.strong_aug_prob:
            noise = torch.randn_like(neg) * self.noise_std
            neg = neg + noise
        
        # 2. Random cutout (set random patches to zero)
        if torch.rand(1).item() < self.strong_aug_prob:
            neg = self._random_cutout(neg)
        
        # 3. Random pixel shuffle (local)
        if torch.rand(1).item() < self.strong_aug_prob:
            neg = self._local_shuffle(neg)
        
        # 4. Brightness/contrast jitter
        if torch.rand(1).item() < self.strong_aug_prob:
            neg = self._color_jitter(neg)
        
        return neg.view(batch_size, -1)
    
    def _augment_flat(self, flat: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to flattened input."""
        batch_size = flat.size(0)
        neg = flat.clone()
        
        # Add noise
        if torch.rand(1).item() < self.strong_aug_prob:
            neg = neg + torch.randn_like(neg) * self.noise_std
        
        # Random dropout (zero out random positions)
        if torch.rand(1).item() < self.strong_aug_prob:
            mask = torch.rand_like(neg) > self.cutout_ratio
            neg = neg * mask
        
        # Random shuffle of positions
        if torch.rand(1).item() < self.strong_aug_prob:
            perm = torch.randperm(neg.size(1), device=neg.device)
            # Only shuffle part of the image
            shuffle_mask = torch.rand(neg.size(1), device=neg.device) > 0.7
            for i in range(batch_size):
                neg[i, shuffle_mask] = neg[i, perm[shuffle_mask]]
        
        return neg
    
    def _random_cutout(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random cutout augmentation."""
        B, C, H, W = img.shape
        cut_h = int(H * self.cutout_ratio)
        cut_w = int(W * self.cutout_ratio)
        
        result = img.clone()
        for i in range(B):
            top = torch.randint(0, H - cut_h + 1, (1,)).item()
            left = torch.randint(0, W - cut_w + 1, (1,)).item()
            result[i, :, top:top+cut_h, left:left+cut_w] = 0
        
        return result
    
    def _local_shuffle(self, img: torch.Tensor) -> torch.Tensor:
        """Shuffle pixels locally within small patches."""
        B, C, H, W = img.shape
        result = img.clone()
        
        patch_size = 4
        for i in range(B):
            for ph in range(0, H, patch_size):
                for pw in range(0, W, patch_size):
                    # Get patch
                    patch = result[i, :, ph:ph+patch_size, pw:pw+patch_size].clone()
                    # Shuffle within patch
                    flat_patch = patch.view(C, -1)
                    perm = torch.randperm(flat_patch.size(1), device=img.device)
                    shuffled = flat_patch[:, perm].view(patch.shape)
                    result[i, :, ph:ph+patch_size, pw:pw+patch_size] = shuffled
        
        return result
    
    def _color_jitter(self, img: torch.Tensor) -> torch.Tensor:
        """Apply brightness/contrast jittering."""
        # Random brightness
        brightness = 1.0 + (torch.rand(1, device=img.device).item() - 0.5) * self.color_jitter
        result = img * brightness
        
        # Random contrast
        contrast = 1.0 + (torch.rand(1, device=img.device).item() - 0.5) * self.color_jitter
        mean = result.mean(dim=(-2, -1), keepdim=True)
        result = (result - mean) * contrast + mean
        
        return result.clamp(-3, 3)  # Clamp to reasonable range
    
    @property
    def requires_labels(self) -> bool:
        return False  # Self-supervised - no labels needed
    
    def get_config(self):
        config = super().get_config()
        config['strong_aug_prob'] = self.strong_aug_prob
        config['noise_std'] = self.noise_std
        config['cutout_ratio'] = self.cutout_ratio
        return config
