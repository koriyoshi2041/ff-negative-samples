"""
Masking Strategy for Forward-Forward Algorithm.

Creates negative samples by randomly masking/corrupting parts of positive images.
This approach generates "hard negatives" that are structurally similar to positives
but contain corrupted information, forcing the network to learn robust features.

The masking strategy is inspired by:
- Masked Autoencoders (MAE) - random patch masking
- Cutout augmentation - block-based masking
- Dropout regularization - random pixel masking

Key insight: By corrupting parts of valid images, we create negatives that
share statistical properties with positives but are semantically invalid,
providing a stronger learning signal than pure noise.
"""

import torch
from typing import Optional
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('masking')
class MaskingStrategy(NegativeStrategy):
    """
    Masking-based negative sample generation strategy.

    Creates negative samples by randomly masking (corrupting) portions of
    positive images. This produces "hard negatives" that share low-level
    statistics with real data but contain corrupted/invalid information.

    Mask Types:
        - 'random': Independent random pixel masking (like dropout)
        - 'block': Contiguous square block masking (like cutout)
        - 'patch': MAE-style patch masking with configurable patch size

    Mask Modes (what to fill masked regions with):
        - 'zero': Set masked pixels to zero (simplest)
        - 'noise': Replace with random Gaussian noise
        - 'shuffle': Shuffle masked pixels with other parts of the image
        - 'mean': Replace with mean pixel value of the image

    Example usage:
        >>> strategy = MaskingStrategy(mask_ratio=0.3, mask_type='random')
        >>> negatives = strategy.generate(images, labels)

        >>> # Patch-based masking (MAE-style)
        >>> strategy = MaskingStrategy(mask_ratio=0.5, mask_type='patch', patch_size=4)
        >>> negatives = strategy.generate(images, labels)

    Note:
        This strategy does not require labels as it only corrupts the input
        images without any label-dependent processing.
    """

    def __init__(
        self,
        num_classes: int = 10,
        mask_ratio: float = 0.3,
        mask_mode: str = 'zero',
        mask_type: str = 'random',
        block_size: int = 4,
        patch_size: int = 4,
        **kwargs
    ):
        """
        Initialize the MaskingStrategy.

        Args:
            num_classes: Number of classes in the dataset. Used by base class
                for compatibility, not used by this strategy.
            mask_ratio: Fraction of pixels/patches to mask, in range (0.0, 1.0).
                - 0.3 (default): Moderate corruption, preserves most structure
                - 0.5: Heavy corruption, used in MAE
                - 0.75: Extreme corruption, very hard negatives
            mask_mode: How to fill masked regions:
                - 'zero': Set to zero (default, simple and effective)
                - 'noise': Replace with N(0, 1) Gaussian noise
                - 'shuffle': Swap with random positions in same image
                - 'mean': Fill with image's mean pixel value
            mask_type: Spatial pattern of masking:
                - 'random': Independent per-pixel (like dropout)
                - 'block': Single contiguous square region (like cutout)
                - 'patch': MAE-style non-overlapping patch masking
            block_size: Size of blocks for 'block' mask_type.
                Larger blocks create more localized corruption.
            patch_size: Size of patches for 'patch' mask_type.
                Should divide image dimensions evenly.
            **kwargs: Additional arguments passed to base class.

        Raises:
            ValueError: If mask_ratio is not in (0.0, 1.0).
        """
        super().__init__(num_classes=num_classes, **kwargs)

        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0.0, 1.0), got {mask_ratio}")

        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.mask_type = mask_type
        self.block_size = block_size
        self.patch_size = patch_size
    
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate negative samples by masking portions of input images.

        The masking process involves two steps:
        1. Generate a binary mask based on mask_type (random/block/patch)
        2. Fill masked regions based on mask_mode (zero/noise/shuffle/mean)

        Args:
            images: Input images with shape (B, C, H, W) for image format
                or (B, D) for flattened format. The strategy handles both.
            labels: Class labels with shape (B,). Not used by this strategy
                but required by the interface for compatibility.
            **kwargs: Additional keyword arguments (unused, for compatibility).

        Returns:
            torch.Tensor: Masked negative samples with shape (B, D) where
                D is the flattened image dimension (C * H * W or D).

        Example:
            >>> strategy = MaskingStrategy(mask_ratio=0.3, mask_type='random')
            >>> images = torch.randn(32, 1, 28, 28)  # MNIST batch
            >>> labels = torch.randint(0, 10, (32,))
            >>> negatives = strategy.generate(images, labels)
            >>> print(negatives.shape)  # torch.Size([32, 784])
        """
        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()
        num_features = flat.size(1)

        # Generate mask based on type
        if self.mask_type == 'random':
            mask = self._random_mask(batch_size, num_features, images.device)
        elif self.mask_type == 'block':
            mask = self._block_mask(images)
        elif self.mask_type == 'patch':
            mask = self._patch_mask(images)
        else:
            raise ValueError(
                f"Unknown mask_type: {self.mask_type}. "
                f"Expected one of: 'random', 'block', 'patch'"
            )

        # Apply mask based on mode
        if self.mask_mode == 'zero':
            # Simple zeroing - most common approach
            flat = flat * (1 - mask)

        elif self.mask_mode == 'noise':
            # Replace with Gaussian noise
            noise = torch.randn_like(flat)
            flat = flat * (1 - mask) + noise * mask

        elif self.mask_mode == 'shuffle':
            # Shuffle masked positions with random positions
            perm = torch.randperm(num_features, device=images.device)
            shuffled = flat[:, perm]
            flat = flat * (1 - mask) + shuffled * mask

        elif self.mask_mode == 'mean':
            # Replace with per-image mean value
            mean_val = flat.mean(dim=1, keepdim=True)
            flat = flat * (1 - mask) + mean_val * mask

        else:
            raise ValueError(
                f"Unknown mask_mode: {self.mask_mode}. "
                f"Expected one of: 'zero', 'noise', 'shuffle', 'mean'"
            )

        return flat
    
    def _random_mask(
        self,
        batch_size: int,
        num_features: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate random per-pixel mask (dropout-style).

        Each pixel is independently masked with probability equal to mask_ratio.
        This creates sparse, scattered corruption throughout the image.

        Args:
            batch_size: Number of images in the batch.
            num_features: Number of pixels per image (flattened).
            device: Device to create the mask on.

        Returns:
            torch.Tensor: Binary mask with shape (batch_size, num_features).
                1.0 indicates masked positions, 0.0 indicates kept positions.
        """
        mask = (torch.rand(batch_size, num_features, device=device) < self.mask_ratio).float()
        return mask

    def _block_mask(self, images: torch.Tensor) -> torch.Tensor:
        """
        Generate contiguous block mask (cutout-style).

        Divides the image into a grid of blocks and randomly selects blocks
        to mask based on mask_ratio. Creates localized corruption regions.

        Args:
            images: Input images with shape (B, C, H, W) or (B, D).
                For flattened inputs, assumes square images.

        Returns:
            torch.Tensor: Binary mask with shape (batch_size, num_features).
                1.0 indicates masked positions, 0.0 indicates kept positions.

        Note:
            Falls back to random masking if image dimensions are not compatible
            with block-based masking (e.g., non-square flattened images).
        """
        batch_size = images.size(0)

        if images.dim() == 2:
            # Flattened input - assume square image
            side = int(images.size(1) ** 0.5)
            if side * side != images.size(1):
                # Not square, fall back to random masking
                return self._random_mask(batch_size, images.size(1), images.device)
        else:
            # (B, C, H, W) format
            side = images.size(-1)

        # Calculate grid dimensions
        blocks_per_side = side // self.block_size
        if blocks_per_side == 0:
            # Block size larger than image, fall back to random
            return self._random_mask(batch_size, side * side, images.device)

        num_blocks = blocks_per_side ** 2
        num_mask_blocks = max(1, int(num_blocks * self.mask_ratio))

        mask = torch.zeros(batch_size, side, side, device=images.device)

        for i in range(batch_size):
            # Randomly select blocks to mask
            block_indices = torch.randperm(num_blocks, device=images.device)[:num_mask_blocks]

            for idx in block_indices:
                row = (idx // blocks_per_side) * self.block_size
                col = (idx % blocks_per_side) * self.block_size
                mask[i, row:row + self.block_size, col:col + self.block_size] = 1

        return mask.view(batch_size, -1)

    def _patch_mask(self, images: torch.Tensor) -> torch.Tensor:
        """
        Generate MAE-style patch mask.

        Divides the image into non-overlapping patches and randomly masks
        a subset of patches based on mask_ratio. This is the approach used
        in Masked Autoencoders (MAE) for self-supervised learning.

        Unlike block masking, patch masking:
        - Uses patch_size instead of block_size
        - Covers the entire image with non-overlapping patches
        - Masks exactly mask_ratio fraction of patches

        Args:
            images: Input images with shape (B, C, H, W) or (B, D).
                For flattened inputs, assumes square images.

        Returns:
            torch.Tensor: Binary mask with shape (batch_size, num_features).
                1.0 indicates masked positions, 0.0 indicates kept positions.

        Note:
            Falls back to random masking if image dimensions are not divisible
            by patch_size.
        """
        batch_size = images.size(0)

        if images.dim() == 2:
            # Flattened input - assume square image
            side = int(images.size(1) ** 0.5)
            if side * side != images.size(1):
                return self._random_mask(batch_size, images.size(1), images.device)
        else:
            side = images.size(-1)

        # Check if patch_size divides image dimension
        if side % self.patch_size != 0:
            # Fall back to random masking
            return self._random_mask(batch_size, side * side, images.device)

        patches_per_side = side // self.patch_size
        num_patches = patches_per_side ** 2
        num_mask_patches = max(1, int(num_patches * self.mask_ratio))

        mask = torch.zeros(batch_size, side, side, device=images.device)

        for i in range(batch_size):
            # Randomly select patches to mask
            patch_indices = torch.randperm(num_patches, device=images.device)[:num_mask_patches]

            for idx in patch_indices:
                row = (idx // patches_per_side) * self.patch_size
                col = (idx % patches_per_side) * self.patch_size
                mask[i, row:row + self.patch_size, col:col + self.patch_size] = 1

        return mask.view(batch_size, -1)
    
    @property
    def requires_labels(self) -> bool:
        """
        Whether this strategy requires labels to generate negatives.

        Returns:
            bool: False - masking only modifies images and does not use labels.
        """
        return False

    def get_config(self) -> dict:
        """
        Return strategy configuration for logging and serialization.

        Returns:
            dict: Configuration dictionary containing all strategy parameters.
        """
        config = super().get_config()
        config['mask_ratio'] = self.mask_ratio
        config['mask_mode'] = self.mask_mode
        config['mask_type'] = self.mask_type
        config['block_size'] = self.block_size
        config['patch_size'] = self.patch_size
        return config

    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return (
            f"{self.name}("
            f"mask_ratio={self.mask_ratio}, "
            f"mask_type='{self.mask_type}', "
            f"mask_mode='{self.mask_mode}')"
        )
