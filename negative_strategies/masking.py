"""
Masking Strategy.
Creates negative samples by randomly masking parts of the image.
"""

import torch
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('masking')
class MaskingStrategy(NegativeStrategy):
    """
    Masking: Randomly mask (zero out or corrupt) portions of the image.
    
    Different masking modes:
    - 'zero': Set masked pixels to zero
    - 'noise': Replace masked pixels with random noise
    - 'shuffle': Shuffle masked pixels with other parts of the image
    - 'mean': Replace with mean pixel value
    
    This is a simpler version of cutout augmentation, used as negative samples.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        mask_ratio: float = 0.5,
        mask_mode: str = 'zero',  # 'zero', 'noise', 'shuffle', 'mean'
        mask_type: str = 'random',  # 'random', 'block', 'structured'
        block_size: int = 4,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes
            mask_ratio: Fraction of pixels to mask (0.0 to 1.0)
            mask_mode: What to put in masked positions
            mask_type: How to select positions to mask
            block_size: Size of blocks for block masking
        """
        super().__init__(num_classes=num_classes, **kwargs)
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.mask_type = mask_type
        self.block_size = block_size
    
    def generate(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate masked negative samples.
        
        Args:
            images: Input images (B, C, H, W) or (B, D)
            labels: Labels (B,) - not used
            
        Returns:
            Masked images (B, D)
        """
        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()
        num_features = flat.size(1)
        
        # Generate mask based on type
        if self.mask_type == 'random':
            mask = self._random_mask(batch_size, num_features, images.device)
        elif self.mask_type == 'block':
            mask = self._block_mask(images)
        elif self.mask_type == 'structured':
            mask = self._structured_mask(batch_size, num_features, images.device)
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")
        
        # Apply mask based on mode
        if self.mask_mode == 'zero':
            flat = flat * (1 - mask)
            
        elif self.mask_mode == 'noise':
            noise = torch.randn_like(flat)
            flat = flat * (1 - mask) + noise * mask
            
        elif self.mask_mode == 'shuffle':
            # Shuffle positions
            perm = torch.randperm(num_features, device=images.device)
            shuffled = flat[:, perm]
            flat = flat * (1 - mask) + shuffled * mask
            
        elif self.mask_mode == 'mean':
            mean_val = flat.mean(dim=1, keepdim=True)
            flat = flat * (1 - mask) + mean_val * mask
            
        else:
            raise ValueError(f"Unknown mask mode: {self.mask_mode}")
        
        return flat
    
    def _random_mask(
        self, 
        batch_size: int, 
        num_features: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Generate random pixel mask."""
        mask = (torch.rand(batch_size, num_features, device=device) < self.mask_ratio).float()
        return mask
    
    def _block_mask(self, images: torch.Tensor) -> torch.Tensor:
        """Generate block-wise mask."""
        batch_size = images.size(0)
        
        if images.dim() == 2:
            # Flattened, assume square
            side = int(images.size(1) ** 0.5)
            if side * side != images.size(1):
                # Not square, fallback to random
                return self._random_mask(batch_size, images.size(1), images.device)
        else:
            side = images.size(-1)
        
        # Create block mask
        num_blocks = (side // self.block_size) ** 2
        num_mask_blocks = int(num_blocks * self.mask_ratio)
        
        mask = torch.zeros(batch_size, side, side, device=images.device)
        
        for i in range(batch_size):
            # Randomly select blocks to mask
            block_indices = torch.randperm(num_blocks, device=images.device)[:num_mask_blocks]
            blocks_per_row = side // self.block_size
            
            for idx in block_indices:
                row = (idx // blocks_per_row) * self.block_size
                col = (idx % blocks_per_row) * self.block_size
                mask[i, row:row+self.block_size, col:col+self.block_size] = 1
        
        return mask.view(batch_size, -1)
    
    def _structured_mask(
        self, 
        batch_size: int, 
        num_features: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Generate structured mask (rows or columns for images)."""
        side = int(num_features ** 0.5)
        if side * side != num_features:
            return self._random_mask(batch_size, num_features, device)
        
        mask = torch.zeros(batch_size, side, side, device=device)
        num_lines = int(side * self.mask_ratio)
        
        for i in range(batch_size):
            # Randomly choose rows or columns
            if torch.rand(1).item() > 0.5:
                # Mask rows
                rows = torch.randperm(side, device=device)[:num_lines]
                mask[i, rows, :] = 1
            else:
                # Mask columns
                cols = torch.randperm(side, device=device)[:num_lines]
                mask[i, :, cols] = 1
        
        return mask.view(batch_size, -1)
    
    @property
    def requires_labels(self) -> bool:
        return False
    
    def get_config(self):
        config = super().get_config()
        config['mask_ratio'] = self.mask_ratio
        config['mask_mode'] = self.mask_mode
        config['mask_type'] = self.mask_type
        config['block_size'] = self.block_size
        return config
