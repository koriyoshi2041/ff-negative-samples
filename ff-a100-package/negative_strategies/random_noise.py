"""
Random Noise Strategy - Simplest baseline.
Uses random noise as negative samples.
"""

import torch
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('random_noise')
class RandomNoiseStrategy(NegativeStrategy):
    """
    Simplest negative sample strategy: pure random noise.
    
    Types of noise:
    - Gaussian: Normal distribution N(mean, std)
    - Uniform: Uniform distribution U(low, high)
    - Matched: Noise matching the statistics of input data
    
    Note: This is the weakest baseline as noise provides
    minimal learning signal. Included for completeness.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        noise_type: str = 'gaussian',
        noise_mean: float = 0.0,
        noise_std: float = 1.0,
        noise_low: float = 0.0,
        noise_high: float = 1.0,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes
            noise_type: 'gaussian', 'uniform', or 'matched'
            noise_mean: Mean for Gaussian noise
            noise_std: Std for Gaussian noise
            noise_low: Lower bound for uniform noise
            noise_high: Upper bound for uniform noise
        """
        super().__init__(num_classes=num_classes, **kwargs)
        self.noise_type = noise_type
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_low = noise_low
        self.noise_high = noise_high
    
    def generate(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate random noise as negative samples.
        
        Args:
            images: Input images (B, C, H, W) or (B, D) - used for shape
            labels: Labels (B,) - not used
            
        Returns:
            Random noise with same shape as flattened input (B, D)
        """
        batch_size = images.size(0)
        flat_shape = (batch_size, images.view(batch_size, -1).size(1))
        
        if self.noise_type == 'gaussian':
            negative = torch.randn(
                flat_shape, 
                device=images.device
            ) * self.noise_std + self.noise_mean
            
        elif self.noise_type == 'uniform':
            negative = torch.rand(
                flat_shape, 
                device=images.device
            ) * (self.noise_high - self.noise_low) + self.noise_low
            
        elif self.noise_type == 'matched':
            # Match the statistics of input data
            flat = images.view(batch_size, -1)
            mean = flat.mean(dim=0, keepdim=True)
            std = flat.std(dim=0, keepdim=True) + 1e-8
            negative = torch.randn(flat_shape, device=images.device) * std + mean
            
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        return negative
    
    @property
    def requires_labels(self) -> bool:
        return False
    
    def get_config(self):
        config = super().get_config()
        config['noise_type'] = self.noise_type
        if self.noise_type == 'gaussian':
            config['noise_mean'] = self.noise_mean
            config['noise_std'] = self.noise_std
        elif self.noise_type == 'uniform':
            config['noise_low'] = self.noise_low
            config['noise_high'] = self.noise_high
        return config
