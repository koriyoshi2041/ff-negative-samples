"""
Mono-Forward Strategy.
No negative samples - uses alternative loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('mono_forward')
class MonoForwardStrategy(NegativeStrategy):
    """
    Mono-Forward: Training without explicit negative samples.
    
    Based on: "Mono-Forward: Backpropagation-Free Algorithm for Efficient 
    Neural Network Training" (arXiv:2501.08756)
    
    Key idea: Instead of contrasting positive vs negative samples,
    use alternative objectives:
    1. Local error signals based on target activations
    2. Energy minimization without negatives
    3. Direct goodness targets
    
    This returns the positive sample unchanged and relies on
    a modified training loop that doesn't use negatives.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        target_goodness: float = 2.0,  # Target goodness value
        loss_type: str = 'mse',  # 'mse', 'hinge', 'softplus'
        use_class_targets: bool = True,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes
            target_goodness: Target goodness value for positive samples
            loss_type: Type of loss to use
            use_class_targets: Whether to use class-specific targets
        """
        super().__init__(num_classes=num_classes, **kwargs)
        self.target_goodness = target_goodness
        self.loss_type = loss_type
        self.use_class_targets = use_class_targets
        
        # Class-specific target patterns (optional)
        self.class_targets: Optional[torch.Tensor] = None
    
    def generate(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        'Generate' negative samples - returns empty/dummy tensor.
        
        Mono-forward doesn't use negatives, so this returns zeros
        or None. The training loop should be modified to not use
        the returned value.
        
        Args:
            images: Input images (B, ...)
            labels: Labels (B,)
            
        Returns:
            Dummy tensor (zeros) or the positive sample itself
        """
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)
        
        # Option 1: Return zeros (dummy)
        # return torch.zeros_like(flat)
        
        # Option 2: Return the positive sample itself (no contrast)
        return flat.clone()
    
    def create_positive(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Create positive samples with optional label embedding."""
        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()
        
        if self.use_class_targets:
            # Embed label information
            one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            flat[:, :self.num_classes] = one_hot
        
        return flat
    
    def compute_mono_loss(
        self, 
        activations: torch.Tensor, 
        labels: torch.Tensor,
        goodness_fn: Callable = None
    ) -> torch.Tensor:
        """
        Compute mono-forward loss (without negatives).
        
        Args:
            activations: Layer activations (B, D)
            labels: Class labels (B,)
            goodness_fn: Function to compute goodness
            
        Returns:
            Loss value
        """
        goodness_fn = goodness_fn or (lambda x: (x ** 2).sum(dim=1))
        goodness = goodness_fn(activations)
        
        if self.loss_type == 'mse':
            # MSE to target goodness
            loss = F.mse_loss(goodness, 
                             torch.full_like(goodness, self.target_goodness))
            
        elif self.loss_type == 'hinge':
            # Hinge loss: encourage goodness above threshold
            loss = F.relu(self.target_goodness - goodness).mean()
            
        elif self.loss_type == 'softplus':
            # Softplus loss (smooth version of hinge)
            loss = F.softplus(self.target_goodness - goodness).mean()
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def compute_class_aware_loss(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        num_activations: int = None
    ) -> torch.Tensor:
        """
        Compute class-aware mono-forward loss.
        
        Different classes should have different target patterns
        to enable classification.
        """
        batch_size = activations.size(0)
        dim = activations.size(1)
        
        if self.class_targets is None or self.class_targets.size(1) != dim:
            # Initialize random class targets
            self.class_targets = torch.randn(
                self.num_classes, dim, 
                device=activations.device
            ) * 2  # Scale to be around target_goodness
        
        # Get target for each sample
        targets = self.class_targets[labels]  # (B, D)
        
        # MSE loss to class-specific target
        loss = F.mse_loss(activations, targets)
        
        return loss
    
    def get_training_config(self):
        """
        Return configuration for mono-forward training loop.
        
        The training loop should use this instead of standard FF:
        1. Only forward pass positives
        2. Use mono_loss instead of FF contrastive loss
        3. No negative samples needed
        """
        return {
            'use_negatives': False,
            'loss_fn': self.compute_mono_loss,
            'target_goodness': self.target_goodness,
            'loss_type': self.loss_type,
        }
    
    @property
    def requires_labels(self) -> bool:
        return True  # Mono-forward typically needs labels for targets
    
    @property
    def uses_negatives(self) -> bool:
        """Mono-forward doesn't use negative samples."""
        return False
    
    def get_config(self):
        config = super().get_config()
        config['target_goodness'] = self.target_goodness
        config['loss_type'] = self.loss_type
        config['use_class_targets'] = self.use_class_targets
        config['uses_negatives'] = False
        return config


@StrategyRegistry.register('energy_minimization')
class EnergyMinimizationStrategy(MonoForwardStrategy):
    """
    Variant using energy-based approach without negatives.
    
    Instead of maximizing/minimizing goodness, directly minimize
    an energy function.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        energy_type: str = 'entropy',  # 'entropy', 'variance', 'sparsity'
        **kwargs
    ):
        super().__init__(num_classes=num_classes, **kwargs)
        self.energy_type = energy_type
    
    def compute_mono_loss(
        self, 
        activations: torch.Tensor, 
        labels: torch.Tensor,
        goodness_fn: Callable = None
    ) -> torch.Tensor:
        """Compute energy-based loss."""
        
        if self.energy_type == 'entropy':
            # Minimize entropy (encourage confident predictions)
            probs = F.softmax(activations, dim=1)
            loss = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            
        elif self.energy_type == 'variance':
            # Maximize variance (encourage diverse activations)
            loss = -activations.var(dim=1).mean()
            
        elif self.energy_type == 'sparsity':
            # Encourage sparse activations
            loss = activations.abs().mean()
            
        else:
            raise ValueError(f"Unknown energy type: {self.energy_type}")
        
        return loss
    
    def get_config(self):
        config = super().get_config()
        config['energy_type'] = self.energy_type
        return config
