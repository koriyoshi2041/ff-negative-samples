"""
Adversarial Strategy.
Generates negative samples by perturbing along gradient direction.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('adversarial')
class AdversarialStrategy(NegativeStrategy):
    """
    Adversarial negative samples: perturb input along gradient direction.
    
    Key idea: Create negatives that are maximally hard to distinguish
    from positives by moving in the direction that increases "positiveness".
    
    Methods:
    1. FGSM-style: Single step in gradient direction
    2. PGD-style: Multiple steps with projection
    3. Random-start: Add random noise before gradient steps
    
    Note: This requires gradient computation, which adds overhead
    and may conflict with FF's "no backprop" philosophy. However,
    the gradient is only used for negative generation, not training.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        epsilon: float = 0.1,  # Perturbation magnitude
        alpha: float = 0.01,  # Step size for iterative methods
        num_steps: int = 1,  # Number of PGD steps
        random_start: bool = False,
        perturbation_norm: str = 'linf',  # 'linf', 'l2'
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes
            epsilon: Maximum perturbation magnitude
            alpha: Step size for PGD
            num_steps: Number of PGD iterations
            random_start: Whether to start from random perturbation
            perturbation_norm: Norm to use for perturbation
        """
        super().__init__(num_classes=num_classes, **kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        self.perturbation_norm = perturbation_norm
        
        # Reference to model for gradient computation
        self._model = None
        self._goodness_fn = None
    
    def set_model(self, model: nn.Module, goodness_fn: Callable = None):
        """
        Set the model for gradient computation.
        
        Args:
            model: FF model or layer
            goodness_fn: Function to compute goodness (default: squared sum)
        """
        self._model = model
        self._goodness_fn = goodness_fn or (lambda x: (x ** 2).sum(dim=1))
    
    def generate(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        model: nn.Module = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate adversarial negative samples.
        
        Args:
            images: Input images (B, ...) 
            labels: Labels (B,) - not used directly
            model: Optional model override
            
        Returns:
            Adversarially perturbed samples (B, D)
        """
        model = model or self._model
        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()
        
        if model is None:
            # Fallback: random perturbation if no model
            return self._random_perturbation(flat)
        
        # Adversarial generation
        flat.requires_grad_(True)
        
        if self.random_start:
            # Random initialization
            noise = torch.rand_like(flat) * 2 - 1  # [-1, 1]
            flat = flat + noise * self.epsilon
            flat = flat.clamp(-3, 3)  # Clamp to reasonable range
        
        # PGD-style iteration
        for _ in range(self.num_steps):
            # Forward pass to get goodness
            output = model(flat) if callable(model) else model.forward(flat)
            if isinstance(output, list):
                output = output[-1]  # Use last layer output
            
            goodness = self._goodness_fn(output)
            
            # Compute gradient of goodness w.r.t. input
            grad = torch.autograd.grad(
                goodness.sum(), 
                flat, 
                create_graph=False
            )[0]
            
            # Move in gradient direction (to maximize goodness -> harder negative)
            # Or move against gradient (to minimize goodness -> easier negative)
            # Here we maximize to create "hard" negatives
            if self.perturbation_norm == 'linf':
                flat = flat + self.alpha * grad.sign()
                # Project back to epsilon ball
                flat = torch.clamp(
                    flat, 
                    images.view(batch_size, -1) - self.epsilon,
                    images.view(batch_size, -1) + self.epsilon
                )
            else:  # L2
                grad_norm = grad.norm(dim=1, keepdim=True) + 1e-8
                flat = flat + self.alpha * grad / grad_norm
                # Project back to epsilon ball
                delta = flat - images.view(batch_size, -1)
                delta_norm = delta.norm(dim=1, keepdim=True)
                factor = torch.min(
                    torch.ones_like(delta_norm),
                    self.epsilon / (delta_norm + 1e-8)
                )
                flat = images.view(batch_size, -1) + delta * factor
            
            flat = flat.detach().requires_grad_(True)
        
        return flat.detach()
    
    def _random_perturbation(self, flat: torch.Tensor) -> torch.Tensor:
        """Fallback random perturbation when no model available."""
        if self.perturbation_norm == 'linf':
            noise = (torch.rand_like(flat) * 2 - 1) * self.epsilon
        else:
            noise = torch.randn_like(flat)
            noise = noise / (noise.norm(dim=1, keepdim=True) + 1e-8) * self.epsilon
        return flat + noise
    
    def generate_easy_negatives(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate easy negatives (move AGAINST gradient to minimize goodness).
        
        These are negatives that are very different from positives.
        """
        model = model or self._model
        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()
        
        if model is None:
            return self._random_perturbation(flat)
        
        flat.requires_grad_(True)
        
        output = model(flat) if callable(model) else model.forward(flat)
        if isinstance(output, list):
            output = output[-1]
        
        goodness = self._goodness_fn(output)
        grad = torch.autograd.grad(goodness.sum(), flat, create_graph=False)[0]
        
        # Move AGAINST gradient to minimize goodness
        if self.perturbation_norm == 'linf':
            flat = flat - self.epsilon * grad.sign()
        else:
            grad_norm = grad.norm(dim=1, keepdim=True) + 1e-8
            flat = flat - self.epsilon * grad / grad_norm
        
        return flat.detach()
    
    @property
    def requires_labels(self) -> bool:
        return False
    
    def get_config(self):
        config = super().get_config()
        config['epsilon'] = self.epsilon
        config['alpha'] = self.alpha
        config['num_steps'] = self.num_steps
        config['random_start'] = self.random_start
        config['perturbation_norm'] = self.perturbation_norm
        return config


@StrategyRegistry.register('fast_adversarial')
class FastAdversarialStrategy(AdversarialStrategy):
    """
    Fast single-step adversarial (FGSM-style) for efficiency.
    """
    
    def __init__(self, num_classes: int = 10, epsilon: float = 0.1, **kwargs):
        super().__init__(
            num_classes=num_classes,
            epsilon=epsilon,
            num_steps=1,
            random_start=False,
            **kwargs
        )
