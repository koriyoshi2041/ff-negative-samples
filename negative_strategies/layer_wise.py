"""
Layer-Wise Generation Strategy.
Uses the previous layer's output to generate layer-specific negatives.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('layer_wise')
class LayerWiseStrategy(NegativeStrategy):
    """
    Layer-wise negative sample generation.
    
    Based on: "Layer Collaboration in the Forward-Forward Algorithm" (arXiv:2305.12393)
    
    Key idea: Generate layer-specific negative samples using:
    1. Previous layer activations
    2. Learned negative generators per layer
    3. Adaptive noise based on layer statistics
    
    This creates more targeted negative samples that challenge
    each layer appropriately.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        layer_dims: List[int] = None,  # Dimensions for each layer
        use_learned_generator: bool = False,
        perturbation_scale: float = 0.5,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes
            layer_dims: List of dimensions for each layer (for generator setup)
            use_learned_generator: Whether to use learnable generators
            perturbation_scale: Scale of perturbations to apply
        """
        super().__init__(num_classes=num_classes, **kwargs)
        self.layer_dims = layer_dims or [784, 500, 500]
        self.use_learned_generator = use_learned_generator
        self.perturbation_scale = perturbation_scale
        
        # Per-layer generators (if learned)
        self.generators = nn.ModuleList() if use_learned_generator else None
        if use_learned_generator and layer_dims:
            for dim in layer_dims[:-1]:  # No generator for output layer
                self.generators.append(
                    nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.ReLU(),
                        nn.Linear(dim, dim)
                    )
                )
        
        # Statistics for adaptive noise (updated during training)
        self.layer_means: List[Optional[torch.Tensor]] = [None] * len(self.layer_dims)
        self.layer_stds: List[Optional[torch.Tensor]] = [None] * len(self.layer_dims)
        self.current_layer = 0
    
    def set_layer(self, layer_idx: int):
        """Set the current layer for generation."""
        self.current_layer = layer_idx
    
    def update_statistics(
        self, 
        layer_idx: int, 
        activations: torch.Tensor
    ):
        """Update running statistics for a layer."""
        with torch.no_grad():
            mean = activations.mean(dim=0)
            std = activations.std(dim=0) + 1e-8
            
            # Exponential moving average
            alpha = 0.1
            if self.layer_means[layer_idx] is None:
                self.layer_means[layer_idx] = mean
                self.layer_stds[layer_idx] = std
            else:
                self.layer_means[layer_idx] = alpha * mean + (1 - alpha) * self.layer_means[layer_idx]
                self.layer_stds[layer_idx] = alpha * std + (1 - alpha) * self.layer_stds[layer_idx]
    
    def generate(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        layer_idx: int = None,
        prev_activations: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate layer-specific negative samples.
        
        Args:
            images: Input images or activations (B, D)
            labels: Labels (B,) - may be used for some modes
            layer_idx: Which layer to generate for (default: current_layer)
            prev_activations: Activations from previous layer
            
        Returns:
            Negative samples for this layer (B, D)
        """
        layer_idx = layer_idx if layer_idx is not None else self.current_layer
        batch_size = images.size(0)
        
        # Use previous activations if available
        if prev_activations is not None:
            input_data = prev_activations
        else:
            input_data = images.view(batch_size, -1)
        
        if self.use_learned_generator and self.generators and layer_idx < len(self.generators):
            # Use learned generator
            negative = self.generators[layer_idx](input_data)
        else:
            # Use adaptive perturbation
            negative = self._adaptive_perturbation(input_data, layer_idx)
        
        return negative
    
    def _adaptive_perturbation(
        self, 
        data: torch.Tensor, 
        layer_idx: int
    ) -> torch.Tensor:
        """Apply perturbation scaled by layer statistics."""
        batch_size, dim = data.shape
        
        if self.layer_stds[layer_idx] is not None:
            # Use learned statistics
            std = self.layer_stds[layer_idx].to(data.device)
            noise = torch.randn_like(data) * std * self.perturbation_scale
        else:
            # Use data-driven statistics
            std = data.std(dim=1, keepdim=True) + 1e-8
            noise = torch.randn_like(data) * std * self.perturbation_scale
        
        # Option 1: Add noise
        negative = data + noise
        
        # Option 2: Shuffle features (alternative)
        # perm = torch.randperm(dim, device=data.device)
        # negative = data[:, perm]
        
        return negative
    
    def generate_from_activations(
        self,
        activations: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Generate negatives directly from layer activations.
        
        This is the main interface for layer-wise training where
        we want to perturb the actual layer activations.
        """
        batch_size = activations.size(0)
        
        # Multiple perturbation strategies
        strategy = layer_idx % 3
        
        if strategy == 0:
            # Add scaled noise
            std = activations.std(dim=1, keepdim=True) + 1e-8
            noise = torch.randn_like(activations) * std * self.perturbation_scale
            negative = activations + noise
            
        elif strategy == 1:
            # Shuffle within batch
            perm = torch.randperm(batch_size, device=activations.device)
            negative = activations[perm]
            
        else:
            # Mix with shuffled version
            perm = torch.randperm(batch_size, device=activations.device)
            alpha = torch.rand(batch_size, 1, device=activations.device) * 0.5
            negative = (1 - alpha) * activations + alpha * activations[perm]
        
        return negative
    
    @property
    def requires_labels(self) -> bool:
        return False
    
    def get_config(self):
        config = super().get_config()
        config['layer_dims'] = self.layer_dims
        config['use_learned_generator'] = self.use_learned_generator
        config['perturbation_scale'] = self.perturbation_scale
        return config
    
    def train(self, mode: bool = True):
        """Set training mode for learnable components."""
        if self.generators:
            for gen in self.generators:
                gen.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
