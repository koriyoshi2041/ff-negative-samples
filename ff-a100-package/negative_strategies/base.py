"""
Base class for negative sample strategies in Forward-Forward algorithm.
"""

from abc import ABC, abstractmethod
import torch
from typing import Optional, Tuple, Dict, Any


class NegativeStrategy(ABC):
    """
    Abstract base class for negative sample generation strategies.
    
    All strategies must implement:
    - generate(): Create negative samples from positive data
    - requires_labels: Property indicating if labels are needed
    """
    
    def __init__(self, num_classes: int = 10, device: Optional[torch.device] = None):
        """
        Initialize the strategy.
        
        Args:
            num_classes: Number of classes in the dataset
            device: Device to create tensors on
        """
        self.num_classes = num_classes
        self.device = device or torch.device('cpu')
    
    @abstractmethod
    def generate(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate negative samples.
        
        Args:
            images: Input images, shape (B, ...) - flattened or not
            labels: Class labels, shape (B,)
            **kwargs: Strategy-specific arguments
            
        Returns:
            negative_samples: Generated negative samples, same shape as input
        """
        pass
    
    def create_positive(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Create positive samples (default: return flattened input).
        Override if strategy needs special positive sample handling.
        
        Args:
            images: Input images
            labels: Class labels
            
        Returns:
            positive_samples: Processed positive samples
        """
        return images.view(images.size(0), -1)
    
    @property
    @abstractmethod
    def requires_labels(self) -> bool:
        """Whether this strategy requires labels to generate negatives."""
        pass
    
    @property
    def name(self) -> str:
        """Strategy name for logging."""
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """Return strategy configuration for logging."""
        return {
            'name': self.name,
            'num_classes': self.num_classes,
            'requires_labels': self.requires_labels,
        }
    
    def to(self, device: torch.device) -> 'NegativeStrategy':
        """Move strategy to device."""
        self.device = device
        return self
    
    def __repr__(self) -> str:
        return f"{self.name}(num_classes={self.num_classes})"


class StrategyRegistry:
    """Registry for negative sample strategies."""
    
    _strategies: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy."""
        def decorator(strategy_cls: type):
            cls._strategies[name] = strategy_cls
            return strategy_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get a strategy class by name."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name]
    
    @classmethod
    def create(cls, name: str, **kwargs) -> NegativeStrategy:
        """Create a strategy instance by name."""
        return cls.get(name)(**kwargs)
    
    @classmethod
    def list_strategies(cls) -> list:
        """List all registered strategies."""
        return list(cls._strategies.keys())
