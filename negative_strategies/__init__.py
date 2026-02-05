"""
Negative Sample Strategies for Forward-Forward Algorithm.

This module provides a collection of strategies for generating
negative samples in FF training, along with a unified interface
and registry for easy experimentation.

Strategies:
    1. LabelEmbedding - Hinton's original method
    2. ImageMixing - Mix two different images
    3. RandomNoise - Pure random noise baseline
    4. ClassConfusion - Correct image + wrong label
    5. SelfContrastive - Strong augmentation as negatives
    6. Masking - Random masking of pixels
    7. LayerWise - Layer-specific generation
    8. Adversarial - Gradient-based perturbation
    9. HardMining - Select hardest negatives
    10. MonoForward - No negatives variant

Usage:
    from negative_strategies import StrategyRegistry, LabelEmbeddingStrategy
    
    # Create by name
    strategy = StrategyRegistry.create('label_embedding', num_classes=10)
    
    # Or directly
    strategy = LabelEmbeddingStrategy(num_classes=10)
    
    # Generate negatives
    neg_samples = strategy.generate(images, labels)
"""

from .base import NegativeStrategy, StrategyRegistry

# Import all strategies to register them
from .label_embedding import LabelEmbeddingStrategy
from .image_mixing import ImageMixingStrategy, ClassAwareMixingStrategy
from .random_noise import RandomNoiseStrategy
from .class_confusion import ClassConfusionStrategy, ClassConfusionEmbeddedStrategy
from .self_contrastive import SelfContrastiveStrategy
from .masking import MaskingStrategy
from .layer_wise import LayerWiseStrategy
from .adversarial import AdversarialStrategy, FastAdversarialStrategy
from .hard_mining import HardMiningStrategy
from .mono_forward import MonoForwardStrategy, EnergyMinimizationStrategy

# Convenience exports
__all__ = [
    # Base
    'NegativeStrategy',
    'StrategyRegistry',
    
    # Strategies
    'LabelEmbeddingStrategy',
    'ImageMixingStrategy',
    'ClassAwareMixingStrategy',
    'RandomNoiseStrategy',
    'ClassConfusionStrategy',
    'ClassConfusionEmbeddedStrategy',
    'SelfContrastiveStrategy',
    'MaskingStrategy',
    'LayerWiseStrategy',
    'AdversarialStrategy',
    'FastAdversarialStrategy',
    'HardMiningStrategy',
    'MonoForwardStrategy',
    'EnergyMinimizationStrategy',
]

# Strategy name mapping for documentation
STRATEGY_INFO = {
    'label_embedding': {
        'class': LabelEmbeddingStrategy,
        'requires_labels': True,
        'description': "Hinton's original - embed wrong label in first pixels",
    },
    'image_mixing': {
        'class': ImageMixingStrategy,
        'requires_labels': False,
        'description': 'Mix two different images pixel-wise',
    },
    'class_aware_mixing': {
        'class': ClassAwareMixingStrategy,
        'requires_labels': True,
        'description': 'Mix images ensuring different classes',
    },
    'random_noise': {
        'class': RandomNoiseStrategy,
        'requires_labels': False,
        'description': 'Pure random noise baseline',
    },
    'class_confusion': {
        'class': ClassConfusionStrategy,
        'requires_labels': True,
        'description': 'Correct image with wrong label',
    },
    'class_confusion_embedded': {
        'class': ClassConfusionEmbeddedStrategy,
        'requires_labels': True,
        'description': 'Class confusion with label embedding',
    },
    'self_contrastive': {
        'class': SelfContrastiveStrategy,
        'requires_labels': False,
        'description': 'Strong augmentation as negatives (SCFF)',
    },
    'masking': {
        'class': MaskingStrategy,
        'requires_labels': False,
        'description': 'Random masking/corruption of pixels',
    },
    'layer_wise': {
        'class': LayerWiseStrategy,
        'requires_labels': False,
        'description': 'Layer-specific adaptive generation',
    },
    'adversarial': {
        'class': AdversarialStrategy,
        'requires_labels': False,
        'description': 'Gradient-based adversarial perturbation',
    },
    'fast_adversarial': {
        'class': FastAdversarialStrategy,
        'requires_labels': False,
        'description': 'Single-step FGSM adversarial',
    },
    'hard_mining': {
        'class': HardMiningStrategy,
        'requires_labels': True,
        'description': 'Select hardest negatives from pool',
    },
    'mono_forward': {
        'class': MonoForwardStrategy,
        'requires_labels': True,
        'description': 'No negatives - alternative loss',
    },
    'energy_minimization': {
        'class': EnergyMinimizationStrategy,
        'requires_labels': True,
        'description': 'Energy-based training without negatives',
    },
}


def list_strategies():
    """List all available strategies with descriptions."""
    print("Available Negative Sample Strategies:")
    print("=" * 60)
    for name, info in STRATEGY_INFO.items():
        label_req = "✓" if info['requires_labels'] else "✗"
        print(f"  {name:<25} [labels:{label_req}] {info['description']}")
    print("=" * 60)


def create_strategy(name: str, **kwargs) -> NegativeStrategy:
    """
    Create a strategy by name.
    
    Args:
        name: Strategy name (see list_strategies())
        **kwargs: Strategy-specific arguments
        
    Returns:
        Initialized strategy instance
    """
    return StrategyRegistry.create(name, **kwargs)
