"""
Class Confusion Strategy.
Uses correct image with wrong label (label-only perturbation).
"""

import torch
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('class_confusion')
class ClassConfusionStrategy(NegativeStrategy):
    """
    Class Confusion: Correct image + Wrong label.
    
    Unlike label embedding, this strategy doesn't modify the image.
    Instead, it pairs unchanged images with wrong labels.
    
    This is useful when the model learns to associate
    features with labels, and we want to break that association
    without corrupting the visual features.
    
    Note: This requires a special handling in the training loop
    where the label information is used separately (not embedded).
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        confusion_mode: str = 'random',  # 'random', 'adjacent', 'hardest'
        temperature: float = 1.0,  # For weighted sampling
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes
            confusion_mode: How to select wrong labels
                - 'random': Uniformly random wrong label
                - 'adjacent': Next class (circular)
                - 'hardest': Most confusing class (requires confusion matrix)
            temperature: Temperature for softmax sampling
        """
        super().__init__(num_classes=num_classes, **kwargs)
        self.confusion_mode = confusion_mode
        self.temperature = temperature
        self.confusion_matrix = None  # Set during training if needed
    
    def generate(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        confusion_matrix: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate negative samples by pairing correct images with wrong labels.
        
        Returns the unchanged flattened image. The wrong labels are returned
        separately via get_wrong_labels() or should be used by the training loop.
        
        Args:
            images: Input images (B, C, H, W) or (B, D)
            labels: Correct labels (B,)
            confusion_matrix: Optional confusion matrix for 'hardest' mode
            
        Returns:
            Unchanged flattened images (B, D) - labels handled separately
        """
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)
        
        # Store wrong labels for external use
        self._last_wrong_labels = self._generate_wrong_labels(
            labels, 
            confusion_matrix
        )
        
        return flat.clone()  # Return unchanged image
    
    def _generate_wrong_labels(
        self, 
        labels: torch.Tensor,
        confusion_matrix: torch.Tensor = None
    ) -> torch.Tensor:
        """Generate wrong labels based on confusion mode."""
        batch_size = labels.size(0)
        device = labels.device
        
        if self.confusion_mode == 'random':
            # Random wrong labels
            wrong_labels = torch.randint(
                0, self.num_classes, (batch_size,), device=device
            )
            mask = wrong_labels == labels
            wrong_labels[mask] = (wrong_labels[mask] + 1) % self.num_classes
            
        elif self.confusion_mode == 'adjacent':
            # Adjacent class (circular)
            wrong_labels = (labels + 1) % self.num_classes
            
        elif self.confusion_mode == 'hardest':
            # Sample from confusion distribution
            if confusion_matrix is None:
                confusion_matrix = self.confusion_matrix
            if confusion_matrix is None:
                # Fallback to random if no confusion matrix
                return self._generate_wrong_labels_fallback(labels, 'random')
            
            # Normalize confusion matrix rows
            probs = confusion_matrix[labels]  # (B, num_classes)
            # Zero out correct class
            mask = torch.arange(self.num_classes, device=device).unsqueeze(0)
            mask = mask == labels.unsqueeze(1)
            probs = probs.masked_fill(mask, 0)
            # Normalize
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
            # Sample
            wrong_labels = torch.multinomial(probs, 1).squeeze(1)
            
        else:
            raise ValueError(f"Unknown confusion mode: {self.confusion_mode}")
        
        return wrong_labels
    
    def get_wrong_labels(self) -> torch.Tensor:
        """Get the wrong labels generated in the last call."""
        if not hasattr(self, '_last_wrong_labels'):
            raise RuntimeError("generate() must be called first")
        return self._last_wrong_labels
    
    def set_confusion_matrix(self, matrix: torch.Tensor):
        """Set confusion matrix for 'hardest' mode."""
        self.confusion_matrix = matrix
    
    @property
    def requires_labels(self) -> bool:
        return True
    
    def get_config(self):
        config = super().get_config()
        config['confusion_mode'] = self.confusion_mode
        config['temperature'] = self.temperature
        return config


@StrategyRegistry.register('class_confusion_embedded')
class ClassConfusionEmbeddedStrategy(ClassConfusionStrategy):
    """
    Variant that embeds the wrong label into the image (like LabelEmbedding).
    Combines class confusion logic with label embedding output format.
    """
    
    def __init__(self, num_classes: int = 10, label_scale: float = 1.0, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.label_scale = label_scale
    
    def create_positive(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Create positive samples with correct label embedded."""
        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()
        
        one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
        one_hot.scatter_(1, labels.unsqueeze(1), self.label_scale)
        flat[:, :self.num_classes] = one_hot
        return flat
    
    def generate(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Embed wrong label selected by confusion mode."""
        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()
        
        wrong_labels = self._generate_wrong_labels(labels, kwargs.get('confusion_matrix'))
        self._last_wrong_labels = wrong_labels
        
        one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
        one_hot.scatter_(1, wrong_labels.unsqueeze(1), self.label_scale)
        flat[:, :self.num_classes] = one_hot
        
        return flat
