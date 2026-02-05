"""
Self-Contrastive Strategy (SCFF).
Based on: "Self-Contrastive Forward-Forward Algorithm" (arXiv:2409.11593)

Key idea from paper:
- Positive sample: x_k + x_k (same image paired with itself)
- Negative sample: x_k + x_n (two different images paired)

The network learns to distinguish "same image pairs" from "different image pairs".
"""

import torch
import torch.nn.functional as F
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('self_contrastive')
class SelfContrastiveStrategy(NegativeStrategy):
    """
    Self-Contrastive Forward-Forward (SCFF).
    
    Based on: "Self-Contrastive Forward-Forward Algorithm" (arXiv:2409.11593)
    
    Key insight from paper:
    - Positive: Concatenation/sum of same image with itself [x_k, x_k] -> W(x_k + x_k)
    - Negative: Concatenation/sum of different images [x_k, x_n] -> W(x_k + x_n)
    
    Since W1 = W2 (proven in paper), we can simplify to:
    - Positive: 2 * x_k (or x_k + augment(x_k))
    - Negative: x_k + x_n (shuffled pairs from batch)
    
    For evaluation, uses label embedding (like Hinton's method) to enable
    fair comparison with other strategies.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        use_augmentation: bool = True,
        noise_std: float = 0.1,
        label_scale: float = 1.0,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes
            use_augmentation: If True, use augmented version for positive pair
            noise_std: Noise level for augmentation
            label_scale: Scale factor for label embedding (used in evaluation)
        """
        super().__init__(num_classes=num_classes, **kwargs)
        self.use_augmentation = use_augmentation
        self.noise_std = noise_std
        self.label_scale = label_scale
        self._training_mode = True  # Default to training mode
    
    def train(self):
        """Set to training mode (use SCFF positive/negative)."""
        self._training_mode = True
        return self
    
    def eval(self):
        """Set to evaluation mode (use label embedding for classification)."""
        self._training_mode = False
        return self
    
    def create_positive(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Create positive samples.
        
        Training mode (SCFF): same image paired with itself
        - Positive = x_k + x_k (or x_k + augment(x_k))
        
        Evaluation mode: label embedding (for classification)
        - Positive = image with label embedded in first pixels
        
        Args:
            images: Input images (B, C, H, W) or (B, D)
            labels: Labels (B,) - used in eval mode for embedding
            
        Returns:
            Positive samples (B, D) - flattened
        """
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)
        
        if self._training_mode:
            # SCFF: x_k + x_k or x_k + augment(x_k)
            if self.use_augmentation:
                augmented = flat + torch.randn_like(flat) * self.noise_std
                return flat + augmented
            else:
                return 2 * flat
        else:
            # Evaluation: use label embedding for classification
            result = flat.clone()
            one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
            one_hot.scatter_(1, labels.unsqueeze(1), self.label_scale)
            result[:, :self.num_classes] = one_hot
            return result
    
    def generate(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate negative samples.
        
        Training mode (SCFF): different images paired together
        - Negative = x_k + x_n where n != k (different images)
        
        Evaluation mode: wrong label embedded (for classification)
        
        Args:
            images: Input images (B, C, H, W) or (B, D)
            labels: Labels (B,) - used in eval mode
            
        Returns:
            Negative samples (B, D) - flattened
        """
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)
        
        if self._training_mode:
            # SCFF: x_k + x_n (different images)
            # Roll by 1 to guarantee different images for each position
            shifted = torch.roll(flat, shifts=1, dims=0)
            return flat + shifted
        else:
            # Evaluation: wrong label embedding
            result = flat.clone()
            wrong_labels = torch.randint(
                0, self.num_classes, (batch_size,), 
                device=images.device
            )
            mask = wrong_labels == labels
            wrong_labels[mask] = (wrong_labels[mask] + 1) % self.num_classes
            
            one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
            one_hot.scatter_(1, wrong_labels.unsqueeze(1), self.label_scale)
            result[:, :self.num_classes] = one_hot
            return result
    
    @property
    def requires_labels(self) -> bool:
        return False  # Self-supervised - no labels needed for training
    
    @property
    def uses_negatives(self) -> bool:
        """SCFF uses negatives (different from mono-forward)."""
        return True
    
    def get_config(self):
        config = super().get_config()
        config['use_augmentation'] = self.use_augmentation
        config['noise_std'] = self.noise_std
        config['label_scale'] = self.label_scale
        return config
