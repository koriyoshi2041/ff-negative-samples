"""
Hard Mining Strategy.
Selects or generates the most difficult negative samples.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, List
from .base import NegativeStrategy, StrategyRegistry


@StrategyRegistry.register('hard_mining')
class HardMiningStrategy(NegativeStrategy):
    """
    Hard Negative Mining: Select or generate challenging negatives.
    
    Methods:
    1. Goodness-based: Select negatives with highest goodness scores
    2. Distance-based: Select negatives closest to decision boundary
    3. Class-based: Use samples from most confusing classes
    4. Feature-based: Select based on feature similarity
    
    This strategy maintains a pool of candidates and selects
    the hardest ones for training.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        mining_mode: str = 'goodness',  # 'goodness', 'distance', 'class', 'feature'
        pool_size: int = 128,  # Size of candidate pool
        top_k_ratio: float = 0.5,  # Select top K% hardest
        use_memory_bank: bool = False,
        memory_size: int = 1024,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes
            mining_mode: How to determine hardness
            pool_size: Number of candidates to generate
            top_k_ratio: Ratio of hardest samples to select
            use_memory_bank: Whether to maintain a memory bank of negatives
            memory_size: Size of memory bank
        """
        super().__init__(num_classes=num_classes, **kwargs)
        self.mining_mode = mining_mode
        self.pool_size = pool_size
        self.top_k_ratio = top_k_ratio
        self.use_memory_bank = use_memory_bank
        self.memory_size = memory_size
        
        # Memory bank for hard negatives
        self.memory_bank: Optional[torch.Tensor] = None
        self.memory_labels: Optional[torch.Tensor] = None
        self.memory_ptr = 0
        
        # Reference to model
        self._model = None
        self._goodness_fn = None
    
    def set_model(self, model: nn.Module, goodness_fn: Callable = None):
        """Set model for hardness computation."""
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
        Generate hard negative samples.
        
        Args:
            images: Input images (B, ...)
            labels: Labels (B,)
            model: Optional model for hardness computation
            
        Returns:
            Hard negative samples (B, D)
        """
        model = model or self._model
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)
        
        if self.mining_mode == 'goodness':
            return self._goodness_mining(flat, labels, model)
        elif self.mining_mode == 'distance':
            return self._distance_mining(flat, labels, model)
        elif self.mining_mode == 'class':
            return self._class_mining(flat, labels)
        elif self.mining_mode == 'feature':
            return self._feature_mining(flat, labels, model)
        else:
            raise ValueError(f"Unknown mining mode: {self.mining_mode}")
    
    def _goodness_mining(
        self, 
        flat: torch.Tensor, 
        labels: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """Select negatives with highest goodness scores."""
        batch_size = flat.size(0)
        
        # Generate candidate pool
        candidates = self._generate_candidates(flat, labels)  # (pool_size, D)
        
        if model is None:
            # No model, return random from candidates
            idx = torch.randperm(candidates.size(0), device=flat.device)[:batch_size]
            return candidates[idx]
        
        # Compute goodness for all candidates
        with torch.no_grad():
            output = model(candidates)
            if isinstance(output, list):
                output = output[-1]
            goodness = self._goodness_fn(output)
        
        # Select top-k hardest (highest goodness = hardest to distinguish from positive)
        k = int(self.pool_size * self.top_k_ratio)
        _, top_indices = torch.topk(goodness, k=min(k, batch_size))
        
        # Sample from top-k to get batch_size negatives
        if k >= batch_size:
            selected = candidates[top_indices[:batch_size]]
        else:
            # Repeat if needed
            repeats = (batch_size // k) + 1
            idx = top_indices.repeat(repeats)[:batch_size]
            selected = candidates[idx]
        
        return selected
    
    def _distance_mining(
        self, 
        flat: torch.Tensor, 
        labels: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """Select negatives closest to positive samples in feature space."""
        batch_size = flat.size(0)
        candidates = self._generate_candidates(flat, labels)
        
        if model is None:
            # Use input space distance
            pos_features = flat
            neg_features = candidates
        else:
            # Use feature space distance
            with torch.no_grad():
                pos_output = model(flat)
                neg_output = model(candidates)
                if isinstance(pos_output, list):
                    pos_features = pos_output[-1]
                    neg_features = neg_output[-1]
                else:
                    pos_features = pos_output
                    neg_features = neg_output
        
        # Compute pairwise distances
        # For efficiency, compute distance to mean positive
        pos_mean = pos_features.mean(dim=0, keepdim=True)
        distances = ((neg_features - pos_mean) ** 2).sum(dim=1)
        
        # Select closest (hardest)
        k = min(batch_size, candidates.size(0))
        _, closest_indices = torch.topk(distances, k=k, largest=False)
        
        selected = candidates[closest_indices]
        
        # Pad if needed
        if selected.size(0) < batch_size:
            repeats = (batch_size // selected.size(0)) + 1
            selected = selected.repeat(repeats, 1)[:batch_size]
        
        return selected
    
    def _class_mining(
        self, 
        flat: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Use samples from similar/confusing classes."""
        batch_size = flat.size(0)
        
        # Strategy: Shuffle within batches such that different labels are paired
        # but prefer similar classes
        
        # Simple version: roll by 1 to get adjacent samples
        # More sophisticated: use confusion matrix
        
        perm = torch.roll(torch.arange(batch_size, device=flat.device), 1)
        
        # Ensure different labels
        attempts = 0
        while (labels[perm] == labels).any() and attempts < 5:
            same_mask = labels[perm] == labels
            new_perm = torch.randperm(batch_size, device=flat.device)
            perm[same_mask] = new_perm[:same_mask.sum()]
            attempts += 1
        
        # Mix original with shuffled (to create "confusing" samples)
        alpha = 0.3  # Keep most of original but add confusion
        negative = (1 - alpha) * flat + alpha * flat[perm]
        
        return negative
    
    def _feature_mining(
        self, 
        flat: torch.Tensor, 
        labels: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """Mine based on feature similarity."""
        batch_size = flat.size(0)
        
        if model is None:
            return self._class_mining(flat, labels)
        
        with torch.no_grad():
            output = model(flat)
            if isinstance(output, list):
                features = output[-1]
            else:
                features = output
        
        # Compute pairwise cosine similarity
        features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        similarity = torch.mm(features_norm, features_norm.t())
        
        # Zero out self-similarity
        similarity.fill_diagonal_(-float('inf'))
        
        # For each sample, find most similar sample with different label
        negative_indices = []
        for i in range(batch_size):
            # Mask same-label samples
            same_label_mask = labels == labels[i]
            sim_row = similarity[i].clone()
            sim_row[same_label_mask] = -float('inf')
            
            # Select most similar with different label
            if (sim_row > -float('inf')).any():
                neg_idx = sim_row.argmax().item()
            else:
                # Fallback to random
                neg_idx = (i + 1) % batch_size
            negative_indices.append(neg_idx)
        
        negative_indices = torch.tensor(negative_indices, device=flat.device)
        return flat[negative_indices]
    
    def _generate_candidates(
        self, 
        flat: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Generate a pool of negative candidates."""
        batch_size = flat.size(0)
        dim = flat.size(1)
        
        candidates = []
        
        # Method 1: Shuffled pairs
        for _ in range(self.pool_size // batch_size + 1):
            perm = torch.randperm(batch_size, device=flat.device)
            alpha = torch.rand(batch_size, 1, device=flat.device)
            mixed = alpha * flat + (1 - alpha) * flat[perm]
            candidates.append(mixed)
        
        candidates = torch.cat(candidates, dim=0)[:self.pool_size]
        
        # Add memory bank samples if available
        if self.use_memory_bank and self.memory_bank is not None:
            candidates = torch.cat([candidates, self.memory_bank], dim=0)
        
        return candidates
    
    def update_memory_bank(self, negatives: torch.Tensor, labels: torch.Tensor = None):
        """Update memory bank with new hard negatives."""
        if not self.use_memory_bank:
            return
        
        batch_size = negatives.size(0)
        dim = negatives.size(1)
        
        if self.memory_bank is None:
            self.memory_bank = torch.zeros(
                self.memory_size, dim, 
                device=negatives.device
            )
            if labels is not None:
                self.memory_labels = torch.zeros(
                    self.memory_size, 
                    dtype=torch.long,
                    device=negatives.device
                )
        
        # Update memory bank (FIFO)
        end_ptr = min(self.memory_ptr + batch_size, self.memory_size)
        actual_batch = end_ptr - self.memory_ptr
        
        self.memory_bank[self.memory_ptr:end_ptr] = negatives[:actual_batch].detach()
        if labels is not None and self.memory_labels is not None:
            self.memory_labels[self.memory_ptr:end_ptr] = labels[:actual_batch].detach()
        
        self.memory_ptr = end_ptr % self.memory_size
    
    @property
    def requires_labels(self) -> bool:
        return True  # Most mining modes benefit from labels
    
    def get_config(self):
        config = super().get_config()
        config['mining_mode'] = self.mining_mode
        config['pool_size'] = self.pool_size
        config['top_k_ratio'] = self.top_k_ratio
        config['use_memory_bank'] = self.use_memory_bank
        config['memory_size'] = self.memory_size
        return config
