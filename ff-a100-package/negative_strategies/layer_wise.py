"""
Layer-Wise Negative Sample Strategy.

Adapts negative generation based on the current layer being trained.
Early layers receive simple perturbations while deep layers receive
more complex semantic corruptions.

This follows the intuition that:
- Early layers learn low-level features (edges, textures) -> simple noise suffices
- Middle layers learn mid-level patterns -> structural perturbations
- Deep layers learn high-level semantics -> semantic corruptions needed
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from .base import NegativeStrategy, StrategyRegistry


class PerturbationType(Enum):
    """Types of perturbations available for layer-wise generation."""
    GAUSSIAN_NOISE = "gaussian_noise"
    UNIFORM_NOISE = "uniform_noise"
    PIXEL_SHUFFLE = "pixel_shuffle"
    RANDOM_MASK = "random_mask"
    BLOCK_MASK = "block_mask"
    FEATURE_DROPOUT = "feature_dropout"
    INTERPOLATE_MIX = "interpolate_mix"
    CUTMIX = "cutmix"
    BATCH_SHUFFLE = "batch_shuffle"
    LABEL_CORRUPTION = "label_corruption"
    ADVERSARIAL_DIRECTION = "adversarial_direction"
    MANIFOLD_MIX = "manifold_mix"


@dataclass
class LayerConfig:
    """
    Configuration for a specific layer's negative generation.

    Attributes:
        perturbation_types: List of perturbation types to apply (sampled randomly)
        intensity: Strength of perturbation (0.0 to 1.0)
        combine_mode: How to combine multiple perturbations ('random', 'sequential', 'weighted')
        custom_params: Additional parameters for specific perturbation types
    """
    perturbation_types: List[PerturbationType] = field(
        default_factory=lambda: [PerturbationType.GAUSSIAN_NOISE]
    )
    intensity: float = 0.3
    combine_mode: str = "random"  # 'random', 'sequential', 'weighted'
    custom_params: Dict[str, Any] = field(default_factory=dict)


def get_default_layer_configs(num_layers: int) -> Dict[int, LayerConfig]:
    """
    Generate default layer configurations based on layer depth.

    Strategy progression:
    - Layer 0-1 (Early): Simple noise-based perturbations
    - Layer 2-3 (Middle): Structural perturbations (masking, dropout)
    - Layer 4+ (Deep): Semantic corruptions (mixing, label confusion)

    Args:
        num_layers: Total number of layers in the network

    Returns:
        Dictionary mapping layer index to LayerConfig
    """
    configs = {}

    for i in range(num_layers):
        depth_ratio = i / max(num_layers - 1, 1)  # 0.0 to 1.0

        if depth_ratio < 0.33:
            # Early layers: simple perturbations
            configs[i] = LayerConfig(
                perturbation_types=[
                    PerturbationType.GAUSSIAN_NOISE,
                    PerturbationType.UNIFORM_NOISE,
                    PerturbationType.PIXEL_SHUFFLE,
                ],
                intensity=0.2 + 0.1 * depth_ratio,  # 0.2 to 0.23
                combine_mode="random",
            )
        elif depth_ratio < 0.66:
            # Middle layers: structural perturbations
            configs[i] = LayerConfig(
                perturbation_types=[
                    PerturbationType.RANDOM_MASK,
                    PerturbationType.BLOCK_MASK,
                    PerturbationType.FEATURE_DROPOUT,
                    PerturbationType.PIXEL_SHUFFLE,
                ],
                intensity=0.3 + 0.2 * (depth_ratio - 0.33),  # 0.3 to 0.43
                combine_mode="random",
                custom_params={
                    "mask_ratio": 0.3,
                    "block_size": 4,
                    "dropout_rate": 0.3,
                },
            )
        else:
            # Deep layers: semantic corruptions
            configs[i] = LayerConfig(
                perturbation_types=[
                    PerturbationType.INTERPOLATE_MIX,
                    PerturbationType.CUTMIX,
                    PerturbationType.BATCH_SHUFFLE,
                    PerturbationType.MANIFOLD_MIX,
                    PerturbationType.LABEL_CORRUPTION,
                ],
                intensity=0.5 + 0.3 * (depth_ratio - 0.66),  # 0.5 to 0.7
                combine_mode="random",
                custom_params={
                    "mix_alpha_range": (0.3, 0.7),
                    "shuffle_ratio": 0.5,
                },
            )

    return configs


@StrategyRegistry.register('layer_wise')
class LayerWiseStrategy(NegativeStrategy):
    """
    Layer-adaptive negative sample generation strategy.

    This strategy generates different types of negative samples depending on
    which layer is currently being trained in the Forward-Forward algorithm.

    Key Design Principles:
    ----------------------
    1. **Early Layers (Low-level features)**:
       - Simple noise-based perturbations (Gaussian, uniform)
       - Minor pixel shuffling
       - Rationale: Early layers learn edges, textures - simple corruptions suffice

    2. **Middle Layers (Mid-level patterns)**:
       - Structural perturbations (masking, dropout)
       - Block-wise corruptions
       - Rationale: Middle layers learn shapes, patterns - need structural breaks

    3. **Deep Layers (High-level semantics)**:
       - Semantic corruptions (image mixing, manifold interpolation)
       - Label-based perturbations
       - Rationale: Deep layers learn abstract concepts - need semantic challenges

    Usage Example:
    --------------
    ```python
    # Basic usage with defaults
    strategy = LayerWiseStrategy(num_classes=10, num_layers=4)

    # Training loop
    for layer_idx, layer in enumerate(model.layers):
        negatives = strategy.generate(
            images, labels,
            layer_idx=layer_idx
        )
        # Train layer with positives and negatives

    # Custom layer configurations
    custom_configs = {
        0: LayerConfig(
            perturbation_types=[PerturbationType.GAUSSIAN_NOISE],
            intensity=0.1
        ),
        1: LayerConfig(
            perturbation_types=[PerturbationType.INTERPOLATE_MIX],
            intensity=0.5
        ),
    }
    strategy = LayerWiseStrategy(
        num_classes=10,
        num_layers=2,
        layer_configs=custom_configs
    )
    ```

    References:
    -----------
    - Hinton (2022): "The Forward-Forward Algorithm"
    - Layer Collaboration in FF: arXiv:2305.12393
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_layers: int = 4,
        layer_configs: Optional[Dict[int, LayerConfig]] = None,
        use_label_embedding: bool = True,
        label_scale: float = 1.0,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize the Layer-Wise Strategy.

        Args:
            num_classes: Number of classes in the dataset (for label-based perturbations)
            num_layers: Total number of layers in the network
            layer_configs: Custom configurations per layer. If None, uses defaults.
            use_label_embedding: Whether to embed labels in the first N pixels
            label_scale: Scale factor for label embedding (if used)
            device: Device to create tensors on
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(num_classes=num_classes, device=device, **kwargs)

        self.num_layers = num_layers
        self.use_label_embedding = use_label_embedding
        self.label_scale = label_scale
        self.current_layer = 0

        # Set up layer configurations
        if layer_configs is not None:
            self.layer_configs = layer_configs
        else:
            self.layer_configs = get_default_layer_configs(num_layers)

        # Running statistics for adaptive perturbations
        self._layer_stats: Dict[int, Dict[str, torch.Tensor]] = {}

        # Perturbation method dispatch table
        self._perturbation_methods: Dict[PerturbationType, Callable] = {
            PerturbationType.GAUSSIAN_NOISE: self._apply_gaussian_noise,
            PerturbationType.UNIFORM_NOISE: self._apply_uniform_noise,
            PerturbationType.PIXEL_SHUFFLE: self._apply_pixel_shuffle,
            PerturbationType.RANDOM_MASK: self._apply_random_mask,
            PerturbationType.BLOCK_MASK: self._apply_block_mask,
            PerturbationType.FEATURE_DROPOUT: self._apply_feature_dropout,
            PerturbationType.INTERPOLATE_MIX: self._apply_interpolate_mix,
            PerturbationType.CUTMIX: self._apply_cutmix,
            PerturbationType.BATCH_SHUFFLE: self._apply_batch_shuffle,
            PerturbationType.LABEL_CORRUPTION: self._apply_label_corruption,
            PerturbationType.ADVERSARIAL_DIRECTION: self._apply_adversarial_direction,
            PerturbationType.MANIFOLD_MIX: self._apply_manifold_mix,
        }

    def set_layer(self, layer_idx: int) -> 'LayerWiseStrategy':
        """
        Set the current layer for generation.

        Args:
            layer_idx: Index of the layer to generate negatives for

        Returns:
            Self for method chaining
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        self.current_layer = layer_idx
        return self

    def set_layer_config(self, layer_idx: int, config: LayerConfig) -> 'LayerWiseStrategy':
        """
        Set configuration for a specific layer.

        Args:
            layer_idx: Index of the layer
            config: Configuration for that layer

        Returns:
            Self for method chaining
        """
        self.layer_configs[layer_idx] = config
        return self

    def create_positive(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Create positive samples, optionally with label embedding.

        Args:
            images: Input images (B, C, H, W) or (B, D)
            labels: Class labels (B,)

        Returns:
            Positive samples (B, D) with optional label embedding
        """
        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()

        if self.use_label_embedding:
            one_hot = torch.zeros(
                batch_size, self.num_classes,
                device=images.device
            )
            one_hot.scatter_(1, labels.unsqueeze(1), self.label_scale)
            flat[:, :self.num_classes] = one_hot

        return flat

    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        layer_idx: Optional[int] = None,
        activations: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate layer-specific negative samples.

        This method selects the appropriate perturbation strategy based on
        the layer being trained and applies it to create negative samples.

        Args:
            images: Input images (B, C, H, W) or (B, D)
            labels: Class labels (B,)
            layer_idx: Which layer to generate for (default: current_layer)
            activations: Optional layer activations for manifold-based perturbations
            **kwargs: Additional arguments for specific perturbations

        Returns:
            Negative samples (B, D) appropriate for the specified layer
        """
        layer_idx = layer_idx if layer_idx is not None else self.current_layer
        config = self.layer_configs.get(
            layer_idx,
            get_default_layer_configs(self.num_layers)[layer_idx]
        )

        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()

        # Update statistics for this layer (for adaptive perturbations)
        self._update_statistics(layer_idx, flat)

        # Select perturbation type based on combine_mode
        if config.combine_mode == "random":
            # Randomly select one perturbation type
            idx = torch.randint(len(config.perturbation_types), (1,)).item()
            perturbation_type = config.perturbation_types[idx]
            negative = self._apply_perturbation(
                flat, labels, perturbation_type, config, activations
            )

        elif config.combine_mode == "sequential":
            # Apply all perturbations in sequence
            negative = flat.clone()
            for perturbation_type in config.perturbation_types:
                negative = self._apply_perturbation(
                    negative, labels, perturbation_type, config, activations
                )

        elif config.combine_mode == "weighted":
            # Weighted combination of all perturbation outputs
            outputs = []
            for perturbation_type in config.perturbation_types:
                out = self._apply_perturbation(
                    flat, labels, perturbation_type, config, activations
                )
                outputs.append(out)

            # Equal weights by default
            negative = torch.stack(outputs).mean(dim=0)

        else:
            raise ValueError(f"Unknown combine_mode: {config.combine_mode}")

        # Apply label embedding for negatives (with wrong label if using label corruption)
        if self.use_label_embedding:
            wrong_labels = self._generate_wrong_labels(labels)
            one_hot = torch.zeros(
                batch_size, self.num_classes,
                device=images.device
            )
            one_hot.scatter_(1, wrong_labels.unsqueeze(1), self.label_scale)
            negative[:, :self.num_classes] = one_hot

        return negative

    def _apply_perturbation(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        perturbation_type: PerturbationType,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply a specific perturbation to the data.

        Args:
            data: Flattened input data (B, D)
            labels: Class labels (B,)
            perturbation_type: Type of perturbation to apply
            config: Layer configuration with intensity and custom params
            activations: Optional activations for manifold perturbations

        Returns:
            Perturbed data (B, D)
        """
        method = self._perturbation_methods.get(perturbation_type)
        if method is None:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        return method(data, labels, config, activations)

    # =========================================================================
    # EARLY LAYER PERTURBATIONS (Simple, noise-based)
    # =========================================================================

    def _apply_gaussian_noise(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add Gaussian noise scaled by data statistics.

        For early layers, this creates simple corruptions that help
        the layer learn robust low-level feature detectors.
        """
        std = data.std(dim=1, keepdim=True) + 1e-8
        noise = torch.randn_like(data) * std * config.intensity
        return data + noise

    def _apply_uniform_noise(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add uniform noise scaled by data range.

        Provides a different noise distribution compared to Gaussian,
        potentially more effective for certain data types.
        """
        data_range = data.max(dim=1, keepdim=True)[0] - data.min(dim=1, keepdim=True)[0]
        noise = (torch.rand_like(data) - 0.5) * data_range * config.intensity
        return data + noise

    def _apply_pixel_shuffle(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Randomly shuffle a portion of pixels/features.

        Creates local disruptions that early layers need to be robust against.
        """
        batch_size, num_features = data.shape
        shuffle_ratio = config.intensity
        num_shuffle = int(num_features * shuffle_ratio)

        result = data.clone()
        for i in range(batch_size):
            # Select random indices to shuffle
            indices = torch.randperm(num_features, device=data.device)[:num_shuffle]
            shuffled_indices = indices[torch.randperm(num_shuffle, device=data.device)]
            result[i, indices] = data[i, shuffled_indices]

        return result

    # =========================================================================
    # MIDDLE LAYER PERTURBATIONS (Structural)
    # =========================================================================

    def _apply_random_mask(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Randomly mask (zero out) portions of the data.

        Creates structural holes that middle layers must learn to handle.
        """
        mask_ratio = config.custom_params.get("mask_ratio", config.intensity)
        mask = (torch.rand_like(data) > mask_ratio).float()
        return data * mask

    def _apply_block_mask(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply block-wise masking (cutout-style).

        More structured masking that removes contiguous regions,
        forcing middle layers to rely on context.
        """
        batch_size, num_features = data.shape
        block_size = config.custom_params.get("block_size", 4)

        # Assume square data for block masking
        side = int(num_features ** 0.5)
        if side * side != num_features:
            # Fall back to random mask for non-square data
            return self._apply_random_mask(data, labels, config, activations)

        result = data.clone().view(batch_size, side, side)
        num_blocks = int((side // block_size) ** 2 * config.intensity)

        for i in range(batch_size):
            for _ in range(num_blocks):
                x = torch.randint(0, side - block_size + 1, (1,)).item()
                y = torch.randint(0, side - block_size + 1, (1,)).item()
                result[i, x:x+block_size, y:y+block_size] = 0

        return result.view(batch_size, -1)

    def _apply_feature_dropout(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply feature-wise dropout.

        Similar to standard dropout but applied as a perturbation
        rather than regularization.
        """
        dropout_rate = config.custom_params.get("dropout_rate", config.intensity)
        mask = (torch.rand_like(data) > dropout_rate).float()
        # Scale to maintain expected value
        return data * mask / (1 - dropout_rate + 1e-8)

    # =========================================================================
    # DEEP LAYER PERTURBATIONS (Semantic)
    # =========================================================================

    def _apply_interpolate_mix(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mix with shuffled batch using interpolation (MixUp-style).

        Creates semantic blends that don't correspond to any real class,
        challenging deep layers to discriminate real vs mixed semantics.
        """
        batch_size = data.size(0)
        alpha_range = config.custom_params.get("mix_alpha_range", (0.3, 0.7))

        # Shuffle to get different samples
        perm = torch.randperm(batch_size, device=data.device)
        other = data[perm]

        # Random mixing coefficient
        alpha = torch.rand(batch_size, 1, device=data.device)
        alpha = alpha * (alpha_range[1] - alpha_range[0]) + alpha_range[0]

        return alpha * data + (1 - alpha) * other

    def _apply_cutmix(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply CutMix-style perturbation.

        Replaces a rectangular region with content from another sample,
        creating chimeric images that challenge semantic understanding.
        """
        batch_size, num_features = data.shape

        # Assume square data
        side = int(num_features ** 0.5)
        if side * side != num_features:
            return self._apply_interpolate_mix(data, labels, config, activations)

        # Shuffle to get different samples
        perm = torch.randperm(batch_size, device=data.device)
        other = data[perm]

        result = data.clone().view(batch_size, side, side)
        other_reshaped = other.view(batch_size, side, side)

        # Random box size based on intensity
        box_size = int(side * config.intensity)

        for i in range(batch_size):
            x = torch.randint(0, side - box_size + 1, (1,)).item()
            y = torch.randint(0, side - box_size + 1, (1,)).item()
            result[i, x:x+box_size, y:y+box_size] = other_reshaped[i, x:x+box_size, y:y+box_size]

        return result.view(batch_size, -1)

    def _apply_batch_shuffle(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Partially shuffle features across the batch.

        Creates samples that are chimeras of multiple real samples,
        challenging the deep layer's semantic coherence detection.
        """
        batch_size, num_features = data.shape
        shuffle_ratio = config.custom_params.get("shuffle_ratio", config.intensity)
        num_shuffle = int(num_features * shuffle_ratio)

        result = data.clone()
        # Select features to shuffle
        shuffle_indices = torch.randperm(num_features, device=data.device)[:num_shuffle]

        # Shuffle these features across the batch
        perm = torch.randperm(batch_size, device=data.device)
        result[:, shuffle_indices] = data[perm][:, shuffle_indices]

        return result

    def _apply_label_corruption(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Corrupt by keeping image but implying wrong semantics.

        The actual label embedding is handled in generate(), but this
        method can apply additional perturbations consistent with
        the wrong label.
        """
        # Return data as-is; the label corruption happens in generate()
        # This could be extended to apply class-conditional perturbations
        return data.clone()

    def _apply_adversarial_direction(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perturb in a pseudo-adversarial direction.

        Without gradients, we approximate adversarial direction using
        the difference between class means (if statistics are available).
        """
        # Simple approximation: perturb toward the mean of a different class
        # This is a placeholder - full adversarial would require model access
        batch_size = data.size(0)

        # Use per-sample statistics as proxy
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True) + 1e-8

        # Move away from mean (simple adversarial approximation)
        direction = (data - mean) / std
        return data + config.intensity * direction * std

    def _apply_manifold_mix(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: LayerConfig,
        activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mix in the manifold/activation space if available.

        If layer activations are provided, mix in that space for
        more semantically meaningful interpolations.
        """
        if activations is not None:
            # Mix in activation space
            batch_size = activations.size(0)
            perm = torch.randperm(batch_size, device=activations.device)
            alpha = torch.rand(batch_size, 1, device=activations.device) * config.intensity

            mixed_activations = (1 - alpha) * activations + alpha * activations[perm]
            return mixed_activations

        # Fall back to input-space mixing
        return self._apply_interpolate_mix(data, labels, config, activations)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _generate_wrong_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate wrong labels for each sample.

        Ensures the wrong label is different from the correct one.
        """
        batch_size = labels.size(0)
        device = labels.device

        wrong_labels = torch.randint(
            0, self.num_classes, (batch_size,), device=device
        )
        # Ensure labels are different
        same_mask = wrong_labels == labels
        wrong_labels[same_mask] = (wrong_labels[same_mask] + 1) % self.num_classes

        return wrong_labels

    def _update_statistics(
        self,
        layer_idx: int,
        data: torch.Tensor
    ) -> None:
        """
        Update running statistics for a layer (EMA).

        Used for adaptive perturbations that scale with data statistics.
        """
        with torch.no_grad():
            mean = data.mean(dim=0)
            std = data.std(dim=0) + 1e-8

            if layer_idx not in self._layer_stats:
                self._layer_stats[layer_idx] = {
                    "mean": mean,
                    "std": std,
                }
            else:
                alpha = 0.1  # EMA coefficient
                self._layer_stats[layer_idx]["mean"] = (
                    alpha * mean + (1 - alpha) * self._layer_stats[layer_idx]["mean"]
                )
                self._layer_stats[layer_idx]["std"] = (
                    alpha * std + (1 - alpha) * self._layer_stats[layer_idx]["std"]
                )

    def get_layer_stats(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get running statistics for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Dictionary with 'mean' and 'std' tensors, or None if not collected
        """
        return self._layer_stats.get(layer_idx)

    @property
    def requires_labels(self) -> bool:
        """
        Whether this strategy requires labels.

        Returns True because label embedding and label corruption
        perturbations require labels.
        """
        return True

    def get_config(self) -> Dict[str, Any]:
        """Return strategy configuration for logging/serialization."""
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "use_label_embedding": self.use_label_embedding,
            "label_scale": self.label_scale,
            "layer_configs": {
                idx: {
                    "perturbation_types": [p.value for p in cfg.perturbation_types],
                    "intensity": cfg.intensity,
                    "combine_mode": cfg.combine_mode,
                    "custom_params": cfg.custom_params,
                }
                for idx, cfg in self.layer_configs.items()
            },
        })
        return config

    def to(self, device: torch.device) -> 'LayerWiseStrategy':
        """
        Move strategy to device.

        Also moves cached statistics.
        """
        super().to(device)
        for layer_idx, stats in self._layer_stats.items():
            self._layer_stats[layer_idx] = {
                k: v.to(device) for k, v in stats.items()
            }
        return self

    def __repr__(self) -> str:
        return (
            f"LayerWiseStrategy("
            f"num_classes={self.num_classes}, "
            f"num_layers={self.num_layers}, "
            f"use_label_embedding={self.use_label_embedding})"
        )
