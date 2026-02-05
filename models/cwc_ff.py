"""
Channel-wise Competitive Forward-Forward (CwC-FF / CwComp) Implementation

Based on: Papachristodoulou et al. (2024) "Convolutional Channel-wise Competitive
Learning for the Forward-Forward Algorithm"
Paper: https://arxiv.org/abs/2312.12668 (AAAI 2024)

Key innovations:
1. NO negative samples needed! Channel competition replaces pos/neg contrast
2. CFSE blocks: Grouped convolutions force channel specialization per class
3. CwCLoss: Cross-entropy over channel-wise goodness scores
4. Faster convergence than standard FF

Reported results:
- MNIST: 0.58% error
- Fashion-MNIST: 7.69% error
- CIFAR-10: 21.89% error
- CIFAR-100: 48.77% error

Architecture pattern:
- Odd layers: Grouped conv (groups=num_classes) + maxpool (CFSE blocks)
- Even layers: Standard conv (groups=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
import time


def get_device() -> torch.device:
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Loss Functions
# ============================================================

class CwCLoss(nn.Module):
    """
    Channel-wise Competitive Loss.

    Computes softmax cross-entropy over channel-wise goodness scores.
    This eliminates the need for negative samples - channels compete against
    each other for representing each class.

    Formula: L = -log(exp(g_pos) / sum(exp(g_all)))

    Equivalent to: CrossEntropyLoss(goodness_matrix, targets)
    """

    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, g_pos: torch.Tensor, goodness_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g_pos: Goodness of correct class channels [B, 1]
            goodness_matrix: Goodness for all classes [B, num_classes]

        Returns:
            Scalar loss value
        """
        # Clamp for numerical stability
        goodness_matrix = torch.clamp(goodness_matrix, min=-50, max=50)
        g_pos = torch.clamp(g_pos, min=-50, max=50)

        # Softmax cross-entropy style loss
        exp_sum = torch.sum(torch.exp(goodness_matrix), dim=1)
        loss = -torch.mean(torch.log((torch.exp(g_pos.squeeze(-1)) + self.eps) / (exp_sum + self.eps)))

        return loss


class CwCLossCE(nn.Module):
    """
    Simplified CwC Loss using standard CrossEntropyLoss.

    Directly applies cross-entropy on the goodness matrix.
    Equivalent to CwCLoss but uses PyTorch's optimized implementation.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, goodness_matrix: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            goodness_matrix: Per-class goodness scores [B, num_classes]
            targets: Ground truth class labels [B]

        Returns:
            Scalar loss value
        """
        return self.criterion(goodness_matrix, targets)


class PvNLoss(nn.Module):
    """
    Traditional Positive vs Negative Loss (for comparison).

    Standard FF loss with threshold-based separation.
    """

    def __init__(self, threshold: float = 2.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, g_pos: torch.Tensor, g_neg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g_pos: Goodness of positive samples [B, 1]
            g_neg: Goodness of negative samples [B, 1]

        Returns:
            Scalar loss value
        """
        errors = torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold
        ])
        loss = torch.log(1 + torch.exp(errors)).mean()
        return loss


# ============================================================
# CFSE Block (Channel-wise Feature Separator and Extractor)
# ============================================================

class CFSEBlock(nn.Module):
    """
    Channel-wise Feature Separator and Extractor block.

    Uses grouped convolutions where groups=num_classes to force
    each channel group to specialize for its assigned class.

    Architecture:
    - Conv2d with optional grouping (groups=num_classes for CFSE)
    - BatchNorm2d
    - ReLU
    - Optional MaxPool2d (applied in CFSE layers)

    IMPORTANT: For CFSE blocks (grouped conv), both in_channels and out_channels
    must be divisible by num_classes. The architecture should be designed to
    ensure this constraint (e.g., [20, 80, 240, 480] channels for 10 classes).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_classes: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 is_cfse: bool = True,
                 dropout_rate: float = 0.0):
        """
        Args:
            in_channels: Number of input channels (must be divisible by num_classes if is_cfse)
            out_channels: Number of output channels (must be divisible by num_classes if is_cfse)
            num_classes: Number of classes (used for grouping)
            kernel_size: Convolution kernel size
            padding: Convolution padding
            is_cfse: If True, use grouped convolution and maxpool
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.num_classes = num_classes
        self.is_cfse = is_cfse
        self.out_channels = out_channels

        # Groups: num_classes for CFSE, 1 for standard conv
        groups = num_classes if is_cfse else 1

        # Validate channel divisibility for CFSE blocks
        if is_cfse:
            if out_channels % num_classes != 0:
                raise ValueError(
                    f"out_channels ({out_channels}) must be divisible by "
                    f"num_classes ({num_classes}) for CFSE blocks"
                )
            if in_channels % num_classes != 0:
                raise ValueError(
                    f"in_channels ({in_channels}) must be divisible by "
                    f"num_classes ({num_classes}) for CFSE blocks. "
                    f"Ensure previous layer output channels are divisible by num_classes."
                )

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # MaxPool only for CFSE blocks
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) if is_cfse else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.bn(x)
        return x


# ============================================================
# CwC-FF Layer
# ============================================================

class CwCFFLayer(nn.Module):
    """
    Channel-wise Competitive Forward-Forward Layer.

    Wraps a CFSEBlock with channel-wise goodness computation and
    local training capability.

    Key features:
    - Channel-wise goodness: Partitions output into num_classes groups
    - Each group's goodness = mean squared activation
    - Competitive loss forces specialization
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_size: int,
                 num_classes: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 is_cfse: bool = True,
                 dropout_rate: float = 0.0,
                 lr: float = 0.01,
                 loss_type: str = 'CwC_CE'):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            input_size: Spatial size of input (H=W assumed)
            num_classes: Number of classes
            kernel_size: Convolution kernel size
            padding: Convolution padding
            is_cfse: If True, use CFSE block (grouped conv + maxpool)
            dropout_rate: Dropout probability
            lr: Learning rate
            loss_type: 'CwC', 'CwC_CE', or 'PvN'
        """
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.is_cfse = is_cfse
        self.loss_type = loss_type

        # CFSE block
        self.block = CFSEBlock(
            in_channels, out_channels, num_classes,
            kernel_size, padding, is_cfse, dropout_rate
        )

        # Compute output spatial size
        self.output_size = input_size // 2 if is_cfse else input_size

        # Loss function
        if loss_type == 'CwC':
            self.criterion = CwCLoss()
        elif loss_type == 'CwC_CE':
            self.criterion = CwCLossCE()
        elif loss_type == 'PvN':
            self.criterion = PvNLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Optimizer
        self.opt = Adam(self.parameters(), lr=lr)

        # Training tracking
        self.epoch_losses: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer."""
        return self.block(x)

    def compute_goodness_channelwise(self, y: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute channel-wise goodness scores.

        Partitions output channels into num_classes groups and computes
        mean squared activation for each group.

        Args:
            y: Layer output [B, C, H, W]
            targets: Ground truth labels [B]

        Returns:
            g_pos: Goodness of correct class [B, 1]
            g_neg: Mean goodness of incorrect classes [B, 1]
            goodness_matrix: Goodness for all classes [B, num_classes]
        """
        batch_size = y.shape[0]
        channels_per_class = self.out_channels // self.num_classes

        # Split channels into class groups
        # y: [B, C, H, W] -> list of [B, C/num_classes, H, W]
        y_splits = torch.split(y, channels_per_class, dim=1)

        # Compute goodness for each class group
        # Goodness = mean of squared activations over (channels, H, W)
        goodness_factors = [y_split.pow(2).mean(dim=(1, 2, 3)).unsqueeze(-1)
                           for y_split in y_splits]
        goodness_matrix = torch.cat(goodness_factors, dim=1)  # [B, num_classes]

        # Create masks for positive (correct class) and negative (other classes)
        pos_mask = torch.zeros((batch_size, self.num_classes), dtype=torch.bool, device=y.device)
        pos_mask[torch.arange(batch_size, device=y.device), targets] = True
        neg_mask = ~pos_mask

        # Extract g_pos and g_neg
        g_pos = goodness_matrix[pos_mask].view(batch_size, 1)
        g_neg = goodness_matrix[neg_mask].view(batch_size, -1).mean(dim=1, keepdim=True)

        return g_pos, g_neg, goodness_matrix

    def train_step(self, x: torch.Tensor, targets: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Single training step for this layer.

        Args:
            x: Input tensor [B, C, H, W]
            targets: Ground truth labels [B]
            verbose: Print training info

        Returns:
            Detached output tensor for next layer
        """
        # Forward pass
        y = self.forward(x)

        # Compute channel-wise goodness
        g_pos, g_neg, goodness_matrix = self.compute_goodness_channelwise(y, targets)

        # Compute loss based on type
        if self.loss_type == 'CwC':
            loss = self.criterion(g_pos, goodness_matrix)
        elif self.loss_type == 'CwC_CE':
            loss = self.criterion(goodness_matrix, targets)
        elif self.loss_type == 'PvN':
            loss = self.criterion(g_pos, g_neg)

        self.epoch_losses.append(loss.item())

        if verbose:
            print(f"    g_pos: {g_pos.mean().item():.4f}, "
                  f"g_neg: {g_neg.mean().item():.4f}, "
                  f"loss: {loss.item():.4f}")

        # Backward pass (local gradient only!)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return y.detach()

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward pass (no gradient)."""
        with torch.no_grad():
            return self.forward(x)

    def get_epoch_loss(self) -> float:
        """Get mean loss for current epoch and reset."""
        if len(self.epoch_losses) == 0:
            return 0.0
        mean_loss = sum(self.epoch_losses) / len(self.epoch_losses)
        self.epoch_losses = []
        return mean_loss


# ============================================================
# CwC-FF Network
# ============================================================

class CwCFFNetwork(nn.Module):
    """
    Channel-wise Competitive Forward-Forward CNN Network.

    Architecture follows the CFSE pattern:
    - Odd layers (idx 1, 3, 5...): CFSE blocks (grouped conv + maxpool)
    - Even layers (idx 0, 2, 4...): Standard conv blocks

    Prediction uses Global Averaging (GA):
    - Reshape final features into class-wise groups
    - Predict class with highest mean squared activation
    """

    def __init__(self,
                 out_channels_list: List[int],
                 num_classes: int = 10,
                 input_channels: int = 1,
                 input_size: int = 28,
                 use_cfse: bool = True,
                 dropout_rate: float = 0.0,
                 lr: float = 0.01,
                 loss_type: str = 'CwC_CE',
                 ilt_schedule: Optional[List[List[int]]] = None):
        """
        Args:
            out_channels_list: List of output channels for each layer
            num_classes: Number of classes
            input_channels: Number of input channels (1 for MNIST, 3 for CIFAR)
            input_size: Input spatial size (28 for MNIST, 32 for CIFAR)
            use_cfse: If True, use alternating CFSE pattern
            dropout_rate: Dropout probability
            lr: Learning rate
            loss_type: Loss function type ('CwC', 'CwC_CE', 'PvN')
            ilt_schedule: Incremental Layer Training schedule [[start, end], ...]
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_cfse = use_cfse
        self.final_channels = out_channels_list[-1]

        # Default ILT schedule (train all layers simultaneously)
        if ilt_schedule is None:
            ilt_schedule = [[0, 100]] * len(out_channels_list)
        self.ilt_schedule = ilt_schedule

        # Build layers
        self.layers = nn.ModuleList()
        current_channels = input_channels
        current_size = input_size

        for i, out_channels in enumerate(out_channels_list):
            # Odd layers are CFSE (if use_cfse is True)
            is_cfse = (i % 2 == 1) and use_cfse

            layer = CwCFFLayer(
                in_channels=current_channels,
                out_channels=out_channels,
                input_size=current_size,
                num_classes=num_classes,
                is_cfse=is_cfse,
                dropout_rate=dropout_rate,
                lr=lr,
                loss_type=loss_type
            )
            self.layers.append(layer)

            # Update dimensions for next layer
            current_channels = out_channels
            current_size = layer.output_size

        self.final_size = current_size

        # Optional softmax classifier for SF prediction
        self.softmax_classifier = None

    def add_softmax_classifier(self, lr: float = 0.01):
        """Add optional softmax classifier for SF prediction."""
        flattened_size = self.final_channels * self.final_size ** 2
        self.softmax_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.0),
            nn.Linear(flattened_size, self.num_classes)
        )
        self.sf_optimizer = Adam(self.softmax_classifier.parameters(), lr=lr)
        self.sf_criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict classes using Global Averaging (GA).

        Computes mean squared activation for each class's channel group
        in the final layer output and returns argmax.
        """
        # Forward through all layers
        h = x
        for layer in self.layers:
            h = layer.infer(h)

        # Reshape: [B, C, H, W] -> [B, num_classes, C/num_classes, H, W]
        channels_per_class = self.final_channels // self.num_classes
        h_reshaped = h.view(
            h.shape[0],
            self.num_classes,
            channels_per_class,
            h.shape[2],
            h.shape[3]
        )

        # Mean squared activation for each class group
        mean_squared = (h_reshaped ** 2).mean(dim=[2, 3, 4])  # [B, num_classes]

        # Predict class with highest activation
        _, predicted = torch.max(mean_squared, dim=1)
        return predicted

    def predict_sf(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using softmax classifier (if available)."""
        if self.softmax_classifier is None:
            raise RuntimeError("Softmax classifier not initialized. Call add_softmax_classifier() first.")

        h = x
        for layer in self.layers:
            h = layer.infer(h)

        logits = self.softmax_classifier(h)
        return logits.argmax(dim=1)

    def train_layer(self,
                    layer_idx: int,
                    train_loader: DataLoader,
                    num_epochs: int,
                    device: torch.device,
                    verbose: bool = True) -> List[float]:
        """
        Train a single layer to convergence (greedy layer-wise training).

        Args:
            layer_idx: Index of layer to train
            train_loader: DataLoader for training data
            num_epochs: Number of epochs to train
            device: Device to use
            verbose: Print training progress

        Returns:
            List of epoch losses
        """
        layer = self.layers[layer_idx]
        epoch_losses = []

        for epoch in range(num_epochs):
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                # Forward through previous layers (frozen)
                h = x
                for i in range(layer_idx):
                    h = self.layers[i].infer(h)

                # Train current layer
                layer.train_step(h, y)

            epoch_loss = layer.get_epoch_loss()
            epoch_losses.append(epoch_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Layer {layer_idx}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        return epoch_losses

    def train_all(self,
                  train_loader: DataLoader,
                  num_epochs: int,
                  device: torch.device,
                  mode: str = 'simultaneous',
                  verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train all layers of the network.

        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs
            device: Device to use
            mode: Training mode
                - 'simultaneous': Train all layers at once (with ILT schedule)
                - 'greedy': Train layer-by-layer to convergence
            verbose: Print training progress

        Returns:
            Dict mapping layer index to list of losses
        """
        self.to(device)
        all_losses: Dict[str, List[float]] = {f'layer_{i}': [] for i in range(len(self.layers))}

        if mode == 'greedy':
            # Greedy layer-by-layer training
            for layer_idx in range(len(self.layers)):
                if verbose:
                    print(f"\nTraining Layer {layer_idx}...")
                losses = self.train_layer(layer_idx, train_loader, num_epochs, device, verbose)
                all_losses[f'layer_{layer_idx}'] = losses

        elif mode == 'simultaneous':
            # Simultaneous training with ILT schedule
            for epoch in range(num_epochs):
                epoch_start = time.time()

                for batch_idx, (x, y) in enumerate(train_loader):
                    x, y = x.to(device), y.to(device)

                    h = x
                    for layer_idx, layer in enumerate(self.layers):
                        # Check ILT schedule
                        start, end = self.ilt_schedule[layer_idx]

                        if start <= epoch < end:
                            # Train this layer
                            h = layer.train_step(h, y)
                        else:
                            # Just forward (frozen)
                            h = layer.infer(h)

                # Record losses
                for layer_idx, layer in enumerate(self.layers):
                    loss = layer.get_epoch_loss()
                    all_losses[f'layer_{layer_idx}'].append(loss)

                if verbose and (epoch + 1) % 5 == 0:
                    epoch_time = time.time() - epoch_start
                    loss_str = ", ".join([f"L{i}:{all_losses[f'layer_{i}'][-1]:.4f}"
                                          for i in range(len(self.layers))])
                    print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s): {loss_str}")

        # Train softmax classifier if available
        if self.softmax_classifier is not None:
            if verbose:
                print("\nTraining Softmax Classifier...")
            self._train_softmax_classifier(train_loader, num_epochs, device, verbose)

        return all_losses

    def _train_softmax_classifier(self,
                                   train_loader: DataLoader,
                                   num_epochs: int,
                                   device: torch.device,
                                   verbose: bool = True):
        """Train the optional softmax classifier."""
        self.softmax_classifier.to(device)

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # Forward through FF layers (frozen)
                with torch.no_grad():
                    h = x
                    for layer in self.layers:
                        h = layer(h)

                # Train classifier
                logits = self.softmax_classifier(h)
                loss = self.sf_criterion(logits, y)

                self.sf_optimizer.zero_grad()
                loss.backward()
                self.sf_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if verbose and (epoch + 1) % 5 == 0:
                avg_loss = total_loss / num_batches
                print(f"  SF Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def get_accuracy(self,
                     data_loader: DataLoader,
                     device: torch.device,
                     method: str = 'GA') -> float:
        """
        Compute accuracy on a dataset.

        Args:
            data_loader: DataLoader for evaluation
            device: Device to use
            method: 'GA' (Global Averaging) or 'SF' (Softmax)

        Returns:
            Accuracy as float
        """
        self.set_eval_mode()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)

                if method == 'GA':
                    predictions = self.predict(x)
                elif method == 'SF':
                    predictions = self.predict_sf(x)
                else:
                    raise ValueError(f"Unknown method: {method}")

                correct += (predictions == y).sum().item()
                total += y.size(0)

        self.set_train_mode()
        return correct / total

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        for layer in self.layers:
            layer.block.bn.eval()
            layer.block.dropout.eval()

    def set_train_mode(self):
        """Set model to training mode."""
        for layer in self.layers:
            layer.block.bn.train()
            layer.block.dropout.train()


# ============================================================
# Data Loading
# ============================================================

def get_mnist_loaders(batch_size: int = 64, data_dir: str = './data'):
    """Get MNIST data loaders for CNN input."""
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_fmnist_loaders(batch_size: int = 64, data_dir: str = './data'):
    """Get Fashion-MNIST data loaders for CNN input."""
    transform = Compose([
        ToTensor(),
        Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar10_loaders(batch_size: int = 64, data_dir: str = './data'):
    """Get CIFAR-10 data loaders for CNN input."""
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================================
# Pre-configured Architectures
# ============================================================

def create_cwc_mnist(use_cfse: bool = True, loss_type: str = 'CwC_CE') -> CwCFFNetwork:
    """
    Create CwC-FF network for MNIST/Fashion-MNIST.

    Architecture: [20, 80, 240, 480] channels
    Expected error rate: ~0.58% (MNIST), ~7.69% (FMNIST)
    """
    return CwCFFNetwork(
        out_channels_list=[20, 80, 240, 480],
        num_classes=10,
        input_channels=1,
        input_size=28,
        use_cfse=use_cfse,
        lr=0.01,
        loss_type=loss_type,
        ilt_schedule=[[0, 2], [0, 3], [0, 4], [0, 5]]  # ILT for MNIST
    )


def create_cwc_cifar10(use_cfse: bool = True, loss_type: str = 'CwC_CE') -> CwCFFNetwork:
    """
    Create CwC-FF network for CIFAR-10.

    Architecture: [20, 80, 240, 480] channels
    Expected error rate: ~21.89%
    """
    return CwCFFNetwork(
        out_channels_list=[20, 80, 240, 480],
        num_classes=10,
        input_channels=3,
        input_size=32,
        use_cfse=use_cfse,
        lr=0.01,
        loss_type=loss_type,
        ilt_schedule=[[0, 11], [0, 16], [0, 21], [0, 25]]  # ILT for CIFAR
    )


# ============================================================
# Training Functions
# ============================================================

def train_cwc_network(model: CwCFFNetwork,
                      train_loader: DataLoader,
                      test_loader: DataLoader,
                      num_epochs: int,
                      device: torch.device,
                      verbose: bool = True) -> Dict:
    """
    Train a CwC-FF network and evaluate.

    Args:
        model: CwCFFNetwork instance
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        device: Device to use
        verbose: Print progress

    Returns:
        Dict with training results
    """
    model.to(device)

    start_time = time.time()
    losses = model.train_all(train_loader, num_epochs, device, mode='simultaneous', verbose=verbose)
    train_time = time.time() - start_time

    # Evaluate
    train_acc = model.get_accuracy(train_loader, device, method='GA')
    test_acc = model.get_accuracy(test_loader, device, method='GA')

    results = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_error': 1.0 - train_acc,
        'test_error': 1.0 - test_acc,
        'train_time': train_time,
        'losses': losses
    }

    return results


# ============================================================
# Main Demo
# ============================================================

def main(dataset: str = 'mnist', num_epochs: int = 20, batch_size: int = 64, seed: int = 1234):
    """
    Run CwC-FF training demo.

    Args:
        dataset: 'mnist', 'fmnist', or 'cifar10'
        num_epochs: Number of training epochs
        batch_size: Batch size
        seed: Random seed
    """
    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # Load data
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_loaders(batch_size)
        model = create_cwc_mnist()
        ref_error = 0.58
    elif dataset == 'fmnist':
        train_loader, test_loader = get_fmnist_loaders(batch_size)
        model = create_cwc_mnist()
        ref_error = 7.69
    elif dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(batch_size)
        model = create_cwc_cifar10()
        ref_error = 21.89
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print("\n" + "="*60)
    print(f"CwC-FF (Channel-wise Competitive Forward-Forward)")
    print("="*60)
    print(f"Dataset: {dataset.upper()}")
    print(f"Architecture: {[l.out_channels for l in model.layers]} channels")
    print(f"CFSE enabled: {model.use_cfse}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")

    # Train
    print("\nTraining...")
    results = train_cwc_network(model, train_loader, test_loader, num_epochs, device)

    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Train accuracy: {results['train_acc']*100:.2f}%")
    print(f"Train error:    {results['train_error']*100:.2f}%")
    print(f"Test accuracy:  {results['test_acc']*100:.2f}%")
    print(f"Test error:     {results['test_error']*100:.2f}%")
    print(f"Training time:  {results['train_time']:.1f}s")

    print("\n" + "-"*60)
    print(f"Reference (paper): ~{ref_error}% test error")
    print("-"*60)

    return model, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CwC-FF Training")
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fmnist', 'cifar10'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()

    model, results = main(
        dataset=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )
