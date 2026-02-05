"""
Debug script to verify FF implementation step by step.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def overlay_y_on_x(x, y):
    """
    Original implementation from mpezeshki.
    y can be scalar (for prediction) or tensor (for training).
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Layer(nn.Linear):
    """Exact copy from mpezeshki implementation."""
    
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train_layer(self, x_pos, x_neg, verbose=True):
        for i in range(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Epoch {i+1}: loss={loss.item():.4f}, "
                      f"g_pos={g_pos.mean().item():.4f}, "
                      f"g_neg={g_neg.mean().item():.4f}")
        
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class Net(torch.nn.Module):
    """Exact copy from mpezeshki implementation (without CUDA)."""
    
    def __init__(self, dims, device):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers.append(Layer(dims[d], dims[d + 1]).to(device))

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(h.pow(2).mean(1))
            goodness_per_label.append(sum(goodness).unsqueeze(1))
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train_net(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print(f'\nTraining layer {i}...')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg)


def main():
    torch.manual_seed(1234)
    device = get_device()
    print(f"Device: {device}")
    
    # Data
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])
    
    train_loader = DataLoader(
        MNIST('./data/', train=True, download=True, transform=transform),
        batch_size=50000, shuffle=True
    )
    
    test_loader = DataLoader(
        MNIST('./data/', train=False, download=True, transform=transform),
        batch_size=10000, shuffle=False
    )
    
    # Load data
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    print(f"Training data: x.shape={x.shape}, y.shape={y.shape}")
    print(f"x range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    
    # Create positive and negative samples
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    
    # Debug: check samples
    print(f"\nx_pos[:5, :12]:\n{x_pos[:5, :12]}")
    print(f"\nx_neg[:5, :12]:\n{x_neg[:5, :12]}")
    print(f"\nLabels y[:5]: {y[:5]}")
    print(f"Shuffled y[rnd][:5]: {y[rnd][:5]}")
    
    # Train
    net = Net([784, 500, 500], device)
    net.train_net(x_pos, x_neg)
    
    # Evaluate train
    print('\n' + '='*50)
    train_acc = (net.predict(x) == y).float().mean().item()
    print(f'Train accuracy: {train_acc*100:.2f}%')
    print(f'Train error: {(1-train_acc)*100:.2f}%')
    
    # Evaluate test
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)
    
    test_acc = (net.predict(x_te) == y_te).float().mean().item()
    print(f'Test accuracy: {test_acc*100:.2f}%')
    print(f'Test error: {(1-test_acc)*100:.2f}%')
    

if __name__ == "__main__":
    main()
