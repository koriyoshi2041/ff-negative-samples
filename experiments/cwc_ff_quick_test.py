"""
CwC-FF 快速测试 - 只运行 3 epochs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time

device = torch.device('cpu')
print(f"Device: {device}")

class CFSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, kernel_size=3, 
                 padding=1, maxpool=True):
        super(CFSEBlock, self).__init__()
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.maxpool_flag = maxpool
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.criterion = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)
        self.losses = []
        
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)
        x = self.bn(x)
        return x
    
    def compute_goodness(self, y, labels):
        batch_size = y.shape[0]
        channels_per_class = self.out_channels // self.num_classes
        y_sets = torch.split(y, channels_per_class, dim=1)
        goodness_factors = []
        for y_set in y_sets:
            gf = y_set.pow(2).mean(dim=(1, 2, 3))
            goodness_factors.append(gf.unsqueeze(-1))
        gf = torch.cat(goodness_factors, dim=1)
        g_pos = gf[torch.arange(batch_size), labels].unsqueeze(-1)
        return g_pos, gf
    
    def train_step(self, x, labels):
        y = self.forward(x)
        g_pos, gf = self.compute_goodness(y, labels)
        loss = self.criterion(gf, labels)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.losses.append(loss.item())
        return y.detach()
    
    def epoch_loss(self):
        if len(self.losses) == 0:
            return 0
        mean_loss = np.mean(self.losses)
        self.losses = []
        return mean_loss


class CwCFFNet(nn.Module):
    def __init__(self, in_channels, num_classes, channels_list):
        super(CwCFFNet, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        in_ch = in_channels
        for i, out_ch in enumerate(channels_list):
            maxpool = (i % 2 == 1)
            layer = CFSEBlock(in_ch, out_ch, num_classes, maxpool=maxpool)
            self.layers.append(layer)
            in_ch = out_ch
        self.final_channels = channels_list[-1]
        self.classifier = None
        
    def _init_classifier(self, feature_dim):
        self.classifier = nn.Linear(feature_dim, self.num_classes).to(device)
        self.classifier_opt = torch.optim.Adam(self.classifier.parameters(), lr=0.01)
        self.classifier_criterion = nn.CrossEntropyLoss()
        
    def train_all_layers(self, x, labels, epoch, total_epochs):
        h = x
        num_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            layer_start = (i * total_epochs) // (num_layers + 1)
            layer_end = ((i + 2) * total_epochs) // (num_layers + 1)
            if layer_start <= epoch < layer_end:
                h = layer.train_step(h, labels)
            else:
                with torch.no_grad():
                    h = layer(h)
        
        if epoch >= total_epochs // 2:
            h_flat = h.view(h.size(0), -1)
            if self.classifier is None:
                self._init_classifier(h_flat.size(1))
            logits = self.classifier(h_flat)
            loss = self.classifier_criterion(logits, labels)
            self.classifier_opt.zero_grad()
            loss.backward()
            self.classifier_opt.step()
        return h.detach()
    
    def predict(self, x):
        h = x
        for layer in self.layers:
            with torch.no_grad():
                h = layer(h)
        h_reshaped = h.view(h.size(0), self.num_classes, 
                           self.final_channels // self.num_classes,
                           h.size(2), h.size(3))
        mean_squared = (h_reshaped ** 2).mean(dim=[2, 3, 4])
        gf_pred = mean_squared.argmax(dim=1)
        if self.classifier is not None:
            h_flat = h.view(h.size(0), -1)
            sf_pred = self.classifier(h_flat).argmax(dim=1)
            return gf_pred, sf_pred
        return gf_pred, gf_pred


def evaluate(model, test_loader):
    model.eval()
    gf_correct = sf_correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            gf_pred, sf_pred = model.predict(x)
            gf_correct += (gf_pred == y).sum().item()
            sf_correct += (sf_pred == y).sum().item()
            total += y.size(0)
    return 1 - gf_correct/total, 1 - sf_correct/total


def train(dataset_name, epochs=3, batch_size=128, subset_size=5000):
    print(f"\n{'='*50}")
    print(f"CwC-FF on {dataset_name} ({epochs} epochs, {subset_size} samples)")
    print(f"{'='*50}")
    
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
        in_channels, num_classes = 1, 10
        channels_list = [20, 80, 240, 480]
    else:  # CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        in_channels, num_classes = 3, 10
        channels_list = [20, 80, 240, 480]
    
    # 使用子集加速
    train_subset = Subset(train_data, range(min(subset_size, len(train_data))))
    test_subset = Subset(test_data, range(min(1000, len(test_data))))
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size)
    
    model = CwCFFNet(in_channels, num_classes, channels_list).to(device)
    print(f"Channels: {channels_list}")
    
    results = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_gf = train_sf = train_total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            model.train_all_layers(x, y, epoch, epochs)
            gf_pred, sf_pred = model.predict(x)
            train_gf += (gf_pred == y).sum().item()
            train_sf += (sf_pred == y).sum().item()
            train_total += y.size(0)
        
        gf_test_err, sf_test_err = evaluate(model, test_loader)
        gf_train_err = 1 - train_gf / train_total
        
        layer_losses = [l.epoch_loss() for l in model.layers]
        
        results.append({
            'epoch': epoch + 1,
            'gf_test_err': gf_test_err,
            'sf_test_err': sf_test_err,
            'layer_losses': layer_losses
        })
        
        print(f"Epoch {epoch+1}: GF Test Err={gf_test_err:.4f}, SF Test Err={sf_test_err:.4f}")
        print(f"  Layer Losses: {[f'{l:.4f}' for l in layer_losses]}")
    
    elapsed = time.time() - start_time
    print(f"\nTime: {elapsed:.1f}s, Final GF Err: {results[-1]['gf_test_err']:.4f}")
    
    return results


if __name__ == '__main__':
    print("CwC-FF Quick Test")
    print("="*50)
    
    # MNIST
    mnist_results = train('MNIST', epochs=3, subset_size=5000)
    
    # CIFAR-10
    cifar_results = train('CIFAR10', epochs=3, subset_size=5000)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"MNIST GF Test Error (3 epochs): {mnist_results[-1]['gf_test_err']:.4f}")
    print(f"CIFAR-10 GF Test Error (3 epochs): {cifar_results[-1]['gf_test_err']:.4f}")
