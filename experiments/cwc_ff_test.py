"""
CwC-FF (Channel-wise Competitive Forward-Forward) 实验脚本
基于 Papachristodoulou et al. AAAI 2024 的实现

此脚本在 MNIST 和 CIFAR-10 上评估 CwC-FF 方法
"""

import sys
sys.path.insert(0, '../repos/CwComp/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CwCLoss(nn.Module):
    """通道级竞争损失函数"""
    def __init__(self):
        super(CwCLoss, self).__init__()
        self.eps = 1e-9

    def forward(self, g_pos, logits):
        logits = torch.clamp(logits, min=-50, max=50)
        g_pos = torch.clamp(g_pos, min=-50, max=50)
        exp_sum = torch.sum(torch.exp(logits), dim=1)
        loss = -torch.mean(torch.log((torch.exp(g_pos) + self.eps) / (exp_sum + self.eps)))
        return loss


class CFSEBlock(nn.Module):
    """
    Channel-wise Feature Separator and Extractor (CFSE) 块
    
    包含:
    1. 标准卷积层 - 学习组合特征
    2. 分组卷积层 - 分离不同类别的特征空间
    """
    def __init__(self, in_channels, out_channels, num_classes, kernel_size=3, 
                 padding=1, use_groups=True, maxpool=True):
        super(CFSEBlock, self).__init__()
        
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.use_groups = use_groups
        self.maxpool_flag = maxpool
        
        # 标准卷积 - 学习组合特征
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
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
        """计算通道级优度因子"""
        batch_size = y.shape[0]
        channels_per_class = self.out_channels // self.num_classes
        
        # 将特征图按类别数分组
        y_sets = torch.split(y, channels_per_class, dim=1)
        
        # 计算每个组的优度 (平方激活的均值)
        goodness_factors = []
        for y_set in y_sets:
            gf = y_set.pow(2).mean(dim=(1, 2, 3))
            goodness_factors.append(gf.unsqueeze(-1))
        
        gf = torch.cat(goodness_factors, dim=1)  # [batch, num_classes]
        
        # 提取正样本优度
        g_pos = gf[torch.arange(batch_size), labels].unsqueeze(-1)
        
        return g_pos, gf
    
    def train_step(self, x, labels):
        """单步训练"""
        y = self.forward(x)
        g_pos, gf = self.compute_goodness(y, labels)
        
        # 使用 CrossEntropy 损失 (CwC_CE)
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
    """CwC-FF 网络"""
    def __init__(self, in_channels, num_classes, channels_list, use_cfse=True):
        super(CwCFFNet, self).__init__()
        
        self.num_classes = num_classes
        self.use_cfse = use_cfse
        
        self.layers = nn.ModuleList()
        
        in_ch = in_channels
        for i, out_ch in enumerate(channels_list):
            use_groups = (i % 2 == 1) and use_cfse  # 奇数层使用分组卷积
            maxpool = (i % 2 == 1)  # 奇数层使用 maxpool
            
            layer = CFSEBlock(in_ch, out_ch, num_classes, 
                             use_groups=use_groups, maxpool=maxpool)
            self.layers.append(layer)
            in_ch = out_ch
        
        self.final_channels = channels_list[-1]
        
        # Softmax 分类器
        # 需要根据最终特征图大小动态设置
        self.classifier = None
        
    def _init_classifier(self, feature_dim):
        self.classifier = nn.Linear(feature_dim, self.num_classes).to(
            next(self.layers[0].parameters()).device
        )
        self.classifier_opt = torch.optim.Adam(self.classifier.parameters(), lr=0.01)
        self.classifier_criterion = nn.CrossEntropyLoss()
        self.classifier_losses = []
        
    def train_layer(self, x, labels, layer_idx):
        """训练单层"""
        h = x
        for i, layer in enumerate(self.layers):
            if i < layer_idx:
                with torch.no_grad():
                    h = layer(h)
            elif i == layer_idx:
                h = layer.train_step(h, labels)
            else:
                break
        return h
    
    def train_all_layers(self, x, labels, epoch, total_epochs):
        """ILT (Interleaved Layer Training) 策略"""
        h = x
        num_layers = len(self.layers)
        
        for i, layer in enumerate(self.layers):
            # 简化的 ILT: 每层训练固定 epochs
            layer_start = (i * total_epochs) // (num_layers + 1)
            layer_end = ((i + 2) * total_epochs) // (num_layers + 1)
            
            if layer_start <= epoch < layer_end:
                h = layer.train_step(h, labels)
            else:
                with torch.no_grad():
                    h = layer(h)
        
        # 训练分类器 (后半部分 epochs)
        if epoch >= total_epochs // 2:
            h_flat = h.view(h.size(0), -1)
            if self.classifier is None:
                self._init_classifier(h_flat.size(1))
            
            logits = self.classifier(h_flat)
            loss = self.classifier_criterion(logits, labels)
            
            self.classifier_opt.zero_grad()
            loss.backward()
            self.classifier_opt.step()
            self.classifier_losses.append(loss.item())
        
        return h.detach()
    
    def predict(self, x):
        """推理"""
        h = x
        for layer in self.layers:
            with torch.no_grad():
                h = layer(h)
        
        # 基于优度的预测
        h_reshaped = h.view(h.size(0), self.num_classes, 
                           self.final_channels // self.num_classes,
                           h.size(2), h.size(3))
        mean_squared = (h_reshaped ** 2).mean(dim=[2, 3, 4])
        gf_pred = mean_squared.argmax(dim=1)
        
        # Softmax 预测
        if self.classifier is not None:
            h_flat = h.view(h.size(0), -1)
            sf_pred = self.classifier(h_flat).argmax(dim=1)
            return gf_pred, sf_pred
        
        return gf_pred, gf_pred


def evaluate(model, test_loader, device):
    """评估模型"""
    model.eval()
    gf_correct = 0
    sf_correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            gf_pred, sf_pred = model.predict(x)
            
            gf_correct += (gf_pred == y).sum().item()
            sf_correct += (sf_pred == y).sum().item()
            total += y.size(0)
    
    return 1 - gf_correct/total, 1 - sf_correct/total  # 返回错误率


def train_cwc_ff(dataset_name='MNIST', epochs=10, batch_size=128):
    """训练 CwC-FF 模型"""
    print(f"\n{'='*60}")
    print(f"Training CwC-FF on {dataset_name}")
    print(f"{'='*60}")
    
    # 数据集配置
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
        in_channels = 1
        num_classes = 10
        channels_list = [20, 80, 240, 480]  # 论文配置
        
    elif dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
        in_channels = 3
        num_classes = 10
        channels_list = [20, 80, 240, 480]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = CwCFFNet(in_channels, num_classes, channels_list, use_cfse=True).to(device)
    
    print(f"Model Architecture:")
    print(f"  Channels: {channels_list}")
    print(f"  CFSE: True")
    print(f"  Device: {device}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    
    results = {
        'gf_train_errors': [],
        'sf_train_errors': [],
        'gf_test_errors': [],
        'sf_test_errors': [],
        'layer_losses': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_gf_correct = 0
        train_sf_correct = 0
        train_total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # 训练所有层
            model.train_all_layers(x, y, epoch, epochs)
            
            # 计算训练准确率
            gf_pred, sf_pred = model.predict(x)
            train_gf_correct += (gf_pred == y).sum().item()
            train_sf_correct += (sf_pred == y).sum().item()
            train_total += y.size(0)
        
        # 评估
        gf_test_err, sf_test_err = evaluate(model, test_loader, device)
        gf_train_err = 1 - train_gf_correct / train_total
        sf_train_err = 1 - train_sf_correct / train_total
        
        # 记录层损失
        layer_losses = [layer.epoch_loss() for layer in model.layers]
        
        results['gf_train_errors'].append(gf_train_err)
        results['sf_train_errors'].append(sf_train_err)
        results['gf_test_errors'].append(gf_test_err)
        results['sf_test_errors'].append(sf_test_err)
        results['layer_losses'].append(layer_losses)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  GF Pred - Train Error: {gf_train_err:.4f}, Test Error: {gf_test_err:.4f}")
        print(f"  SF Pred - Train Error: {sf_train_err:.4f}, Test Error: {sf_test_err:.4f}")
        print(f"  Layer Losses: {[f'{l:.4f}' for l in layer_losses]}")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Time: {elapsed_time:.1f}s")
    print(f"  Best GF Test Error: {min(results['gf_test_errors']):.4f}")
    print(f"  Best SF Test Error: {min(results['sf_test_errors']):.4f}")
    print(f"{'='*60}")
    
    return model, results


if __name__ == '__main__':
    # 测试 MNIST
    mnist_model, mnist_results = train_cwc_ff('MNIST', epochs=10, batch_size=128)
    
    # 测试 CIFAR-10
    cifar_model, cifar_results = train_cwc_ff('CIFAR10', epochs=10, batch_size=128)
    
    # 保存结果
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"MNIST Final Test Error (GF): {mnist_results['gf_test_errors'][-1]:.4f}")
    print(f"MNIST Final Test Error (SF): {mnist_results['sf_test_errors'][-1]:.4f}")
    print(f"CIFAR-10 Final Test Error (GF): {cifar_results['gf_test_errors'][-1]:.4f}")
    print(f"CIFAR-10 Final Test Error (SF): {cifar_results['sf_test_errors'][-1]:.4f}")
