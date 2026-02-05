#!/usr/bin/env python3
"""
Layer Collaboration + Transfer Learning Experiment
===================================================

研究问题：Layer Collaboration 机制是否改善 FF 的迁移学习能力？

实验设计：
1. 源任务预训练：MNIST / CIFAR-10
2. 目标任务迁移：Fashion-MNIST / CIFAR-100
3. 对比组：BP, Original FF, Layer Collab FF (γ=all), Layer Collab FF (γ<t)

使用方法：
    python transfer_experiment.py --config mnist_fmnist --seed 42
    python transfer_experiment.py --config cifar10_100 --seed 42 --full

作者：Clawd (for Parafee)
日期：2026-02-05
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 数据集
    source_dataset: str = 'mnist'
    target_dataset: str = 'fashion_mnist'
    
    # 网络架构
    hidden_sizes: List[int] = None
    threshold: float = 2.0
    
    # 训练参数
    pretrain_epochs: int = 60
    transfer_epochs: int = 100
    batch_size: int = 64
    lr_ff: float = 0.03
    lr_bp: float = 0.001
    lr_finetune: float = 0.001
    lr_head: float = 0.01
    
    # 实验设置
    seed: int = 42
    device: str = 'auto'
    num_workers: int = 4
    
    # 输出
    results_dir: str = './results/transfer'
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [500, 500, 500]
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else \
                         'mps' if torch.backends.mps.is_available() else 'cpu'


CONFIGS = {
    'mnist_fmnist': ExperimentConfig(
        source_dataset='mnist',
        target_dataset='fashion_mnist',
        hidden_sizes=[500, 500, 500],
    ),
    'cifar10_100': ExperimentConfig(
        source_dataset='cifar10',
        target_dataset='cifar100',
        hidden_sizes=[1000, 1000, 1000],
        pretrain_epochs=100,
        transfer_epochs=150,
    ),
}


# ============================================================
# Data Loading
# ============================================================

def get_dataset(name: str, train: bool = True) -> Tuple[torch.utils.data.Dataset, int, int]:
    """
    获取数据集
    
    Returns:
        dataset, input_size, num_classes
    """
    data_dir = './data'
    
    if name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform)
        return dataset, 784, 10
    
    elif name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        dataset = datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)
        return dataset, 784, 10
    
    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
        return dataset, 3072, 10
    
    elif name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        dataset = datasets.CIFAR100(data_dir, train=train, download=True, transform=transform)
        return dataset, 3072, 100
    
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ============================================================
# Models
# ============================================================

class FFLayer(nn.Module):
    """Forward-Forward 层（支持 Layer Collaboration）"""
    
    def __init__(self, in_features: int, out_features: int, threshold: float = 2.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.optimizer = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer normalization (FF requires this)
        x = x / (x.norm(2, dim=1, keepdim=True) + 1e-8)
        return self.relu(self.linear(x))
    
    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """计算 goodness (squared L2 norm)"""
        return (h ** 2).sum(dim=1)
    
    def ff_loss(self, pos_g: torch.Tensor, neg_g: torch.Tensor,
                gamma_pos: torch.Tensor = None, gamma_neg: torch.Tensor = None) -> torch.Tensor:
        """
        计算 FF 损失
        
        Args:
            pos_g: 正样本 goodness
            neg_g: 负样本 goodness
            gamma_pos: 正样本全局 goodness (Layer Collab)
            gamma_neg: 负样本全局 goodness (Layer Collab)
        """
        if gamma_pos is None:
            gamma_pos = torch.zeros_like(pos_g)
        if gamma_neg is None:
            gamma_neg = torch.zeros_like(neg_g)
        
        # 正样本应该有高 goodness，负样本应该有低 goodness
        pos_logit = pos_g + gamma_pos - self.threshold
        neg_logit = neg_g + gamma_neg - self.threshold
        
        # Binary cross entropy style loss
        loss_pos = torch.log(1 + torch.exp(-pos_logit)).mean()
        loss_neg = torch.log(1 + torch.exp(neg_logit)).mean()
        
        return loss_pos + loss_neg


class FFNetwork(nn.Module):
    """Forward-Forward 网络（支持 Original 和 Layer Collaboration）"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], 
                 num_classes: int, threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.threshold = threshold
        
        # 构建层
        layer_sizes = [input_size] + hidden_sizes
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = FFLayer(layer_sizes[i], layer_sizes[i+1], threshold)
            layer.optimizer = optim.Adam(layer.parameters(), lr=lr)
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """返回所有层的激活"""
        activations = []
        h = x
        for layer in self.layers:
            h = layer(h)
            activations.append(h)
        return activations
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取最后一层特征（用于迁移）"""
        with torch.no_grad():
            h = x
            for layer in self.layers:
                h = layer(h)
            return h
    
    def embed_label(self, images: torch.Tensor, labels: torch.Tensor, 
                    wrong: bool = False) -> torch.Tensor:
        """嵌入标签到图像前几个像素"""
        batch_size = images.size(0)
        flat = images.view(batch_size, -1).clone()
        
        if wrong:
            # 生成错误标签
            rand_labels = torch.randint(0, self.num_classes, (batch_size,), device=images.device)
            mask = rand_labels == labels
            rand_labels[mask] = (rand_labels[mask] + 1) % self.num_classes
            labels = rand_labels
        
        one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        flat[:, :self.num_classes] = one_hot
        
        return flat
    
    def compute_all_goodness(self, x: torch.Tensor) -> List[torch.Tensor]:
        """计算所有层的 goodness（detached，用于 γ 计算）"""
        goodness_list = []
        h = x
        for layer in self.layers:
            h = layer(h)
            g = layer.goodness(h).detach()
            goodness_list.append(g)
            h = h.detach()
        return goodness_list
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor,
                   collab_mode: str = 'none') -> Dict[str, float]:
        """
        训练一步
        
        Args:
            collab_mode: 'none' (Original FF), 'all' (γ=所有其他层), 'previous' (γ=前驱层)
        """
        # 创建正负样本
        pos_data = self.embed_label(images, labels, wrong=False)
        neg_data = self.embed_label(images, labels, wrong=True)
        
        # Layer Collaboration: 预计算所有 goodness
        if collab_mode != 'none':
            pos_goodness_all = self.compute_all_goodness(pos_data)
            neg_goodness_all = self.compute_all_goodness(neg_data)
        
        losses = {}
        pos_input = pos_data
        neg_input = neg_data
        
        for i, layer in enumerate(self.layers):
            # 前向传播
            pos_output = layer(pos_input)
            neg_output = layer(neg_input)
            
            pos_g = layer.goodness(pos_output)
            neg_g = layer.goodness(neg_output)
            
            # 计算 γ
            if collab_mode == 'all':
                gamma_pos = sum(g for j, g in enumerate(pos_goodness_all) if j != i)
                gamma_neg = sum(g for j, g in enumerate(neg_goodness_all) if j != i)
            elif collab_mode == 'previous':
                gamma_pos = sum(pos_goodness_all[:i]) if i > 0 else torch.zeros_like(pos_g)
                gamma_neg = sum(neg_goodness_all[:i]) if i > 0 else torch.zeros_like(neg_g)
            else:
                gamma_pos = gamma_neg = None
            
            # 计算损失并更新
            loss = layer.ff_loss(pos_g, neg_g, gamma_pos, gamma_neg)
            
            layer.optimizer.zero_grad()
            loss.backward()
            layer.optimizer.step()
            
            losses[f'layer_{i}'] = loss.item()
            
            # Detach for next layer
            pos_input = pos_output.detach()
            neg_input = neg_output.detach()
        
        return losses
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """预测（枚举所有标签，选择 goodness 最高的）"""
        batch_size = images.size(0)
        best_goodness = torch.full((batch_size,), float('-inf'), device=images.device)
        predictions = torch.zeros(batch_size, dtype=torch.long, device=images.device)
        
        for label in range(self.num_classes):
            labels = torch.full((batch_size,), label, device=images.device)
            x = self.embed_label(images, labels)
            
            total_goodness = torch.zeros(batch_size, device=images.device)
            h = x
            for layer in self.layers:
                h = layer(h)
                total_goodness += layer.goodness(h)
            
            better = total_goodness > best_goodness
            predictions[better] = label
            best_goodness[better] = total_goodness[better]
        
        return predictions


class BPNetwork(nn.Module):
    """标准反向传播网络（baseline）"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], 
                 num_classes: int, lr: float = 0.001):
        super().__init__()
        
        layers = []
        sizes = [input_size] + hidden_sizes + [num_classes]
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:  # No ReLU after last layer
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.view(x.size(0), -1))
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取倒数第二层特征（用于迁移）"""
        with torch.no_grad():
            h = x.view(x.size(0), -1)
            for layer in list(self.network.children())[:-1]:  # Exclude last linear
                h = layer(h)
            return h
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images).argmax(dim=1)


# ============================================================
# Transfer Learning
# ============================================================

class LinearClassifier(nn.Module):
    """线性分类头（用于 Linear Probe）"""
    
    def __init__(self, feature_dim: int, num_classes: int, lr: float = 0.01):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def train_step(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        outputs = self.forward(features)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward(features).argmax(dim=1)


def extract_features(model: nn.Module, dataloader: DataLoader, 
                     device: str, model_type: str = 'ff') -> Tuple[torch.Tensor, torch.Tensor]:
    """提取特征"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            
            if model_type == 'ff':
                features = model.get_features(images.view(images.size(0), -1))
            else:
                features = model.get_features(images)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_features), torch.cat(all_labels)


def transfer_linear_probe(source_model: nn.Module, 
                          target_train_loader: DataLoader,
                          target_test_loader: DataLoader,
                          device: str,
                          model_type: str = 'ff',
                          epochs: int = 100,
                          target_num_classes: int = 10) -> Dict:
    """
    Linear Probe 迁移测试
    
    冻结预训练特征提取器，仅训练新的线性分类头
    """
    print(f"\n{'='*60}")
    print("Linear Probe Transfer")
    print(f"{'='*60}")
    
    # 提取特征
    train_features, train_labels = extract_features(
        source_model, target_train_loader, device, model_type)
    test_features, test_labels = extract_features(
        source_model, target_test_loader, device, model_type)
    
    feature_dim = train_features.size(1)
    print(f"Feature dimension: {feature_dim}")
    
    # 训练线性分类器
    classifier = LinearClassifier(feature_dim, target_num_classes).to(device)
    
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    history = {'train_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # 训练
        classifier.train()
        # Mini-batch training
        indices = torch.randperm(len(train_features))
        batch_losses = []
        for i in range(0, len(indices), 64):
            batch_idx = indices[i:i+64]
            loss = classifier.train_step(train_features[batch_idx], train_labels[batch_idx])
            batch_losses.append(loss)
        
        # 评估
        classifier.eval()
        with torch.no_grad():
            preds = classifier.predict(test_features)
            acc = (preds == test_labels).float().mean().item()
        
        history['train_loss'].append(np.mean(batch_losses))
        history['test_acc'].append(acc)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {np.mean(batch_losses):.4f} | Acc: {acc*100:.2f}%")
    
    return {
        'final_accuracy': history['test_acc'][-1],
        'best_accuracy': max(history['test_acc']),
        'history': history
    }


# ============================================================
# Training & Evaluation
# ============================================================

def train_ff_network(model: FFNetwork, train_loader: DataLoader, 
                     test_loader: DataLoader, device: str,
                     epochs: int, collab_mode: str = 'none') -> Dict:
    """训练 FF 网络"""
    history = {'train_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            losses = model.train_step(images, labels, collab_mode)
            epoch_losses.append(sum(losses.values()))
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model.predict(images)
                correct += (preds == labels).sum().item()
                total += len(labels)
        
        acc = correct / total
        history['train_loss'].append(np.mean(epoch_losses))
        history['test_acc'].append(acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {np.mean(epoch_losses):.4f} | Acc: {acc*100:.2f}%")
    
    return history


def train_bp_network(model: BPNetwork, train_loader: DataLoader,
                     test_loader: DataLoader, device: str,
                     epochs: int) -> Dict:
    """训练 BP 网络"""
    history = {'train_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            loss = model.train_step(images, labels)
            epoch_losses.append(loss)
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model.predict(images)
                correct += (preds == labels).sum().item()
                total += len(labels)
        
        acc = correct / total
        history['train_loss'].append(np.mean(epoch_losses))
        history['test_acc'].append(acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {np.mean(epoch_losses):.4f} | Acc: {acc*100:.2f}%")
    
    return history


# ============================================================
# CKA Analysis
# ============================================================

def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """计算 CKA (Centered Kernel Alignment)"""
    def centering_matrix(n):
        return torch.eye(n) - torch.ones(n, n) / n
    
    def hsic(K, L):
        n = K.shape[0]
        H = centering_matrix(n).to(K.device)
        return torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)
    
    K = X @ X.T
    L = Y @ Y.T
    
    hsic_KL = hsic(K, L)
    hsic_KK = hsic(K, K)
    hsic_LL = hsic(L, L)
    
    return (hsic_KL / torch.sqrt(hsic_KK * hsic_LL)).item()


def analyze_representations(ff_model: FFNetwork, bp_model: BPNetwork,
                            dataloader: DataLoader, device: str) -> Dict:
    """分析 FF vs BP 的表征"""
    print("\nAnalyzing representations (CKA)...")
    
    # 提取特征
    ff_features, _ = extract_features(ff_model, dataloader, device, 'ff')
    bp_features, _ = extract_features(bp_model, dataloader, device, 'bp')
    
    # 子采样（CKA 计算量大）
    n_samples = min(1000, len(ff_features))
    indices = torch.randperm(len(ff_features))[:n_samples]
    
    ff_sub = ff_features[indices].to(device)
    bp_sub = bp_features[indices].to(device)
    
    cka_score = compute_cka(ff_sub, bp_sub)
    print(f"  CKA (FF vs BP features): {cka_score:.4f}")
    
    return {'cka_ff_bp': cka_score}


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(config: ExperimentConfig) -> Dict:
    """运行完整实验"""
    print("="*70)
    print("LAYER COLLABORATION + TRANSFER LEARNING EXPERIMENT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Source: {config.source_dataset}")
    print(f"  Target: {config.target_dataset}")
    print(f"  Device: {config.device}")
    print(f"  Seed: {config.seed}")
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    device = config.device
    
    # 加载数据集
    print("\n" + "-"*60)
    print("Loading datasets...")
    
    source_train, input_size, source_classes = get_dataset(config.source_dataset, train=True)
    source_test, _, _ = get_dataset(config.source_dataset, train=False)
    target_train, _, target_classes = get_dataset(config.target_dataset, train=True)
    target_test, _, _ = get_dataset(config.target_dataset, train=False)
    
    source_train_loader = DataLoader(source_train, batch_size=config.batch_size, shuffle=True)
    source_test_loader = DataLoader(source_test, batch_size=config.batch_size)
    target_train_loader = DataLoader(target_train, batch_size=config.batch_size, shuffle=True)
    target_test_loader = DataLoader(target_test, batch_size=config.batch_size)
    
    print(f"  Source: {len(source_train)} train, {len(source_test)} test, {source_classes} classes")
    print(f"  Target: {len(target_train)} train, {len(target_test)} test, {target_classes} classes")
    
    results = {}
    
    # ============================================================
    # 1. Train BP Baseline
    # ============================================================
    print("\n" + "="*60)
    print("TRAINING: BP Baseline")
    print("="*60)
    
    torch.manual_seed(config.seed)
    bp_model = BPNetwork(input_size, config.hidden_sizes, source_classes, lr=config.lr_bp).to(device)
    bp_history = train_bp_network(bp_model, source_train_loader, source_test_loader, 
                                   device, config.pretrain_epochs)
    
    results['bp'] = {
        'source_accuracy': bp_history['test_acc'][-1],
        'pretrain_history': bp_history,
    }
    
    # Transfer
    bp_transfer = transfer_linear_probe(
        bp_model, target_train_loader, target_test_loader, 
        device, 'bp', config.transfer_epochs, target_classes)
    results['bp']['transfer_accuracy'] = bp_transfer['final_accuracy']
    results['bp']['transfer_history'] = bp_transfer['history']
    
    # ============================================================
    # 2. Train Original FF
    # ============================================================
    print("\n" + "="*60)
    print("TRAINING: Original Forward-Forward")
    print("="*60)
    
    torch.manual_seed(config.seed)
    ff_orig = FFNetwork(input_size, config.hidden_sizes, source_classes, 
                        config.threshold, config.lr_ff).to(device)
    ff_orig_history = train_ff_network(ff_orig, source_train_loader, source_test_loader,
                                       device, config.pretrain_epochs, collab_mode='none')
    
    results['ff_original'] = {
        'source_accuracy': ff_orig_history['test_acc'][-1],
        'pretrain_history': ff_orig_history,
    }
    
    # Transfer
    ff_orig_transfer = transfer_linear_probe(
        ff_orig, target_train_loader, target_test_loader,
        device, 'ff', config.transfer_epochs, target_classes)
    results['ff_original']['transfer_accuracy'] = ff_orig_transfer['final_accuracy']
    results['ff_original']['transfer_history'] = ff_orig_transfer['history']
    
    # ============================================================
    # 3. Train Layer Collab FF (γ = all)
    # ============================================================
    print("\n" + "="*60)
    print("TRAINING: Layer Collaboration FF (γ = all other layers)")
    print("="*60)
    
    torch.manual_seed(config.seed)
    ff_collab_all = FFNetwork(input_size, config.hidden_sizes, source_classes,
                              config.threshold, config.lr_ff).to(device)
    ff_collab_all_history = train_ff_network(ff_collab_all, source_train_loader, source_test_loader,
                                              device, config.pretrain_epochs, collab_mode='all')
    
    results['ff_collab_all'] = {
        'source_accuracy': ff_collab_all_history['test_acc'][-1],
        'pretrain_history': ff_collab_all_history,
    }
    
    # Transfer
    ff_collab_all_transfer = transfer_linear_probe(
        ff_collab_all, target_train_loader, target_test_loader,
        device, 'ff', config.transfer_epochs, target_classes)
    results['ff_collab_all']['transfer_accuracy'] = ff_collab_all_transfer['final_accuracy']
    results['ff_collab_all']['transfer_history'] = ff_collab_all_transfer['history']
    
    # ============================================================
    # 4. Train Layer Collab FF (γ = previous)
    # ============================================================
    print("\n" + "="*60)
    print("TRAINING: Layer Collaboration FF (γ = previous layers)")
    print("="*60)
    
    torch.manual_seed(config.seed)
    ff_collab_prev = FFNetwork(input_size, config.hidden_sizes, source_classes,
                               config.threshold, config.lr_ff).to(device)
    ff_collab_prev_history = train_ff_network(ff_collab_prev, source_train_loader, source_test_loader,
                                               device, config.pretrain_epochs, collab_mode='previous')
    
    results['ff_collab_prev'] = {
        'source_accuracy': ff_collab_prev_history['test_acc'][-1],
        'pretrain_history': ff_collab_prev_history,
    }
    
    # Transfer
    ff_collab_prev_transfer = transfer_linear_probe(
        ff_collab_prev, target_train_loader, target_test_loader,
        device, 'ff', config.transfer_epochs, target_classes)
    results['ff_collab_prev']['transfer_accuracy'] = ff_collab_prev_transfer['final_accuracy']
    results['ff_collab_prev']['transfer_history'] = ff_collab_prev_transfer['history']
    
    # ============================================================
    # 5. Random Baseline (no pretraining)
    # ============================================================
    print("\n" + "="*60)
    print("BASELINE: Random Initialization")
    print("="*60)
    
    torch.manual_seed(config.seed)
    ff_random = FFNetwork(input_size, config.hidden_sizes, source_classes,
                          config.threshold, config.lr_ff).to(device)
    
    # No training - just transfer
    ff_random_transfer = transfer_linear_probe(
        ff_random, target_train_loader, target_test_loader,
        device, 'ff', config.transfer_epochs, target_classes)
    results['random'] = {
        'transfer_accuracy': ff_random_transfer['final_accuracy'],
        'transfer_history': ff_random_transfer['history']
    }
    
    # ============================================================
    # 6. CKA Analysis
    # ============================================================
    print("\n" + "="*60)
    print("REPRESENTATION ANALYSIS")
    print("="*60)
    
    # Use subset for analysis
    analysis_loader = DataLoader(Subset(source_test, list(range(1000))), batch_size=256)
    
    # FF Original vs BP
    cka_orig = analyze_representations(ff_orig, bp_model, analysis_loader, device)
    results['ff_original']['cka_vs_bp'] = cka_orig['cka_ff_bp']
    
    # FF Collab All vs BP
    cka_collab = analyze_representations(ff_collab_all, bp_model, analysis_loader, device)
    results['ff_collab_all']['cka_vs_bp'] = cka_collab['cka_ff_bp']
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Method':<30} {'Source Acc':>12} {'Transfer Acc':>14} {'CKA vs BP':>12}")
    print("-" * 70)
    
    for name, res in results.items():
        source = res.get('source_accuracy', 'N/A')
        transfer = res.get('transfer_accuracy', 'N/A')
        cka = res.get('cka_vs_bp', 'N/A')
        
        source_str = f"{source*100:.2f}%" if isinstance(source, float) else source
        transfer_str = f"{transfer*100:.2f}%" if isinstance(transfer, float) else transfer
        cka_str = f"{cka:.4f}" if isinstance(cka, float) else cka
        
        print(f"{name:<30} {source_str:>12} {transfer_str:>14} {cka_str:>12}")
    
    # Transfer improvement analysis
    print("\n" + "-"*60)
    print("TRANSFER IMPROVEMENT ANALYSIS")
    print("-"*60)
    
    random_baseline = results['random']['transfer_accuracy']
    bp_transfer = results['bp']['transfer_accuracy']
    
    for name in ['ff_original', 'ff_collab_all', 'ff_collab_prev']:
        transfer_acc = results[name]['transfer_accuracy']
        gain_vs_random = (transfer_acc - random_baseline) * 100
        gap_vs_bp = (bp_transfer - transfer_acc) * 100
        print(f"{name}:")
        print(f"  Transfer gain vs random: {gain_vs_random:+.2f}%")
        print(f"  Gap vs BP baseline: {gap_vs_bp:.2f}%")
    
    # Check hypotheses
    print("\n" + "-"*60)
    print("HYPOTHESIS VERIFICATION")
    print("-"*60)
    
    h1 = results['ff_collab_all']['transfer_accuracy'] > results['ff_original']['transfer_accuracy']
    print(f"H1 (Collab > Original): {'✓ SUPPORTED' if h1 else '✗ NOT SUPPORTED'}")
    print(f"    Original: {results['ff_original']['transfer_accuracy']*100:.2f}%")
    print(f"    Collab:   {results['ff_collab_all']['transfer_accuracy']*100:.2f}%")
    
    h3 = results['ff_collab_all'].get('cka_vs_bp', 0) > results['ff_original'].get('cka_vs_bp', 0)
    print(f"H3 (Collab CKA > Original CKA): {'✓ SUPPORTED' if h3 else '✗ NOT SUPPORTED'}")
    print(f"    Original CKA: {results['ff_original'].get('cka_vs_bp', 'N/A')}")
    print(f"    Collab CKA:   {results['ff_collab_all'].get('cka_vs_bp', 'N/A')}")
    
    # Save results
    results['config'] = asdict(config)
    results['timestamp'] = datetime.now().isoformat()
    
    return results


def save_results(results: Dict, config: ExperimentConfig):
    """保存结果"""
    os.makedirs(config.results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config.source_dataset}_{config.target_dataset}_seed{config.seed}_{timestamp}.json"
    filepath = os.path.join(config.results_dir, filename)
    
    # Convert numpy/torch to python types
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: {filepath}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Layer Collab + Transfer Learning Experiment")
    parser.add_argument('--config', type=str, default='mnist_fmnist',
                        choices=list(CONFIGS.keys()),
                        help='Experiment configuration')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pretrain-epochs', type=int, default=None,
                        help='Override pretrain epochs')
    parser.add_argument('--transfer-epochs', type=int, default=None,
                        help='Override transfer epochs')
    parser.add_argument('--full', action='store_true',
                        help='Run full experiment (more epochs)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (fewer epochs)')
    
    args = parser.parse_args()
    
    # Get config
    config = CONFIGS[args.config]
    config.seed = args.seed
    
    # Override epochs if specified
    if args.pretrain_epochs:
        config.pretrain_epochs = args.pretrain_epochs
    if args.transfer_epochs:
        config.transfer_epochs = args.transfer_epochs
    
    if args.quick:
        config.pretrain_epochs = 5
        config.transfer_epochs = 10
    elif args.full:
        config.pretrain_epochs = 100
        config.transfer_epochs = 150
    
    # Run experiment
    results = run_experiment(config)
    
    # Save results
    save_results(results, config)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
