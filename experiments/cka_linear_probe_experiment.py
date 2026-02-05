#!/usr/bin/env python3
"""
CKA + Linear Probe 表征分析实验
================================

对比 FF 和 BP 网络的内部表征：
1. CKA 分析：层间相似度
2. Linear Probe：各层特征质量
3. t-SNE 可视化

架构：784 → 500 → 500 (两个隐藏层)
训练：30 epochs
"""

import sys
import os
sys.path.insert(0, os.path.expanduser('~/Desktop/Rios/ff-experiment/corrected'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import json
from datetime import datetime

from ff_core import FFNetwork, embed_label, normalize

# ============================================================================
# 模型定义
# ============================================================================

class BPNetwork(nn.Module):
    """标准 BP 网络，用于对比"""
    
    def __init__(self, layer_sizes: List[int], lr: float = 0.001):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_layer_activations(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取各层激活"""
        x = images.view(images.size(0), -1)
        activations = {'input': x.clone()}
        
        layer_idx = 0
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations[f'layer_{layer_idx}'] = x.clone()
                layer_idx += 1
        
        return activations
    
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


class FFNetworkWithHooks:
    """FF 网络激活提取器"""
    
    def __init__(self, model: FFNetwork):
        self.model = model
    
    def get_layer_activations(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取各层激活（使用正样本）"""
        batch_size = images.size(0)
        images_flat = images.view(batch_size, -1)
        
        # 正样本：嵌入正确标签
        x = embed_label(images_flat, labels, self.model.num_classes)
        x = normalize(x)
        
        activations = {'input': images_flat.clone()}
        
        for i, layer in enumerate(self.model.layers):
            x = layer.get_output(x)
            activations[f'layer_{i}'] = x.clone()
            x = normalize(x)
        
        return activations


# ============================================================================
# CKA 分析
# ============================================================================

def centering_matrix(n: int) -> torch.Tensor:
    """Centering matrix H = I - 1/n * 11^T"""
    return torch.eye(n) - torch.ones(n, n) / n


def hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """HSIC (Hilbert-Schmidt Independence Criterion)"""
    n = K.shape[0]
    H = centering_matrix(n).to(K.device)
    return torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)


def linear_kernel(X: torch.Tensor) -> torch.Tensor:
    """Linear kernel K = X @ X^T"""
    return X @ X.T


def cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """CKA (Centered Kernel Alignment)"""
    K = linear_kernel(X)
    L = linear_kernel(Y)
    
    hsic_KL = hsic(K, L)
    hsic_KK = hsic(K, K)
    hsic_LL = hsic(L, L)
    
    return (hsic_KL / torch.sqrt(hsic_KK * hsic_LL)).item()


def minibatch_cka(X: torch.Tensor, Y: torch.Tensor, batch_size: int = 512) -> float:
    """Minibatch CKA for memory efficiency"""
    n = X.shape[0]
    if n <= batch_size:
        return cka(X, Y)
    
    indices = torch.randperm(n)[:batch_size]
    return cka(X[indices], Y[indices])


# ============================================================================
# Linear Probe
# ============================================================================

def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       max_iter: int = 1000) -> Tuple[float, float]:
    """训练 linear probe 并返回准确率"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(
        max_iter=max_iter,
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    
    return train_acc, test_acc


# ============================================================================
# 数据收集
# ============================================================================

def collect_activations(model, dataloader, device: str, model_type: str = 'ff',
                        labels_for_ff: torch.Tensor = None) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    """收集所有层的激活"""
    all_activations = {}
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Collecting {model_type.upper()} activations"):
            images, labels = images.to(device), labels.to(device)
            
            if model_type == 'ff':
                batch_acts = model.get_layer_activations(images, labels)
            else:
                batch_acts = model.get_layer_activations(images)
            
            for name, act in batch_acts.items():
                if name not in all_activations:
                    all_activations[name] = []
                all_activations[name].append(act.cpu())
            
            all_labels.append(labels.cpu().numpy())
    
    features = {name: torch.cat(acts, dim=0) for name, acts in all_activations.items()}
    labels = np.concatenate(all_labels, axis=0)
    
    return features, labels


# ============================================================================
# 可视化
# ============================================================================

def plot_cka_heatmap(cka_matrix: np.ndarray, ff_names: List[str], bp_names: List[str],
                     save_path: str):
    """绘制 CKA 热力图"""
    plt.figure(figsize=(10, 8))
    
    row_labels = [f"FF {l.replace('layer_', 'L')}" for l in ff_names]
    col_labels = [f"BP {l.replace('layer_', 'L')}" for l in bp_names]
    
    ax = sns.heatmap(cka_matrix,
                     xticklabels=col_labels,
                     yticklabels=row_labels,
                     annot=True,
                     fmt='.3f',
                     cmap='RdYlBu_r',
                     vmin=0, vmax=1,
                     cbar_kws={'label': 'CKA Similarity'},
                     annot_kws={'size': 12})
    
    plt.title('CKA Similarity: FF vs BP Networks (MNIST)\n784 → 500 → 500, 30 epochs', fontsize=14)
    plt.xlabel('BP Network Layers', fontsize=12)
    plt.ylabel('FF Network Layers', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_linear_probe_results(ff_results: Dict, bp_results: Dict, save_path: str):
    """绘制 Linear Probe 结果"""
    ff_layers = sorted([k for k in ff_results.keys() if k.startswith('layer')],
                       key=lambda x: int(x.split('_')[1]))
    bp_layers = sorted([k for k in bp_results.keys() if k.startswith('layer')],
                       key=lambda x: int(x.split('_')[1]))
    
    n_layers = min(len(ff_layers), len(bp_layers))
    
    ff_test_accs = [ff_results[l]['test'] * 100 for l in ff_layers[:n_layers]]
    bp_test_accs = [bp_results[l]['test'] * 100 for l in bp_layers[:n_layers]]
    
    # 添加 input layer
    if 'input' in ff_results:
        ff_test_accs.insert(0, ff_results['input']['test'] * 100)
        bp_test_accs.insert(0, bp_results['input']['test'] * 100)
        layer_labels = ['Input'] + [f'Layer {i}' for i in range(n_layers)]
    else:
        layer_labels = [f'Layer {i}' for i in range(n_layers)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(layer_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ff_test_accs, width, label='Forward-Forward',
                   color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, bp_test_accs, width, label='Backpropagation',
                   color='#3498db', edgecolor='black')
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Linear Probe Test Accuracy (%)', fontsize=12)
    ax.set_title('Linear Probe: Layer-wise Feature Quality\n784 → 500 → 500, 30 epochs', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_tsne(ff_activations: Dict, bp_activations: Dict, labels: np.ndarray,
              save_path: str, n_samples: int = 2000):
    """t-SNE 可视化最后一层表征"""
    # 只用最后一层
    ff_layer_names = sorted([k for k in ff_activations.keys() if k.startswith('layer')],
                           key=lambda x: int(x.split('_')[1]))
    bp_layer_names = sorted([k for k in bp_activations.keys() if k.startswith('layer')],
                           key=lambda x: int(x.split('_')[1]))
    
    ff_last = ff_activations[ff_layer_names[-1]].numpy()
    bp_last = bp_activations[bp_layer_names[-1]].numpy()
    
    # 采样
    if len(labels) > n_samples:
        idx = np.random.choice(len(labels), n_samples, replace=False)
        ff_last = ff_last[idx]
        bp_last = bp_last[idx]
        labels_subset = labels[idx]
    else:
        labels_subset = labels
    
    print("Running t-SNE for FF...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    ff_embedded = tsne.fit_transform(ff_last)
    
    print("Running t-SNE for BP...")
    bp_embedded = tsne.fit_transform(bp_last)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(10):
        mask = labels_subset == i
        axes[0].scatter(ff_embedded[mask, 0], ff_embedded[mask, 1],
                       c=[colors[i]], label=str(i), alpha=0.6, s=10)
        axes[1].scatter(bp_embedded[mask, 0], bp_embedded[mask, 1],
                       c=[colors[i]], label=str(i), alpha=0.6, s=10)
    
    axes[0].set_title('FF Last Layer (t-SNE)', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[1].set_title('BP Last Layer (t-SNE)', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Feature Space Visualization: FF vs BP', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# 主实验流程
# ============================================================================

def train_ff_model(train_loader, test_loader, device: str, 
                   hidden_sizes: List[int], epochs: int) -> FFNetwork:
    """训练 FF 网络"""
    print("\n" + "=" * 60)
    print("Training Forward-Forward Network")
    print(f"Architecture: 784 → {' → '.join(map(str, hidden_sizes))}")
    print(f"Epochs: {epochs}")
    print("=" * 60)
    
    model = FFNetwork(input_size=784, hidden_sizes=hidden_sizes,
                      num_classes=10, threshold=2.0, lr=0.03)
    model.to(device)
    
    for epoch in range(epochs):
        epoch_stats = []
        for images, labels in tqdm(train_loader, desc=f"FF Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            stats = model.train_batch(images, labels)
            epoch_stats.append(stats)
        
        # 评估
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model.predict(images)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total * 100
        avg_loss = np.mean([s[0]['loss'] for s in epoch_stats])
        print(f"Epoch {epoch+1:2d}: Test Acc = {acc:.2f}%, Loss = {avg_loss:.4f}")
    
    return model


def train_bp_model(train_loader, test_loader, device: str,
                   layer_sizes: List[int], epochs: int) -> BPNetwork:
    """训练 BP 网络"""
    print("\n" + "=" * 60)
    print("Training Backpropagation Network")
    print(f"Architecture: {' → '.join(map(str, layer_sizes))}")
    print(f"Epochs: {epochs}")
    print("=" * 60)
    
    model = BPNetwork(layer_sizes, lr=0.001)
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, labels in tqdm(train_loader, desc=f"BP Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            loss = model.train_step(images, labels)
            epoch_loss += loss
        
        # 评估
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model.predict(images)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total * 100
        print(f"Epoch {epoch+1:2d}: Test Acc = {acc:.2f}%, Loss = {epoch_loss/len(train_loader):.4f}")
    
    return model


def main():
    """主实验流程"""
    # 配置
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    hidden_sizes = [500, 500]  # 两个隐藏层
    epochs = 30
    
    # 输出目录
    results_dir = os.path.expanduser('~/Desktop/Rios/ff-research/results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # ========================================
    # 1. 训练模型
    # ========================================
    ff_model = train_ff_model(train_loader, test_loader, device, hidden_sizes, epochs)
    bp_model = train_bp_model(train_loader, test_loader, device, 
                              [784] + hidden_sizes + [10], epochs)
    
    # 包装 FF 模型
    ff_wrapped = FFNetworkWithHooks(ff_model)
    
    # 计算最终准确率
    ff_model.eval()
    bp_model.eval()
    
    ff_correct = bp_correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            ff_correct += (ff_model.predict(images) == labels).sum().item()
            bp_correct += (bp_model.predict(images) == labels).sum().item()
            total += labels.size(0)
    
    ff_final_acc = ff_correct / total * 100
    bp_final_acc = bp_correct / total * 100
    print(f"\nFinal Accuracy: FF = {ff_final_acc:.2f}%, BP = {bp_final_acc:.2f}%")
    
    # ========================================
    # 2. 收集激活
    # ========================================
    print("\n" + "=" * 60)
    print("Collecting Activations")
    print("=" * 60)
    
    # 用子集进行 CKA（内存效率）
    cka_subset = Subset(test_dataset, list(range(2000)))
    cka_loader = DataLoader(cka_subset, batch_size=256, shuffle=False)
    
    ff_activations_cka, _ = collect_activations(ff_wrapped, cka_loader, device, 'ff')
    bp_activations_cka, labels_cka = collect_activations(bp_model, cka_loader, device, 'bp')
    
    # 完整训练/测试集用于 linear probe
    ff_train_features, train_labels = collect_activations(ff_wrapped, train_loader, device, 'ff')
    ff_test_features, test_labels = collect_activations(ff_wrapped, test_loader, device, 'ff')
    
    bp_train_features, _ = collect_activations(bp_model, train_loader, device, 'bp')
    bp_test_features, _ = collect_activations(bp_model, test_loader, device, 'bp')
    
    # ========================================
    # 3. CKA 分析
    # ========================================
    print("\n" + "=" * 60)
    print("CKA Analysis")
    print("=" * 60)
    
    ff_layer_names = sorted([k for k in ff_activations_cka.keys() if k.startswith('layer')],
                           key=lambda x: int(x.split('_')[1]))
    bp_layer_names = sorted([k for k in bp_activations_cka.keys() if k.startswith('layer')],
                           key=lambda x: int(x.split('_')[1]))
    
    cka_matrix = np.zeros((len(ff_layer_names), len(bp_layer_names)))
    
    for i, ff_name in enumerate(tqdm(ff_layer_names, desc="Computing CKA")):
        for j, bp_name in enumerate(bp_layer_names):
            ff_act = ff_activations_cka[ff_name].to(device)
            bp_act = bp_activations_cka[bp_name].to(device)
            cka_matrix[i, j] = minibatch_cka(ff_act, bp_act, batch_size=1000)
    
    print("\nCKA Matrix (FF layers × BP layers):")
    print(cka_matrix)
    
    # ========================================
    # 4. Linear Probe
    # ========================================
    print("\n" + "=" * 60)
    print("Linear Probe Analysis")
    print("=" * 60)
    
    ff_probe_results = {}
    bp_probe_results = {}
    
    all_layer_names = ['input'] + sorted([k for k in ff_train_features.keys() if k.startswith('layer')],
                                         key=lambda x: int(x.split('_')[1]))
    
    print("\nFF Network:")
    for name in tqdm(all_layer_names, desc="FF Linear Probe"):
        train_acc, test_acc = train_linear_probe(
            ff_train_features[name].numpy(), train_labels,
            ff_test_features[name].numpy(), test_labels
        )
        ff_probe_results[name] = {'train': train_acc, 'test': test_acc}
        print(f"  {name}: train={train_acc*100:.2f}%, test={test_acc*100:.2f}%")
    
    print("\nBP Network:")
    for name in tqdm(all_layer_names, desc="BP Linear Probe"):
        if name in bp_train_features:
            train_acc, test_acc = train_linear_probe(
                bp_train_features[name].numpy(), train_labels,
                bp_test_features[name].numpy(), test_labels
            )
            bp_probe_results[name] = {'train': train_acc, 'test': test_acc}
            print(f"  {name}: train={train_acc*100:.2f}%, test={test_acc*100:.2f}%")
    
    # ========================================
    # 5. 生成可视化
    # ========================================
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # CKA 热力图
    plot_cka_heatmap(cka_matrix, ff_layer_names, bp_layer_names,
                     os.path.join(results_dir, 'cka_heatmap.png'))
    
    # Linear Probe 结果
    plot_linear_probe_results(ff_probe_results, bp_probe_results,
                              os.path.join(results_dir, 'linear_probe_comparison.png'))
    
    # t-SNE 可视化
    plot_tsne(ff_activations_cka, bp_activations_cka, labels_cka,
              os.path.join(results_dir, 'tsne_visualization.png'))
    
    # ========================================
    # 6. 保存结果
    # ========================================
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    # 保存 JSON 结果
    linear_probe_output = {
        'ff_results': ff_probe_results,
        'bp_results': bp_probe_results,
        'architecture': f"784 → {' → '.join(map(str, hidden_sizes))}",
        'epochs': epochs,
        'ff_final_accuracy': ff_final_acc,
        'bp_final_accuracy': bp_final_acc
    }
    
    with open(os.path.join(results_dir, 'linear_probe_results.json'), 'w') as f:
        json.dump(linear_probe_output, f, indent=2)
    
    # ========================================
    # 7. 分析报告
    # ========================================
    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)
    
    # CKA 分析
    diagonal_cka = np.diag(cka_matrix)
    most_similar_layer = np.argmax(diagonal_cka)
    least_similar_layer = np.argmin(diagonal_cka)
    
    print(f"\nCKA 对角线（同层相似度）: {diagonal_cka}")
    print(f"最相似的层: Layer {most_similar_layer} (CKA = {diagonal_cka[most_similar_layer]:.3f})")
    print(f"最不相似的层: Layer {least_similar_layer} (CKA = {diagonal_cka[least_similar_layer]:.3f})")
    
    # 找出每个 FF 层最相似的 BP 层
    ff_to_bp_mapping = []
    for i in range(len(ff_layer_names)):
        best_bp = np.argmax(cka_matrix[i])
        ff_to_bp_mapping.append((i, best_bp, cka_matrix[i, best_bp]))
    
    print("\nFF 层与 BP 层的最佳匹配:")
    for ff_idx, bp_idx, cka_val in ff_to_bp_mapping:
        print(f"  FF Layer {ff_idx} → BP Layer {bp_idx} (CKA = {cka_val:.3f})")
    
    # Linear Probe 分析
    ff_layer_only = [k for k in ff_probe_results.keys() if k.startswith('layer')]
    ff_accs = [ff_probe_results[l]['test'] * 100 for l in sorted(ff_layer_only, key=lambda x: int(x.split('_')[1]))]
    bp_accs = [bp_probe_results[l]['test'] * 100 for l in sorted([k for k in bp_probe_results.keys() if k.startswith('layer')], key=lambda x: int(x.split('_')[1]))]
    
    ff_worst_layer = np.argmin(ff_accs)
    ff_worst_acc = ff_accs[ff_worst_layer]
    
    print(f"\nLinear Probe 准确率:")
    print(f"  FF: {ff_accs}")
    print(f"  BP: {bp_accs}")
    print(f"\nFF 特征质量最差的层: Layer {ff_worst_layer} ({ff_worst_acc:.2f}%)")
    
    # 生成分析报告
    report = f"""# CKA + Linear Probe 表征分析报告

**实验日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 实验配置

| 参数 | 值 |
|------|-----|
| 架构 | 784 → {' → '.join(map(str, hidden_sizes))} |
| 训练轮数 | {epochs} |
| FF 最终准确率 | {ff_final_acc:.2f}% |
| BP 最终准确率 | {bp_final_acc:.2f}% |
| 设备 | {device} |

## 关键发现

### 1. CKA 分析：FF vs BP 层间相似度

**CKA 矩阵** (FF 行 × BP 列):
```
{np.array2string(cka_matrix, precision=3, separator=', ')}
```

**对角线相似度** (同层对比):
"""
    
    for i, val in enumerate(diagonal_cka):
        report += f"- Layer {i}: CKA = {val:.3f}\n"
    
    report += f"""
**结论**:
- **最相似的层**: FF Layer {most_similar_layer} vs BP Layer {most_similar_layer} (CKA = {diagonal_cka[most_similar_layer]:.3f})
- **最不相似的层**: FF Layer {least_similar_layer} vs BP Layer {least_similar_layer} (CKA = {diagonal_cka[least_similar_layer]:.3f})

### 2. Linear Probe 分析：各层特征质量

| Layer | FF Test Acc | BP Test Acc | Gap |
|-------|-------------|-------------|-----|
"""
    
    for i, (ff_acc, bp_acc) in enumerate(zip(ff_accs, bp_accs)):
        gap = bp_acc - ff_acc
        report += f"| Layer {i} | {ff_acc:.2f}% | {bp_acc:.2f}% | {gap:+.2f}% |\n"
    
    report += f"""
**结论**:
- **FF 特征质量最差的层**: Layer {ff_worst_layer} ({ff_worst_acc:.2f}%)
- **BP 在所有层的特征质量** 均{"高于" if np.mean(bp_accs) > np.mean(ff_accs) else "低于"} FF

### 3. 核心结论

1. **FF 和 BP 哪些层最相似？**
   - Layer {most_similar_layer} 表征最相似 (CKA = {diagonal_cka[most_similar_layer]:.3f})
"""
    
    if most_similar_layer == 0:
        report += "   - 这表明 **早期特征提取** 在两种学习方法中相似\n"
    else:
        report += f"   - 深层的相似性可能源于任务目标的共同约束\n"
    
    report += f"""
2. **FF 哪层特征质量最差？**
   - Layer {ff_worst_layer} 的 linear probe 准确率最低 ({ff_worst_acc:.2f}%)
"""
    
    if ff_worst_layer == len(ff_accs) - 1:
        report += "   - **最后一层** 特征质量下降，可能是 local learning 导致的信息损失\n"
    elif ff_worst_layer == 0:
        report += "   - **第一层** 特征质量较差，可能是 label embedding 干扰了特征学习\n"
    else:
        report += f"   - 中间层（Layer {ff_worst_layer}）特征质量下降，可能存在 **信息瓶颈**\n"
    
    report += f"""
3. **FF vs BP 的根本差异**
   - 平均 CKA 相似度: {np.mean(diagonal_cka):.3f}
   - FF 平均 linear probe 准确率: {np.mean(ff_accs):.2f}%
   - BP 平均 linear probe 准确率: {np.mean(bp_accs):.2f}%
   - 差距: {np.mean(bp_accs) - np.mean(ff_accs):+.2f}%

## 可视化文件

- `cka_heatmap.png` - CKA 相似度热力图
- `linear_probe_comparison.png` - Linear Probe 对比
- `tsne_visualization.png` - 最后一层 t-SNE 可视化

## 数据文件

- `linear_probe_results.json` - 完整数值结果
"""
    
    with open(os.path.join(results_dir, 'representation_analysis.md'), 'w') as f:
        f.write(report)
    
    print(f"\n报告已保存: {os.path.join(results_dir, 'representation_analysis.md')}")
    print("\n实验完成！")
    
    return {
        'cka_matrix': cka_matrix,
        'ff_probe_results': ff_probe_results,
        'bp_probe_results': bp_probe_results,
        'most_similar_layer': most_similar_layer,
        'ff_worst_layer': ff_worst_layer
    }


if __name__ == '__main__':
    main()
