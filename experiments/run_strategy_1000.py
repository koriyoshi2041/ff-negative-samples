"""
负样本策略对比实验 - 1000 epochs完整版
确保正确的goodness计算(mean)和label embedding(x.max())
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import json
import time
import sys
import os

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数
EPOCHS_PER_LAYER = 1000
BATCH_SIZE = 50000  # Full batch
HIDDEN_SIZES = [784, 500, 500]
THRESHOLD = 2.0
LR = 0.03
SEED = 42

torch.manual_seed(SEED)

# 加载数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

x_train = train_dataset.data.float().view(-1, 784) / 255.0
y_train = train_dataset.targets
x_test = test_dataset.data.float().view(-1, 784) / 255.0
y_test = test_dataset.targets

# 标准化
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / (std + 1e-8)
x_test = (x_test - mean) / (std + 1e-8)

x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

print(f"Train: {x_train.shape}, Test: {x_test.shape}")

class FFLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(out_features)
        
    def forward(self, x):
        return self.ln(self.relu(self.linear(x)))
    
    def goodness(self, x):
        # CRITICAL: Use mean, not sum!
        return (x ** 2).mean(dim=1)

class FFNet(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList([
            FFLayer(layer_sizes[i], layer_sizes[i+1]) 
            for i in range(len(layer_sizes)-1)
        ])
        
    def forward(self, x, layer_idx=None):
        if layer_idx is not None:
            for i in range(layer_idx + 1):
                x = self.layers[i](x)
            return x
        for layer in self.layers:
            x = layer(x)
        return x

def overlay_label(x, y, scale=None):
    """CRITICAL: Use x.max() for label embedding, not 1.0!"""
    x = x.clone()
    x[:, :10] = 0
    if scale is None:
        scale = x.max().item()
    x[range(len(y)), y] = scale
    return x

def train_ff(model, x_pos, x_neg, epochs, verbose=True):
    """逐层贪婪训练"""
    for layer_idx, layer in enumerate(model.layers):
        optimizer = optim.Adam(layer.parameters(), lr=LR)
        
        # 获取当前层输入
        with torch.no_grad():
            if layer_idx == 0:
                h_pos, h_neg = x_pos, x_neg
            else:
                h_pos = model.forward(x_pos, layer_idx - 1)
                h_neg = model.forward(x_neg, layer_idx - 1)
        
        for epoch in range(epochs):
            # Forward through current layer
            out_pos = layer(h_pos)
            out_neg = layer(h_neg)
            
            g_pos = layer.goodness(out_pos)
            g_neg = layer.goodness(out_neg)
            
            # FF Loss
            loss_pos = torch.log(1 + torch.exp(-(g_pos - THRESHOLD))).mean()
            loss_neg = torch.log(1 + torch.exp(g_neg - THRESHOLD)).mean()
            loss = loss_pos + loss_neg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if verbose and (epoch + 1) % 100 == 0:
                acc_pos = (g_pos > THRESHOLD).float().mean().item()
                acc_neg = (g_neg < THRESHOLD).float().mean().item()
                print(f"  Layer {layer_idx}, Epoch {epoch+1}: loss={loss.item():.4f}, "
                      f"pos_acc={acc_pos:.3f}, neg_acc={acc_neg:.3f}")
        
        # Update h for next layer
        with torch.no_grad():
            h_pos = layer(h_pos)
            h_neg = layer(h_neg)

def get_accuracy(model, x, y):
    """计算准确率 - 测试所有10个label embedding"""
    model.eval()
    with torch.no_grad():
        goodness_per_label = []
        for label in range(10):
            x_test_labeled = overlay_label(x, torch.full((len(x),), label, device=device))
            h = model(x_test_labeled)
            g = (h ** 2).mean(dim=1)
            goodness_per_label.append(g)
        
        goodness_matrix = torch.stack(goodness_per_label, dim=1)
        predictions = goodness_matrix.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()
    return accuracy

# ========== 负样本策略 ==========

def generate_label_embedding_neg(x, y):
    """Hinton原始方法: 错误标签"""
    wrong_labels = (y + torch.randint(1, 10, y.shape, device=device)) % 10
    return overlay_label(x, wrong_labels)

def generate_image_mixing_neg(x, y):
    """图像混合: 两张不同类别图片混合"""
    perm = torch.randperm(len(x), device=device)
    # 确保混合的是不同类别
    mask = (y == y[perm])
    while mask.any():
        new_perm = torch.randperm(mask.sum().item(), device=device)
        perm[mask] = perm[mask][new_perm]
        mask = (y == y[perm])
    
    alpha = 0.5
    x_mixed = alpha * x + (1 - alpha) * x[perm]
    return x_mixed

def generate_random_noise_neg(x, y):
    """随机噪声: 匹配统计特性"""
    noise = torch.randn_like(x) * x.std() + x.mean()
    return noise

# ========== 主实验 ==========

results = {}

strategies = [
    ("label_embedding", generate_label_embedding_neg, True),
    ("image_mixing", generate_image_mixing_neg, False),
    ("random_noise", generate_random_noise_neg, False),
]

for name, neg_fn, uses_labels in strategies:
    print(f"\n{'='*60}")
    print(f"Strategy: {name}")
    print(f"{'='*60}")
    
    # 初始化模型
    model = FFNet(HIDDEN_SIZES).to(device)
    
    # 生成正负样本
    x_pos = overlay_label(x_train, y_train)
    x_neg = neg_fn(x_train, y_train)
    
    # 训练
    start_time = time.time()
    train_ff(model, x_pos, x_neg, EPOCHS_PER_LAYER, verbose=True)
    train_time = time.time() - start_time
    
    # 评估
    train_acc = get_accuracy(model, x_train, y_train)
    test_acc = get_accuracy(model, x_test, y_test)
    
    print(f"\n{name}: Train={train_acc*100:.2f}%, Test={test_acc*100:.2f}%, Time={train_time:.1f}s")
    
    results[name] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_time": train_time,
        "uses_labels": uses_labels
    }
    
    # 保存中间结果
    results["metadata"] = {
        "epochs_per_layer": EPOCHS_PER_LAYER,
        "batch_size": BATCH_SIZE,
        "architecture": HIDDEN_SIZES,
        "threshold": THRESHOLD,
        "learning_rate": LR,
        "device": str(device),
        "seed": SEED
    }
    
    with open("results/strategy_comparison_1000ep.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to results/strategy_comparison_1000ep.json")

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
for name in ["label_embedding", "image_mixing", "random_noise"]:
    if name in results:
        r = results[name]
        print(f"{name}: {r['test_acc']*100:.2f}%")
