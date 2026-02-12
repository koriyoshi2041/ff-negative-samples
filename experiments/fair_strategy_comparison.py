#!/usr/bin/env python3
"""
公平的负样本策略对比
====================
统一标准：
- 正样本: x + 正确label（所有策略相同）
- 负样本: 不同生成方式（策略差异点）
- 评估: 标准FF方法（10个label取最高goodness）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import json, time, os

# Device
device = torch.device("mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Config
EPOCHS = 1000
LR = 0.03
THRESHOLD = 2.0
SEED = 42
torch.manual_seed(SEED)

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

x_train = train_data.data.float().view(-1, 784) / 255.0
y_train = train_data.targets
x_test = test_data.data.float().view(-1, 784) / 255.0
y_test = test_data.targets

# Normalize
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / (std + 1e-8)
x_test = (x_test - mean) / (std + 1e-8)
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# ============== Model ==============
class FFLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(out_f)
    def forward(self, x): 
        return self.ln(self.relu(self.linear(x)))
    def goodness(self, x): 
        return (x**2).mean(dim=1)  # MEAN not SUM!

class FFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FFLayer(784, 500), FFLayer(500, 500)])
    def forward(self, x, upto=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if upto is not None and i == upto:
                break
        return x

# ============== Label Embedding ==============
def overlay_label(x, y):
    """统一的label embedding方法"""
    x = x.clone()
    x[:, :10] = 0
    scale = x.max().item()
    x[range(len(y)), y] = scale
    return x

# ============== 负样本策略（统一接口）==============

def neg_wrong_label(x, y):
    """策略1: 错误标签（Hinton原始方法）"""
    wrong_y = (y + torch.randint(1, 10, y.shape, device=device)) % 10
    return overlay_label(x, wrong_y)

def neg_class_confusion(x, y):
    """策略2: 类别混淆（不同图片+当前标签）"""
    perm = torch.randperm(len(x), device=device)
    return overlay_label(x[perm], y)  # 不同图片，保持原标签

def neg_hybrid_mix(x, y):
    """策略3: 混合图像+随机标签"""
    perm = torch.randperm(len(x), device=device)
    mixed = 0.5 * x + 0.5 * x[perm]
    wrong_y = (y + torch.randint(1, 10, y.shape, device=device)) % 10
    return overlay_label(mixed, wrong_y)

def neg_same_class_diff_image(x, y):
    """策略4: 同类别不同图片+错误标签"""
    # 找同类别的其他图片
    perm = torch.randperm(len(x), device=device)
    wrong_y = (y + torch.randint(1, 10, y.shape, device=device)) % 10
    return overlay_label(x[perm], wrong_y)

def neg_noise_augmented(x, y):
    """策略5: 噪声增强+错误标签"""
    noise = torch.randn_like(x) * 0.3
    noisy_x = x + noise
    wrong_y = (y + torch.randint(1, 10, y.shape, device=device)) % 10
    return overlay_label(noisy_x, wrong_y)

def neg_masked(x, y):
    """策略6: 遮挡+错误标签"""
    mask = (torch.rand_like(x) > 0.3).float()
    masked_x = x * mask
    wrong_y = (y + torch.randint(1, 10, y.shape, device=device)) % 10
    return overlay_label(masked_x, wrong_y)

# ============== Training ==============
def train_ff(model, x_pos, x_neg, epochs=EPOCHS):
    for li, layer in enumerate(model.layers):
        opt = optim.Adam(layer.parameters(), lr=LR)
        with torch.no_grad():
            h_pos = x_pos if li == 0 else model.forward(x_pos, li-1)
            h_neg = x_neg if li == 0 else model.forward(x_neg, li-1)
        
        for ep in range(epochs):
            out_pos, out_neg = layer(h_pos), layer(h_neg)
            g_pos, g_neg = layer.goodness(out_pos), layer.goodness(out_neg)
            loss = (torch.log(1 + torch.exp(-(g_pos - THRESHOLD))).mean() + 
                    torch.log(1 + torch.exp(g_neg - THRESHOLD)).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if (ep + 1) % 200 == 0:
                print(f"  Layer {li}, Epoch {ep+1}: loss={loss.item():.4f}")
        
        with torch.no_grad():
            h_pos, h_neg = layer(h_pos), layer(h_neg)

def evaluate(model, x, y):
    """标准FF评估：10个label取最高goodness"""
    model.eval()
    with torch.no_grad():
        goodness_scores = []
        for label in range(10):
            x_labeled = overlay_label(x, torch.full((len(x),), label, device=device))
            h = model(x_labeled)
            g = (h ** 2).mean(dim=1)
            goodness_scores.append(g)
        predictions = torch.stack(goodness_scores, dim=1).argmax(dim=1)
        return (predictions == y).float().mean().item()

# ============== Run All Strategies ==============
strategies = {
    "1_wrong_label": neg_wrong_label,
    "2_class_confusion": neg_class_confusion,
    "3_hybrid_mix": neg_hybrid_mix,
    "4_same_class_diff_img": neg_same_class_diff_image,
    "5_noise_augmented": neg_noise_augmented,
    "6_masked": neg_masked,
}

results = {}
os.makedirs("results", exist_ok=True)

for name, neg_fn in strategies.items():
    print(f"\n{'='*60}")
    print(f"Strategy: {name}")
    print(f"{'='*60}")
    
    model = FFNet().to(device)
    
    # 正样本（所有策略统一）
    x_pos = overlay_label(x_train, y_train)
    # 负样本（各策略不同）
    x_neg = neg_fn(x_train, y_train)
    
    t0 = time.time()
    train_ff(model, x_pos, x_neg, EPOCHS)
    train_time = time.time() - t0
    
    train_acc = evaluate(model, x_train, y_train)
    test_acc = evaluate(model, x_test, y_test)
    
    print(f"\n{name}: Train={train_acc*100:.2f}%, Test={test_acc*100:.2f}%, Time={train_time:.0f}s")
    
    results[name] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "time": train_time
    }
    
    # 每个策略完成后保存
    with open("results/fair_strategy_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

print("\n" + "="*60)
print("FINAL RESULTS (Fair Comparison)")
print("="*60)
for name, r in sorted(results.items(), key=lambda x: -x[1]["test_acc"]):
    print(f"{name}: {r['test_acc']*100:.2f}%")
