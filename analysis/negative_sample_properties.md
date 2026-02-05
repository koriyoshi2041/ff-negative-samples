# 负样本策略有效性分析框架

> **目标**: 不只是比较准确率，而是理解"为什么"某些负样本策略更有效

---

## 1. 理论框架：什么使负样本"好"？

### 1.1 对比学习的视角

基于文献调研（SimCLR, MoCo, [Robinson et al. 2021]），好的负样本应该满足：

```
好的负样本 = f(Hardness, Diversity, Validity)
```

| 特性 | 定义 | 为什么重要 |
|------|------|-----------|
| **难度 (Hardness)** | 负样本与正样本的区分难度 | 提供更强的学习信号，迫使模型学习细粒度特征 |
| **多样性 (Diversity)** | 负样本在特征空间的覆盖程度 | 防止过拟合特定类型的负样本 |
| **有效性 (Validity)** | 负样本与真实分布的匹配度 | 避免学到 spurious correlations |

### 1.2 Forward-Forward 的特殊性

FF 与标准对比学习有关键差异：

```
标准对比学习:  L = -log[exp(z·z⁺) / Σexp(z·zᵢ⁻)]  # 同一嵌入空间
Forward-Forward: L = -log σ(g(pos) - θ) - log σ(θ - g(neg))  # 分离处理
```

**关键洞察**:
1. FF 不直接在嵌入空间比较正负样本
2. 而是学习一个判别器：`goodness(x) > θ` for positive
3. 这意味着负样本的作用是"标定"goodness 的基线，而非直接对比

**假设**: FF 中负样本的作用机制可能与对比学习不同：
- 对比学习：负样本定义"推开"的方向
- FF：负样本定义"低 goodness"的特征模式

### 1.3 三种假设

**H1: 难度假设** - 更难的负样本总是更好
```
如果 |g(pos) - g(neg)| 越小（neg 更难），则学习信号越强
但：可能导致训练不稳定
```

**H2: 多样性假设** - 多样性比难度更重要
```
如果 Var(neg_features) 越大，则模型泛化越好
但：可能导致学习信号稀释
```

**H3: 互补假设** - 两者存在最优平衡点
```
Performance = α·Hardness + β·Diversity + γ·Hardness×Diversity
存在某个 (α*, β*, γ*) 使性能最优
```

---

## 2. 特性量化方法

### 2.1 难度 (Hardness) 度量

#### 方法 1: Goodness-Based Hardness
```python
def compute_goodness_hardness(model, pos_samples, neg_samples):
    """
    基于 goodness 分数的难度度量
    难度 = 正负样本 goodness 差距的倒数
    """
    with torch.no_grad():
        pos_goodness = model.compute_goodness(pos_samples)  # (B,)
        neg_goodness = model.compute_goodness(neg_samples)  # (B,)
    
    # 难度：差距越小越难
    gap = pos_goodness - neg_goodness
    hardness = 1.0 / (gap.abs() + 1e-6)
    
    return {
        'mean_hardness': hardness.mean().item(),
        'std_hardness': hardness.std().item(),
        'mean_gap': gap.mean().item(),
        'min_gap': gap.min().item(),  # 最难的样本
    }
```

#### 方法 2: Feature Distance Hardness
```python
def compute_feature_hardness(model, pos_samples, neg_samples):
    """
    基于特征距离的难度度量
    难度 = 正负样本特征距离的倒数
    """
    with torch.no_grad():
        pos_features = model.get_features(pos_samples)  # (B, D)
        neg_features = model.get_features(neg_samples)  # (B, D)
    
    # L2 距离
    l2_dist = torch.norm(pos_features - neg_features, dim=1)
    
    # 余弦相似度
    cos_sim = F.cosine_similarity(pos_features, neg_features)
    
    return {
        'mean_l2_dist': l2_dist.mean().item(),
        'mean_cosine_sim': cos_sim.mean().item(),
        'feature_hardness': (1.0 / (l2_dist + 1e-6)).mean().item(),
    }
```

#### 方法 3: Classification-Based Hardness
```python
def compute_classification_hardness(pos_labels, neg_samples, classifier):
    """
    使用预训练分类器评估负样本的"可信度"
    如果分类器认为负样本属于正确类别，则该负样本很难
    """
    with torch.no_grad():
        logits = classifier(neg_samples)  # (B, num_classes)
        probs = F.softmax(logits, dim=1)
        
        # 负样本被分类为对应正样本标签的概率
        confusion_prob = probs.gather(1, pos_labels.unsqueeze(1))
    
    return {
        'mean_confusion_prob': confusion_prob.mean().item(),
        'max_confusion_prob': confusion_prob.max().item(),
    }
```

### 2.2 多样性 (Diversity) 度量

#### 方法 1: Feature Space Coverage
```python
def compute_feature_diversity(model, neg_samples):
    """
    负样本在特征空间的覆盖范围
    """
    with torch.no_grad():
        neg_features = model.get_features(neg_samples)  # (B, D)
    
    # 类内方差（同一 batch 内）
    intra_variance = neg_features.var(dim=0).mean()
    
    # 特征维度激活分布
    activated_dims = (neg_features.abs() > 0.1).float().mean(dim=0)
    coverage = activated_dims.mean()
    
    return {
        'intra_variance': intra_variance.item(),
        'feature_coverage': coverage.item(),
        'effective_dimensions': (activated_dims > 0.5).sum().item(),
    }
```

#### 方法 2: Pairwise Distance Distribution
```python
def compute_pairwise_diversity(neg_samples, sample_size=1000):
    """
    负样本之间的两两距离分布
    """
    if neg_samples.size(0) > sample_size:
        idx = torch.randperm(neg_samples.size(0))[:sample_size]
        neg_samples = neg_samples[idx]
    
    # 计算两两距离矩阵
    dist_matrix = torch.cdist(neg_samples, neg_samples)
    
    # 取上三角（排除对角线和重复）
    triu_indices = torch.triu_indices(dist_matrix.size(0), dist_matrix.size(1), offset=1)
    pairwise_dists = dist_matrix[triu_indices[0], triu_indices[1]]
    
    return {
        'mean_pairwise_dist': pairwise_dists.mean().item(),
        'std_pairwise_dist': pairwise_dists.std().item(),
        'min_pairwise_dist': pairwise_dists.min().item(),
        'max_pairwise_dist': pairwise_dists.max().item(),
    }
```

#### 方法 3: Entropy-Based Diversity
```python
def compute_entropy_diversity(neg_samples, n_bins=50):
    """
    基于熵的多样性度量
    """
    # 将特征值离散化
    neg_flat = neg_samples.view(-1).cpu().numpy()
    hist, _ = np.histogram(neg_flat, bins=n_bins, density=True)
    hist = hist + 1e-10  # 避免 log(0)
    
    # 计算熵
    entropy = -np.sum(hist * np.log(hist)) / np.log(n_bins)  # 归一化
    
    return {
        'feature_entropy': entropy,
    }
```

### 2.3 分布差距 (Distribution Gap) 度量

#### 方法 1: Goodness Distribution Comparison
```python
def compute_goodness_distribution_gap(model, real_data, neg_samples):
    """
    比较负样本与真实数据的 goodness 分布
    """
    with torch.no_grad():
        real_goodness = model.compute_goodness(real_data)
        neg_goodness = model.compute_goodness(neg_samples)
    
    # KL 散度（近似）
    # 注意：需要将连续值离散化
    real_hist = torch.histc(real_goodness, bins=50)
    neg_hist = torch.histc(neg_goodness, bins=50)
    
    real_hist = real_hist / real_hist.sum() + 1e-10
    neg_hist = neg_hist / neg_hist.sum() + 1e-10
    
    kl_div = (real_hist * torch.log(real_hist / neg_hist)).sum()
    
    # Wasserstein 距离（1D）
    from scipy.stats import wasserstein_distance
    w_dist = wasserstein_distance(
        real_goodness.cpu().numpy(), 
        neg_goodness.cpu().numpy()
    )
    
    return {
        'kl_divergence': kl_div.item(),
        'wasserstein_distance': w_dist,
        'mean_goodness_gap': (real_goodness.mean() - neg_goodness.mean()).item(),
    }
```

#### 方法 2: FID-like Distance
```python
def compute_feature_distribution_gap(model, real_data, neg_samples):
    """
    类似 FID 的特征分布距离
    """
    with torch.no_grad():
        real_features = model.get_features(real_data)
        neg_features = model.get_features(neg_samples)
    
    # 计算均值和协方差
    real_mean = real_features.mean(dim=0)
    neg_mean = neg_features.mean(dim=0)
    
    real_cov = torch.cov(real_features.T)
    neg_cov = torch.cov(neg_features.T)
    
    # Fréchet 距离
    mean_diff = torch.norm(real_mean - neg_mean) ** 2
    
    # 简化版：trace 差
    cov_diff = torch.trace(real_cov + neg_cov - 2 * torch.sqrt(real_cov @ neg_cov + 1e-6 * torch.eye(real_cov.size(0))))
    
    fid_score = mean_diff + cov_diff
    
    return {
        'fid_score': fid_score.item(),
        'mean_diff': mean_diff.item(),
    }
```

---

## 3. 实验设计

### 3.1 实验 A: 特性与性能的相关性分析

**目的**: 验证哪些特性最能预测策略性能

```python
# 实验流程
strategies = [
    'label_embedding', 'image_mixing', 'random_noise',
    'class_confusion', 'self_contrastive', 'masking',
    'layer_wise', 'adversarial', 'hard_mining'
]

for strategy in strategies:
    # 1. 训练模型
    model = train_ff_model(strategy, epochs=50)
    
    # 2. 收集负样本
    neg_samples = strategy.generate(test_images, test_labels)
    
    # 3. 计算所有特性指标
    hardness = compute_all_hardness_metrics(model, test_images, neg_samples)
    diversity = compute_all_diversity_metrics(model, neg_samples)
    distribution = compute_all_distribution_metrics(model, test_images, neg_samples)
    
    # 4. 记录最终性能
    accuracy = evaluate_model(model, test_loader)
    
    # 5. 相关性分析
    results.append({
        'strategy': strategy,
        'accuracy': accuracy,
        **hardness, **diversity, **distribution
    })

# 计算 Pearson/Spearman 相关系数
correlations = compute_correlations(results, target='accuracy')
```

**期望输出**:
- 特性-性能相关性矩阵
- 最重要特性的排序
- 验证/拒绝 H1, H2, H3

### 3.2 实验 B: 训练动态分析

**目的**: 观察不同策略在训练过程中特性的变化

```python
# 每 5 个 epoch 记录一次
checkpoints = [0, 5, 10, 20, 30, 50]

for strategy in strategies:
    dynamics = []
    model = FFModel()
    
    for epoch in range(50):
        train_one_epoch(model, strategy)
        
        if epoch in checkpoints:
            metrics = {
                'epoch': epoch,
                'accuracy': evaluate(model),
                **compute_all_metrics(model, strategy)
            }
            dynamics.append(metrics)
    
    plot_training_dynamics(dynamics, strategy)
```

**观察重点**:
- 难度是否随训练增加（自适应策略）
- 多样性是否保持
- goodness 分布如何演化

### 3.3 实验 C: 消融研究

**目的**: 分离难度和多样性的贡献

```python
# 构造控制实验
def create_controlled_negatives(base_strategy, control='hardness'):
    """
    固定一个特性，改变另一个
    """
    if control == 'hardness':
        # 固定难度，改变多样性
        return [
            ('low_diversity', sample_hard_negatives(n=100)),
            ('medium_diversity', sample_hard_negatives(n=500)),
            ('high_diversity', sample_hard_negatives(n=2000)),
        ]
    elif control == 'diversity':
        # 固定多样性，改变难度
        return [
            ('easy', sample_diverse_negatives(hardness=0.2)),
            ('medium', sample_diverse_negatives(hardness=0.5)),
            ('hard', sample_diverse_negatives(hardness=0.8)),
        ]
```

### 3.4 实验 D: MonoForward 分析

**目的**: 理解为什么无负样本也能工作

**关键问题**: 
- MonoForward 是否隐式地创造了"负样本效应"？
- 什么替代了负样本的作用？

```python
def analyze_mono_forward():
    # 假设: MonoForward 通过以下机制工作
    
    # 1. 类别分离假设
    # 不同类别的 target goodness 不同，相当于隐式对比
    class_targets = mono_strategy.class_targets
    inter_class_distance = compute_pairwise_distances(class_targets)
    
    # 2. 正则化假设
    # loss 函数本身提供了约束，不需要显式负样本
    loss_landscape = analyze_loss_landscape(mono_model)
    
    # 3. 对比: 与显式负样本策略的表征质量
    mono_features = extract_features(mono_model, test_data)
    ff_features = extract_features(ff_model, test_data)
    
    cka_score = compute_cka(mono_features, ff_features)
```

---

## 4. 可视化方案

### 4.1 特性空间可视化

```python
def plot_strategy_properties(results):
    """
    3D 散点图: 难度 vs 多样性 vs 性能
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for result in results:
        ax.scatter(
            result['hardness'], 
            result['diversity'],
            result['accuracy'],
            s=100,
            label=result['strategy']
        )
    
    ax.set_xlabel('Hardness')
    ax.set_ylabel('Diversity')
    ax.set_zlabel('Accuracy')
```

### 4.2 Goodness 分布对比

```python
def plot_goodness_distributions(strategies, model, data):
    """
    violin plot 对比不同策略的 goodness 分布
    """
    fig, axes = plt.subplots(2, len(strategies), figsize=(20, 8))
    
    for i, strategy in enumerate(strategies):
        pos_goodness = compute_goodness(model, data)
        neg_goodness = compute_goodness(model, strategy.generate(data))
        
        axes[0, i].violinplot([pos_goodness.numpy()])
        axes[0, i].set_title(f'{strategy.name}\nPositive')
        
        axes[1, i].violinplot([neg_goodness.numpy()])
        axes[1, i].set_title('Negative')
```

### 4.3 特性演化追踪

```python
def plot_property_evolution(dynamics):
    """
    折线图: 训练过程中特性的变化
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = [d['epoch'] for d in dynamics]
    
    axes[0, 0].plot(epochs, [d['accuracy'] for d in dynamics])
    axes[0, 0].set_title('Accuracy')
    
    axes[0, 1].plot(epochs, [d['hardness'] for d in dynamics])
    axes[0, 1].set_title('Hardness')
    
    axes[1, 0].plot(epochs, [d['diversity'] for d in dynamics])
    axes[1, 0].set_title('Diversity')
    
    axes[1, 1].plot(epochs, [d['gap'] for d in dynamics])
    axes[1, 1].set_title('Distribution Gap')
```

---

## 5. 文献对比

### 5.1 对比学习中的发现

| 发现来源 | 核心结论 | FF 验证状态 |
|----------|---------|------------|
| SimCLR | 大 batch size 提供更多负样本 → 更好 | 待验证 |
| MoCo | Memory bank 比 in-batch 更好 | 待验证 |
| Robinson 2021 | Hard negatives 显著提升性能 | 待验证 |
| Hard Negative Mixing | 合成困难负样本有效 | 待验证 |
| Curriculum Learning | 从易到难的负样本更好 | 待验证 |

### 5.2 FF 的独特发现

| 策略 | 与对比学习的差异 |
|------|-----------------|
| Label Embedding | 改变输入分布，非特征空间对比 |
| Image Mixing | 像素级混合，非表示级混合 |
| MonoForward | 完全消除负样本 |

---

## 6. 核心问题回答框架

### Q1: 是否"更难"的负样本总是更好？

**预测答案**: 不完全是

**原因**:
1. 太难的负样本可能导致训练不稳定（正负 goodness 重叠）
2. 可能存在"甜蜜点"——中等难度最优
3. FF 的局部学习机制可能对难度更敏感

**验证实验**: 实验 C（消融研究）

### Q2: 多样性和难度哪个更重要？

**预测答案**: 取决于任务阶段

**原因**:
- 早期训练：多样性可能更重要（建立基础表示）
- 后期训练：难度可能更重要（精细化决策边界）
- Curriculum 学习的成功暗示两者都重要，但需要顺序

**验证实验**: 实验 B（训练动态）

### Q3: 为什么 MonoForward（无负样本）也能工作？

**假设答案**:

1. **隐式对比假设**: 
   - 不同类别有不同的 target patterns
   - 学习过程中类别间自然形成对比

2. **正则化替代假设**:
   - Loss 函数（MSE to target）提供了类似的约束
   - 不需要显式的"推开"负样本

3. **信息论视角**:
   - 标签信息足以定义 goodness 的目标
   - 负样本只是定义目标的一种方式

**验证实验**: 实验 D

---

## 7. 完整代码模板

```python
# analysis/negative_property_analyzer.py

class NegativePropertyAnalyzer:
    """
    分析负样本策略特性的工具类
    """
    
    def __init__(self, model, strategies, test_loader):
        self.model = model
        self.strategies = strategies
        self.test_loader = test_loader
        
    def run_full_analysis(self):
        """运行完整分析流程"""
        results = []
        
        for strategy_name, strategy in self.strategies.items():
            print(f"Analyzing {strategy_name}...")
            
            # 收集样本
            pos_samples, neg_samples = [], []
            for images, labels in self.test_loader:
                pos = strategy.create_positive(images, labels)
                neg = strategy.generate(images, labels)
                pos_samples.append(pos)
                neg_samples.append(neg)
            
            pos_samples = torch.cat(pos_samples)
            neg_samples = torch.cat(neg_samples)
            
            # 计算所有指标
            metrics = {
                'strategy': strategy_name,
                **self.compute_hardness(pos_samples, neg_samples),
                **self.compute_diversity(neg_samples),
                **self.compute_distribution_gap(pos_samples, neg_samples),
            }
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def compute_hardness(self, pos, neg):
        # ... 实现所有难度指标
        pass
    
    def compute_diversity(self, neg):
        # ... 实现所有多样性指标
        pass
    
    def compute_distribution_gap(self, pos, neg):
        # ... 实现所有分布差距指标
        pass
```

---

## 8. 执行计划

| 阶段 | 任务 | 时间 | 产出 |
|------|------|------|------|
| 1 | 实现所有度量函数 | 2天 | `analysis/metrics.py` |
| 2 | 运行实验 A（相关性） | 3天 | 相关性矩阵 |
| 3 | 运行实验 B（动态） | 2天 | 动态图表 |
| 4 | 运行实验 C（消融） | 2天 | 消融结果 |
| 5 | 运行实验 D（Mono） | 1天 | Mono 分析 |
| 6 | 撰写报告 | 2天 | 最终报告 |

---

*Created: 2026-02-05*
*Author: Rios (FF Research Subagent)*
*Status: Framework Ready, Experiments Pending*
