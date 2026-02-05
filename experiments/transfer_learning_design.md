# Layer Collaboration + Transfer Learning 实验设计

**研究目标**：验证 Layer Collaboration 机制是否改善 Forward-Forward 算法的迁移学习能力

**背景**：
- Lorberbom et al. (AAAI 2024) 提出 Layer Collaboration FF，显著提升单任务性能
- **但论文未测试迁移学习** — 这是我们填补的空白
- Brenig et al. (2023) 证明原始 FF 迁移性能显著落后于 BP

---

## 1. 核心假设

### 主假设
> **H0**: Layer Collaboration 机制通过促进层间信息流动，使 FF 学到更通用、更可迁移的特征表示

### 子假设

| ID | 假设 | 理论依据 |
|----|------|----------|
| H1 | Layer Collab FF 迁移准确率 > Original FF | 层协作减少信息丢失 |
| H2 | Layer Collab FF 早期层特征更通用 | 全局 goodness 信号指导早期层 |
| H3 | Layer Collab FF 与 BP 的 CKA 相似度更高 | 层级化表示更接近 BP |
| H4 | γ=all 变体迁移性能 ≥ γ=previous 变体 | 双向协作信号更丰富 |

---

## 2. 实验设计

### 2.1 对比组设置

| 模型 | 描述 | 代码实现 |
|------|------|----------|
| **BP Baseline** | 标准反向传播训练 | 全连接网络 + CrossEntropy |
| **Original FF** | Hinton 2022 原始 FF | 逐层独立训练 |
| **Layer Collab FF (γ=all)** | 使用所有层 goodness | AAAI 2024 方法 |
| **Layer Collab FF (γ<t)** | 仅使用前驱层 goodness | AAAI 2024 变体 |
| **Random Init** | 随机初始化（下界） | 无训练 |

### 2.2 数据集选择

参考 Brenig 2023 + 迁移学习最佳实践：

#### 主实验：同源不同语义

| 源数据集 | 目标数据集 | 迁移类型 | 难度 |
|----------|------------|----------|------|
| **MNIST** | **Fashion-MNIST** | 同结构不同语义 | 低 |
| **CIFAR-10** | **CIFAR-100** | 同源不同粒度 | 中 |

#### 扩展实验：跨域迁移

| 源数据集 | 目标数据集 | 迁移类型 | 难度 |
|----------|------------|----------|------|
| CIFAR-10 | STL-10 | 相似类别不同分辨率 | 中-高 |
| SVHN | MNIST | 数字识别跨域 | 中 |

### 2.3 迁移策略

#### 策略 A：完全冻结（Linear Probe）
```
源任务训练 → 冻结所有隐藏层 → 仅训练新分类头
```
- 最严格的迁移测试
- 直接评估特征质量

#### 策略 B：分层冻结
```
源任务训练 → 冻结前 k 层 → 微调后 (n-k) 层 + 新分类头
k ∈ {0, 1, 2, ..., n}
```
- 绘制"冻结层数 vs 性能"曲线
- 揭示哪些层特征更通用

#### 策略 C：全参数微调
```
源任务训练 → 使用较小学习率微调全网络
```
- 评估预训练初始化的价值
- 与随机初始化对比

### 2.4 网络架构

```python
# 统一架构（公平对比）
architecture = {
    'type': 'MLP',
    'input_size': 784 (MNIST/F-MNIST) 或 3072 (CIFAR),
    'hidden_layers': [500, 500, 500],  # 3层便于分析
    'output_size': 10,
    'activation': 'ReLU',
    'threshold': 2.0,  # FF 阈值
}
```

**注意**：FF 需要 layer normalization，BP 不需要

---

## 3. 评估指标

### 3.1 主要指标

| 指标 | 公式/定义 | 用途 |
|------|-----------|------|
| **迁移准确率** | Top-1 Accuracy on target test set | 主要性能指标 |
| **迁移增益** | $\Delta = Acc_{pretrained} - Acc_{random}$ | 预训练的价值 |
| **相对差距** | $(Acc_{BP} - Acc_{FF}) / Acc_{BP}$ | 与 BP baseline 的差距 |

### 3.2 冻结分析指标

| 指标 | 描述 |
|------|------|
| **冻结曲线 AUC** | 冻结 k 层的平均性能 |
| **最优冻结点** | 性能最高时的冻结层数 |
| **衰减斜率** | 冻结更多层时性能下降速率 |

### 3.3 表征分析指标

| 指标 | 工具 | 目的 |
|------|------|------|
| **CKA** | Centered Kernel Alignment | FF vs BP 表征相似度 |
| **Linear Probe** | 线性分类器 | 每层特征质量 |
| **t-SNE** | 降维可视化 | 类别分离度 |

---

## 4. 实验协议

### 4.1 训练参数

```python
# 源任务预训练
pretrain_config = {
    'epochs': 60,
    'batch_size': 64,
    'optimizer': 'Adam',
    'lr_ff': 0.03,      # FF 需要更高学习率
    'lr_bp': 0.001,     # BP 标准学习率
    'threshold': 2.0,   # FF goodness 阈值
}

# 目标任务迁移
transfer_config = {
    'epochs': 100,
    'batch_size': 64,
    'lr_finetune': 0.001,      # 微调学习率（较小）
    'lr_head': 0.01,           # 新分类头学习率（较大）
}
```

### 4.2 重复实验

```python
# 统计显著性
num_seeds = 5  # 最少 5 次独立运行
seeds = [42, 123, 456, 789, 1024]

# 报告
# - Mean ± Std
# - p-value (t-test 或 Mann-Whitney U)
```

### 4.3 公平对比原则

1. **相同架构**：所有方法使用相同层数和神经元数
2. **相同数据**：相同的训练/验证/测试划分
3. **相同预算**：相同 epoch 数（或相同 FLOPs）
4. **相同初始化**：使用相同随机种子

---

## 5. Layer Collab 改善迁移的理论分析

### 5.1 为什么 Original FF 迁移差？

Brenig 2023 诊断：
1. **逐层优化导致信息丢失**：每层只优化自己的 goodness，丢弃"不相关"但可能有用的特征
2. **没有反向信号**：早期层不知道后期层的需求
3. **过于 task-specific**：特征针对当前任务的 goodness 优化

### 5.2 Layer Collab 如何可能改善？

| 机制 | 对迁移的影响 |
|------|-------------|
| **全局 goodness 信号 (γ)** | 每层优化时考虑整体性能，减少局部最优 |
| **层间信息流** | 类似 BP 的反向信号效果（虽然形式不同） |
| **交替训练** | 层之间协调发展，避免冲突特征 |
| **减少贪婪丢弃** | γ 惩罚破坏其他层 goodness 的更新 |

### 5.3 预期结果

**乐观预期**（如果 Layer Collab 有效）：
- 迁移准确率提升 5-15%
- CKA 相似度（与 BP）提升 0.1-0.2
- 冻结曲线更平缓（层特征更通用）

**保守预期**（如果效果有限）：
- 迁移准确率提升 1-5%
- 主要改善在后期层
- 与 BP 差距仍显著

---

## 6. 实验流程

### Phase 1: Baseline 验证 (Week 1)

```
1. 复现 Brenig 2023 的 Original FF 迁移结果
2. 训练 BP baseline
3. 确认实验 pipeline 正常工作
```

**检查点**：Original FF 迁移准确率应比 BP 低 10-20%

### Phase 2: Layer Collab 迁移实验 (Week 2-3)

```
1. 在源任务训练 4 种模型（BP, Original FF, Collab-all, Collab-prev）
2. Linear Probe 迁移测试
3. 分层冻结迁移测试
4. 统计显著性分析
```

### Phase 3: 表征分析 (Week 3-4)

```
1. CKA 分析：FF vs BP 各层相似度
2. Linear Probe：每层特征在目标任务的可分性
3. t-SNE 可视化：类别分离度对比
```

### Phase 4: 扩展实验 (Week 4-5)

```
1. CIFAR-10 → CIFAR-100 实验
2. 跨域迁移测试
3. 敏感性分析（threshold θ, 学习率等）
```

---

## 7. 预期资源与时间

### 7.1 计算资源

```python
# 单次实验估计（MNIST → Fashion-MNIST, 单 GPU）
per_experiment = {
    'pretrain': 60 epochs × 60s/epoch = 1 小时,
    'transfer': 100 epochs × 30s/epoch = 50 分钟,
    'total': ~2 小时/配置,
}

# 完整实验（5 seeds × 4 models × 3 冻结配置）
total_runs = 5 × 4 × 3 = 60 runs
total_time = 60 × 2 = 120 小时 ≈ 5 天（单 GPU）

# 推荐：并行 4 GPU 或使用 M2 MPS
```

### 7.2 数据集大小

| 数据集 | 训练集 | 测试集 | 图像尺寸 |
|--------|--------|--------|----------|
| MNIST | 60,000 | 10,000 | 28×28 |
| Fashion-MNIST | 60,000 | 10,000 | 28×28 |
| CIFAR-10 | 50,000 | 10,000 | 32×32 |
| CIFAR-100 | 50,000 | 10,000 | 32×32 |

**可靠性分析**：
- 60,000 训练样本足够得出统计显著结论
- 5 次独立运行 + t-test 可检测 2-3% 的性能差异

### 7.3 时间估计

| 阶段 | 时间 | 并行化后 |
|------|------|----------|
| Phase 1 (Baseline) | 2 天 | 1 天 |
| Phase 2 (主实验) | 5 天 | 2 天 |
| Phase 3 (分析) | 3 天 | 2 天 |
| Phase 4 (扩展) | 4 天 | 2 天 |
| **总计** | **14 天** | **7 天** |

---

## 8. 代码结构

```
ff-research/experiments/
├── transfer_experiment.py      # 主实验脚本
├── models/
│   ├── ff_original.py         # Original FF
│   ├── ff_collab.py           # Layer Collab FF
│   └── bp_baseline.py         # BP baseline
├── data/
│   └── datasets.py            # 数据加载
├── analysis/
│   ├── cka_analysis.py        # CKA 计算
│   ├── linear_probe.py        # Linear Probe
│   └── visualization.py       # 可视化
├── configs/
│   ├── mnist_fmnist.yaml      # MNIST→F-MNIST 配置
│   └── cifar10_100.yaml       # CIFAR-10→100 配置
└── results/
    ├── tables/                # 结果表格
    └── figures/               # 图表
```

---

## 9. 关键问题 & 风险

### 9.1 可能的挑战

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Layer Collab 对迁移无帮助 | 主假设失败 | 仍有学术价值（阴性结果） |
| 超参数敏感 | 结果不稳定 | 网格搜索 + 敏感性分析 |
| 计算时间过长 | 延期 | 减少 seeds / 并行化 |
| 与 BP 差距仍太大 | 实用价值有限 | 探索其他改进方向 |

### 9.2 开放问题

1. **γ 的最优计算方式**？论文测试了 all 和 previous，是否有更好的方案？
2. **是否需要针对迁移的特殊训练策略**？例如：渐进式协作、warm-up 等
3. **CNN 版本**？目前只测 MLP，CNN 可能更有实际价值

---

## 10. 参考文献

1. **Lorberbom et al. (2024)** - Layer Collaboration in the Forward-Forward Algorithm, AAAI
2. **Brenig et al. (2023)** - A Study of Forward-Forward Algorithm for Self-Supervised Learning, arXiv
3. **Hinton (2022)** - The Forward-Forward Algorithm: Some Preliminary Investigations, arXiv
4. **Yosinski et al. (2014)** - How transferable are features in deep neural networks?, NeurIPS
5. **Kornblith et al. (2019)** - Similarity of Neural Network Representations Revisited, ICML

---

*设计完成：2026-02-05*
*作者：Clawd (Subagent)*
*状态：待实施*
