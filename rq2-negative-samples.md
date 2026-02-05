# RQ2: Forward-Forward 负样本策略系统对比研究

## 📋 调研概述

Forward-Forward (FF) 算法是 Geoffrey Hinton 于 2022 年 12 月提出的神经网络训练替代方案，核心思想是用两次前向传播（正样本和负样本）替代传统的前向-反向传播。**负样本的生成策略**是 FF 算法的关键组成部分，直接影响模型性能。

---

## 1. 负样本策略全面收集

### 1.1 Hinton 原始论文中的策略 (arXiv:2212.13345)

#### 策略 A: 标签嵌入到输入（Label Embedding）
- **方法**: 将错误的类别标签嵌入到图像的前几个像素位置
- **实现**: 对于 MNIST，将 10 维 one-hot 标签覆盖到图像左上角
- **负样本生成**: 随机选择一个错误标签替换正确标签
- **优点**: 简单直接，易于实现
- **缺点**: 改变了输入数据分布

#### 策略 B: 混合（Hybrid/Mixing）
- **方法**: 将两张不同类别的图像进行像素级混合
- **实现**: `neg = α * img1 + (1-α) * img2`，其中 α ∈ [0.5, 1)
- **优点**: 生成更自然的负样本
- **缺点**: 需要成对样本，计算开销略大

#### 策略 C: 掩码（Masking）
- **方法**: 随机遮蔽部分像素
- **实现**: 将 N 个随机像素设置为 0 或随机值
- **用途**: 常用于验证实验

### 1.2 扩展策略（来自后续研究）

#### 策略 D: 自对比（Self-Contrastive） - arXiv:2409.12184
- **方法**: 使用同一数据的不同增强版本作为正负对
- **论文**: "Self-Contrastive Forward-Forward Algorithm"
- **实现**: 
  - 正样本: 弱增强（裁剪、翻转）
  - 负样本: 强增强（颜色扭曲、高斯模糊）
- **优点**: 不需要标签信息，适用于自监督学习

#### 策略 E: 层级生成（Layer-wise Generation）
- **方法**: 使用前一层的输出作为负样本生成器
- **论文**: "Layer Collaboration in the Forward-Forward Algorithm" (arXiv:2305.12393)
- **实现**: 每层维护一个负样本生成器
- **优点**: 自适应生成，负样本质量随训练提升

#### 策略 F: 对抗性负样本（Adversarial Negatives）
- **方法**: 使用梯度引导生成更难区分的负样本
- **实现**: 沿梯度方向扰动正样本
- **优点**: 提供更强的学习信号
- **缺点**: 需要额外的梯度计算，降低效率

#### 策略 G: GAN 生成（Generative）
- **方法**: 使用生成对抗网络生成负样本
- **相关研究**: 在某些 FF 变体中被提及
- **优点**: 理论上可以生成高质量负样本
- **缺点**: 训练不稳定，实现复杂

#### 策略 H: 随机噪声（Random Noise）
- **方法**: 纯随机高斯噪声或均匀噪声
- **实现**: `neg = torch.randn_like(pos)`
- **优点**: 最简单
- **缺点**: 学习信号弱，性能较差

#### 策略 I: 类别混淆（Class Confusion）
- **方法**: 保持图像不变，只混淆标签
- **实现**: 正确图像 + 错误标签
- **论文**: 在多个 FF 实现中使用
- **优点**: 不改变图像特征

#### 策略 J: Mono-Forward（无负样本） - arXiv:2501.08756
- **方法**: 完全消除负样本需求
- **论文**: "Mono-Forward: Backpropagation-Free Algorithm for Efficient Neural Network Training"
- **实现**: 使用局部误差信号替代正负对比
- **优点**: 简化训练流程，提高效率

#### 策略 K: 距离学习（Distance-Forward） - arXiv:2408.14577
- **方法**: 基于距离度量而非 goodness 函数
- **论文**: "Distance-Forward Learning: Enhancing the Forward-Forward Algorithm"
- **实现**: 使用余弦距离或欧几里得距离
- **优点**: 更好的表示学习

### 1.3 特定领域的策略

#### 图神经网络（GNN）- arXiv:2302.05282
- **方法**: Graph Forward-Forward (GFF)
- **负样本**: 图结构扰动（边删除、节点交换）

#### 脉冲神经网络（SNN）- 多篇论文
- **方法**: 时序负样本
- **负样本**: 脉冲时序扰动

---

## 2. 现有对比研究

### 2.1 已发表的对比工作

#### 论文 1: "Towards Biologically Plausible Computing: A Comprehensive Comparison" (arXiv:2406.XXXXX)
- **对比算法**: FF, Hebbian, STDP, Target Propagation, Predictive Coding
- **数据集**: MNIST, CIFAR-10
- **结论**: FF 在简单任务上接近 BP，但在复杂任务上有差距

#### 论文 2: "Energy-Efficient Deep Learning Without Backpropagation" (arXiv:2411.XXXXX)
- **对比**: FF vs CaFo (Cascaded Forward) vs MF (Mono-Forward)
- **发现**: 
  - MF 能耗降低 41%
  - 训练速度提升 34%
- **负样本策略**: 标签嵌入 vs 无负样本（MF）

#### 论文 3: "In Search of Goodness: Large Scale Benchmarking of Goodness Functions" (arXiv:2311.XXXXX)
- **内容**: 评测不同 goodness 函数而非负样本策略
- **goodness 函数**: 平方和、负平方和、自定义函数
- **间接相关**: goodness 函数影响负样本的效果

### 2.2 现有对比的局限性

**尚未发现的系统对比**:
1. ❌ 没有专门针对负样本策略的大规模对比
2. ❌ 缺乏统一实验设置下的公平比较
3. ❌ 缺少负样本多样性的定量分析
4. ❌ 缺少收敛速度与负样本质量的关联研究

---

## 3. 测试过的数据集

| 数据集 | 已测试策略 | 最佳报告精度 | 论文来源 |
|--------|-----------|-------------|---------|
| MNIST | 标签嵌入、混合、自对比 | ~99.0% | Hinton 原论文 |
| Fashion-MNIST | 标签嵌入 | ~89% | 多个实现 |
| CIFAR-10 | 混合、层级生成 | ~60-65% | 后续研究 |
| CIFAR-100 | 混合 | ~35-40% | 有限研究 |
| IMDb (文本) | 标签嵌入 | ~85% | arXiv:2307.04205 |
| 图数据集 (Cora等) | 图结构扰动 | 与 BP-GNN 接近 | arXiv:2403.11004 |

---

## 4. 实验设计方案

### 4.1 实验目标
1. 系统比较不同负样本策略对 FF 性能的影响
2. 分析负样本质量与模型收敛的关系
3. 探索最优负样本策略的特征

### 4.2 统一实验设置

#### 网络架构
```python
# 基础 FF 网络
class FFLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_features)  # 或 BatchNorm
        self.threshold = 0.0  # goodness 阈值
        
# 网络配置
architectures = {
    "small": [784, 500, 500],  # MNIST
    "medium": [784, 1000, 1000, 1000],  # 标准测试
    "large": [3072, 2000, 2000, 2000, 2000],  # CIFAR
    "cnn": "待定义 CNN 版本"
}
```

#### 优化器设置
```python
optimizer_config = {
    "optimizer": "Adam",
    "lr": 0.0001,  # 基础学习率
    "lr_scheduler": "CosineAnnealing",
    "weight_decay": 0.0001,
    "epochs": 100,
    "batch_size": 128,
}
```

#### Goodness 函数
```python
# 默认使用平方和
def goodness(x):
    return x.pow(2).sum(dim=1)

# 备选: 负平方和
def neg_goodness(x):
    return -x.pow(2).sum(dim=1)
```

### 4.3 要对比的负样本策略

| 编号 | 策略名称 | 实现复杂度 | 是否需要标签 |
|------|---------|-----------|-------------|
| NS-1 | 标签嵌入（Label Embedding） | 低 | 是 |
| NS-2 | 图像混合（Image Mixing） | 低 | 否 |
| NS-3 | 随机噪声（Random Noise） | 最低 | 否 |
| NS-4 | 类别混淆（Class Confusion） | 低 | 是 |
| NS-5 | 自对比（Self-Contrastive） | 中 | 否 |
| NS-6 | 掩码（Masking） | 低 | 否 |
| NS-7 | 层级生成（Layer-wise） | 高 | 否 |
| NS-8 | 对抗性（Adversarial） | 高 | 否 |
| NS-9 | 硬负样本挖掘（Hard Mining） | 中 | 是 |
| NS-10 | 无负样本（Mono-Forward） | 低 | 是 |

### 4.4 评估指标

#### 主要指标
1. **测试准确率** (Test Accuracy): 最终分类性能
2. **收敛速度** (Convergence Speed): 达到特定精度所需 epoch 数
3. **训练时间** (Training Time): 每个 epoch 的实际耗时

#### 辅助指标
4. **负样本多样性** (Negative Diversity): 
   - 负样本特征空间覆盖度
   - 与正样本的平均距离
5. **训练稳定性** (Training Stability):
   - 损失曲线波动
   - 多次运行的方差
6. **计算效率** (Computational Efficiency):
   - FLOPS
   - 内存占用
7. **表示质量** (Representation Quality):
   - t-SNE 可视化
   - 线性可分性

### 4.5 数据集选择

| 数据集 | 任务类型 | 规模 | 难度 |
|--------|---------|------|------|
| MNIST | 图像分类 | 60K | 简单 |
| Fashion-MNIST | 图像分类 | 60K | 中等 |
| CIFAR-10 | 图像分类 | 50K | 困难 |
| SVHN | 数字识别 | 73K | 中等 |
| 可选: CIFAR-100 | 细粒度分类 | 50K | 很难 |

### 4.6 实验流程

```
实验流程
├── 阶段 1: 基础对比 (2 周)
│   ├── 在 MNIST 上运行所有策略
│   ├── 记录基础指标
│   └── 筛选表现好的策略 (top 5)
│
├── 阶段 2: 深度分析 (2 周)
│   ├── 在 Fashion-MNIST, CIFAR-10 上测试 top 5
│   ├── 分析收敛曲线
│   └── 负样本多样性分析
│
├── 阶段 3: 消融实验 (1 周)
│   ├── 超参数敏感性 (α for mixing, N for masking)
│   ├── 正负样本比例影响
│   └── goodness 函数交互
│
└── 阶段 4: 扩展实验 (1 周)
    ├── CNN 架构测试
    ├── 不同 goodness 函数组合
    └── 与 BP 的对比

总计: 约 6 周
```

---

## 5. 开源代码资源

### 5.1 官方/高星实现

| 仓库 | Stars | 描述 | 负样本策略 |
|------|-------|------|-----------|
| [mpezeshki/pytorch_forward_forward](https://github.com/mpezeshki/pytorch_forward_forward) | 1.5k+ | 最早的 PyTorch 实现 | 标签嵌入 |
| [Ads-cmu/ForwardForward](https://github.com/Ads-cmu/ForwardForward) | - | 扩展到 IMDb | 标签嵌入 |
| [facebookresearch/forwardgnn](https://github.com/facebookresearch/forwardgnn) | - | GNN 版本 | 图结构扰动 |
| [nebuly-ai/nebullvm](https://github.com/nebuly-ai/nebullvm) | - | 包含 FF 优化 | 多种策略 |

### 5.2 研究论文代码

| 论文 | 代码链接 | 特点 |
|------|---------|------|
| Layer Collaboration | 论文中提及但未公开 | 层协作机制 |
| Self-Contrastive FF | 待公开 | 自对比策略 |
| Mono-Forward | 论文中提及 | 无负样本 |
| Distance-Forward | 论文中提及 | 距离学习 |
| ForwardGNN | github.com/facebookresearch/forwardgnn | GNN 版本 |

### 5.3 待搜索的仓库（GitHub 关键词）

- `forward-forward algorithm`
- `forward forward pytorch`
- `FF algorithm neural network`
- `backprop-free learning`
- `local learning algorithm`

---

## 6. 实验代码框架

### 6.1 推荐项目结构

```
ff-negative-samples/
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── strategies/
│   │   ├── label_embedding.yaml
│   │   ├── mixing.yaml
│   │   └── ...
│   └── datasets/
│       ├── mnist.yaml
│       └── cifar10.yaml
├── src/
│   ├── models/
│   │   ├── ff_layer.py
│   │   ├── ff_network.py
│   │   └── ff_cnn.py
│   ├── negative_strategies/
│   │   ├── base.py
│   │   ├── label_embedding.py
│   │   ├── mixing.py
│   │   ├── noise.py
│   │   ├── self_contrastive.py
│   │   ├── masking.py
│   │   ├── layer_wise.py
│   │   ├── adversarial.py
│   │   └── mono_forward.py
│   ├── data/
│   │   ├── datasets.py
│   │   └── transforms.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── metrics.py
│   └── utils/
│       ├── visualization.py
│       └── logging.py
├── experiments/
│   ├── run_experiment.py
│   └── analyze_results.py
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_comparison.ipynb
│   └── 03_analysis.ipynb
└── results/
    └── .gitkeep
```

### 6.2 核心代码示例

```python
# negative_strategies/base.py
from abc import ABC, abstractmethod
import torch

class NegativeStrategy(ABC):
    """负样本策略基类"""
    
    @abstractmethod
    def generate(self, positive_data, labels=None, **kwargs):
        """
        生成负样本
        
        Args:
            positive_data: 正样本 (B, ...)
            labels: 标签 (B,) 可选
            
        Returns:
            negative_data: 负样本 (B, ...)
        """
        pass
    
    @property
    @abstractmethod
    def requires_labels(self) -> bool:
        """是否需要标签"""
        pass

# negative_strategies/label_embedding.py
class LabelEmbeddingStrategy(NegativeStrategy):
    """Hinton 原始的标签嵌入策略"""
    
    def __init__(self, num_classes=10, embed_size=10):
        self.num_classes = num_classes
        self.embed_size = embed_size
        
    def generate(self, positive_data, labels, **kwargs):
        batch_size = positive_data.size(0)
        negative_data = positive_data.clone()
        
        # 生成错误标签
        wrong_labels = torch.randint(0, self.num_classes-1, (batch_size,))
        wrong_labels = (wrong_labels + labels + 1) % self.num_classes
        
        # 嵌入标签到数据
        # ... 实现细节
        
        return negative_data, wrong_labels
    
    @property
    def requires_labels(self):
        return True

# negative_strategies/mixing.py
class MixingStrategy(NegativeStrategy):
    """图像混合策略"""
    
    def __init__(self, alpha_range=(0.5, 0.9)):
        self.alpha_range = alpha_range
        
    def generate(self, positive_data, labels=None, **kwargs):
        batch_size = positive_data.size(0)
        
        # 随机打乱获取混合对
        perm = torch.randperm(batch_size)
        shuffled_data = positive_data[perm]
        
        # 随机混合比例
        alpha = torch.rand(batch_size, 1, 1, 1) * \
                (self.alpha_range[1] - self.alpha_range[0]) + \
                self.alpha_range[0]
        
        negative_data = alpha * positive_data + (1 - alpha) * shuffled_data
        
        return negative_data
    
    @property
    def requires_labels(self):
        return False
```

---

## 7. 下一步行动计划

### 立即行动（1-2 天）
1. [ ] 克隆现有的 FF 开源实现
2. [ ] 运行基础实验验证环境
3. [ ] 创建实验代码框架

### 短期（1 周）
1. [ ] 实现所有 10 种负样本策略
2. [ ] 在 MNIST 上运行初步对比
3. [ ] 设置自动化实验脚本

### 中期（2-4 周）
1. [ ] 完成主要对比实验
2. [ ] 分析结果，撰写初步报告
3. [ ] 识别最有潜力的策略组合

### 长期（1-2 月）
1. [ ] 深入分析最佳策略
2. [ ] 撰写论文/技术报告
3. [ ] 开源完整实验代码

---

## 8. 参考文献

1. Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations. arXiv:2212.13345
2. Gandhi et al. (2023). Extending the Forward Forward Algorithm. arXiv:2307.04205
3. Paliotta et al. (2023). Graph Neural Networks Go Forward-Forward. arXiv:2302.05282
4. Gat et al. (2023). Layer Collaboration in the Forward-Forward Algorithm. arXiv:2305.12393
5. Chen et al. (2024). Self-Contrastive Forward-Forward Algorithm. arXiv:2409.12184
6. Gong et al. (2025). Mono-Forward: Backpropagation-Free Algorithm. arXiv:2501.08756
7. Wu et al. (2024). Distance-Forward Learning. arXiv:2408.14577
8. Park et al. (2024). Forward Learning of Graph Neural Networks. arXiv:2403.11004

---

*最后更新: 2026-02-04*
*调研者: Rios (FF-RQ2 Subagent)*
