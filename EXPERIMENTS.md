# Experiments Log

> 实验动机、目的和结果的清晰记录

---

## Experiment 1: Negative Sample Strategy Comparison

**状态**: 🔄 Running

**动机**:
- 文献中缺乏系统性的负样本策略对比
- 不同策略对 FF 性能影响未知
- 为后续研究选择最佳策略提供依据

**目的**:
- 系统对比 10 种负样本策略在 MNIST 上的表现
- 测量：准确率、收敛速度、训练时间
- 识别最佳策略和策略特点

**设置**:
- 数据集: MNIST
- 网络: 784 → 500 → 500
- 优化器: Adam, lr=0.03
- Epochs: 10
- 重复: 3 次取平均

**策略列表**:
1. LabelEmbedding (Hinton original)
2. ImageMixing
3. RandomNoise (baseline)
4. ClassConfusion
5. SelfContrastive
6. Masking
7. LayerWise
8. Adversarial
9. HardMining
10. MonoForward (no negatives)

**结果**: ⏳ Pending

---

## Experiment 2: CKA + Linear Probe Representation Analysis

**状态**: 🔄 Running

**动机**:
- Brenig 2023 指出 FF 迁移学习失败与特征质量有关
- 需要定量分析 FF vs BP 学到的表征差异
- 识别 FF 的"弱点层"

**目的**:
- 用 CKA 对比 FF 和 BP 各层表征相似度
- 用 Linear Probe 评估各层特征的分类能力
- 理解 FF 特征是否过于 task-specific

**设置**:
- 数据集: MNIST
- 网络: 784 → 500 → 500
- FF: LabelEmbedding 策略
- BP: CrossEntropy loss
- Epochs: 30

**分析内容**:
1. CKA 热力图
2. Linear Probe 各层准确率
3. t-SNE 特征空间可视化

**预期**:
- FF 后层可能与 BP 差异大
- FF 线性可分性可能逐层下降更快

**结果**: ⏳ Pending

---

## Experiment 3: Layer Collaboration Implementation

**状态**: 🔄 Running

**动机**:
- AAAI 2024 论文提出层协作机制改善 FF
- 论文未测试迁移学习（我们的研究机会）
- 需要先实现才能测试迁移学习

**目的**:
- 正确实现 Layer Collaboration FF
- 验证实现正确性（复现论文结果）
- 为迁移学习实验准备

**核心改动**:
```python
# Original FF
p_i = sigmoid(goodness_i - θ)

# Layer Collab FF  
γ = sum(goodness_j for j != i)  # detached
p_i = sigmoid(goodness_i + γ - θ)
```

**预期结果** (复现论文):
- MNIST error: 3.3% → 2.1%

**结果**: ⏳ Pending

---

## Planned Experiments

### Experiment 4: Layer Collab + Transfer Learning
- **动机**: 填补文献空白
- **目的**: 测试 Layer Collab 能否改善 FF 迁移学习
- **设置**: CIFAR-10 → CIFAR-100 迁移

### Experiment 5: CwComp vs Negative Samples
- **动机**: CwComp 完全消除负样本，对比意义重大
- **目的**: 有负样本 vs 无负样本的系统对比

---

*Last updated: 2026-02-05 09:07 UTC*
