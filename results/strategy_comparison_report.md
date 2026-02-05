# 负样本策略对比实验报告

**生成时间**: 2026-02-05 10:03

## 实验配置

- **数据集**: MNIST
- **网络架构**: 784 → 500 → 500
- **优化器**: Adam (lr=0.03)
- **Batch Size**: 64
- **Epochs**: 10
- **设备**: mps

## 实验进度

- **已完成**: 4/10 策略
- **待完成**: 6/10 策略

## 已完成策略排名

| 排名 | 策略 | 准确率 | 训练时间 | 使用标签嵌入 |
|------|------|--------|----------|-------------|
| 1 | label_embedding | 38.81% | 150.2s | ✅ |
| 2 | class_confusion | 38.81% | 105.5s | ✅ |
| 3 | image_mixing | 9.80% | 100.6s | ❌ |
| 4 | random_noise | 9.80% | 99.4s | ❌ |

## 待完成策略

- **self_contrastive** (in_progress): Strong augmentation as negatives (SCFF) - experiment in progress
- **masking** (pending): Random pixel masking - experiment pending
- **layer_wise** (pending): Layer-specific adaptive negatives - experiment pending
- **adversarial** (pending): Gradient-based perturbation - experiment pending
- **hard_mining** (pending): Select hardest negatives from pool - experiment pending
- **mono_forward** (pending): No negatives variant - experiment pending

## 关键发现

### 🥇 最佳策略: label_embedding

- **准确率**: 38.81%
- **训练时间**: 150.2s
- **描述**: Hinton's original method - embed label in first pixels

### 标签嵌入的重要性

- 使用标签嵌入的策略平均准确率: **38.8%**
- 不使用标签嵌入的策略平均准确率: **9.8%**
- 差距: **29.0** 个百分点

> **结论**: 标签嵌入对于 Forward-Forward 算法的分类性能至关重要。
> 不使用标签嵌入的策略（如 image_mixing, random_noise）达到接近随机水平（~10%），
> 因为网络无法学习将图像与类别关联。

---
*由 Forward-Forward Research 自动生成*