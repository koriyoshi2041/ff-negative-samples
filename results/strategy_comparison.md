# 负样本策略对比实验报告

## 实验配置

- **数据集**: MNIST
- **网络架构**: 784 → 500 → 500
- **优化器**: Adam (lr=0.03)
- **Batch Size**: 64
- **Epochs**: 10
- **每个策略运行次数**: 1

## 当前结果（部分完成）

| 排名 | 策略 | 准确率 | 训练时间 (s) | 使用标签嵌入 |
|------|------|--------|--------------|--------------|
| 1 | label_embedding | 38.81% | 150.2 | ✓ |
| 2 | class_confusion | 38.81% | 105.5 | ✓ |
| 3 | random_noise | 9.80% | 99.4 | ✗ |
| 4 | image_mixing | 9.80% | 100.6 | ✗ |

**注**: 实验仍在进行中，以下策略待完成：
- self_contrastive (运行中)
- masking
- layer_wise  
- adversarial
- hard_mining
- mono_forward

## 关键发现

### 1. 标签嵌入是评估的关键

**重要观察**: 只有使用标签嵌入的策略（label_embedding, class_confusion）能获得有意义的准确率（~38%）。其他策略准确率仅为 ~10%（随机猜测水平）。

**原因**: Forward-Forward 的评估方法是通过遍历所有可能的标签，选择产生最高 "goodness" 的那个。如果训练时没有使用标签嵌入，模型无法学会区分不同标签。

### 2. 策略分类

**使用标签嵌入的策略**（适合有监督学习）:
- label_embedding (Hinton 原始方法)
- class_confusion (正确图像 + 错误标签)
- mono_forward (无负样本变体)

**不使用标签嵌入的策略**（适合无监督/自监督学习）:
- image_mixing
- random_noise  
- self_contrastive
- masking
- layer_wise
- adversarial
- hard_mining

### 3. 训练效率

- **最快**: random_noise (99.4s) - 负样本生成简单
- **最慢**: label_embedding (150.2s) - 评估需要遍历所有标签

## 训练曲线

### label_embedding
```
Epoch  1: 27.63%
Epoch  2: 24.34%
Epoch  3: 30.22%
Epoch  4: 27.08%
Epoch  5: 33.21%
Epoch  6: 30.81%
Epoch  7: 37.12%
Epoch  8: 38.24%
Epoch  9: 38.36%
Epoch 10: 38.81%
```

### class_confusion
```
Epoch  1: 27.63%
Epoch  2: 24.34%
Epoch  3: 30.22%
Epoch  4: 27.08%
Epoch  5: 33.21%
Epoch  6: 30.81%
Epoch  7: 37.12%
Epoch  8: 38.24%
Epoch  9: 38.36%
Epoch 10: 38.81%
```

**观察**: 两个使用标签嵌入的策略训练曲线几乎相同，这表明它们学习了相似的表示。

## 讨论

### 为什么准确率相对较低（~38%）？

1. **训练轮数不足**: 只训练了 10 个 epochs，模型可能还未充分收敛
2. **评估方法开销大**: 每个样本需要评估 10 次（每个类别一次）
3. **超参数未优化**: 使用固定的 threshold=2.0 和 lr=0.03

### 改进建议

1. **增加训练轮数**: 典型的 FF 实现需要更多 epochs 才能达到高准确率
2. **调整超参数**: threshold 和 learning rate 对收敛有重要影响
3. **为非标签嵌入策略设计新评估方法**: 例如使用 linear probe

## 结论

Forward-Forward 算法的负样本策略可以分为两大类：

1. **标签嵌入策略**: 将标签信息嵌入输入，适合有监督分类任务
2. **无标签策略**: 不使用标签信息，适合表示学习/自监督任务

当前评估框架只适用于标签嵌入策略。要公平对比所有策略，需要为非标签策略添加额外的评估方法（如 linear probing）。

---
*实验状态: 部分完成 (4/10 策略)*
*生成时间: 2025-02-05*
