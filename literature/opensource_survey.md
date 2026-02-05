# FF 开源实现调研报告

> 调研日期：2026-02-05  
> 目标：系统收集和分析 Forward-Forward 算法的开源实现，找到可复用代码

---

## 📊 仓库清单总览

| 仓库 | Stars | 最后更新 | 框架 | 数据集 | 特点 |
|------|-------|----------|------|--------|------|
| [mpezeshki/pytorch_forward_forward](https://github.com/mpezeshki/pytorch_forward_forward) | ~1.5k | 2023-01 | PyTorch | MNIST | 最流行的基础实现 |
| [loeweX/Forward-Forward](https://github.com/loeweX/Forward-Forward) | ~200+ | 2023 | PyTorch | MNIST | 代码质量高，1.45% test error |
| [andreaspapac/CwComp](https://github.com/andreaspapac/CwComp) | - | 2024 | PyTorch | MNIST/FMNIST/CIFAR-10/100 | **AAAI 2024**, CNN+竞争学习 |
| [neurophysics-cnrsthales/contrastive-forward-forward](https://github.com/neurophysics-cnrsthales/contrastive-forward-forward) | - | 2025 | PyTorch | MNIST/CIFAR-10/STL-10/TinyImageNet | **Nature Comm. 2025**, SCFF 官方 |
| [LumenPallidium/backprop-alts](https://github.com/LumenPallidium/backprop-alts) | - | 2025-01 | PyTorch | MNIST | 多种 BP 替代方案对比 |
| [miladsikaroudi/forward-forward-cifar](https://github.com/miladsikaroudi/forward-forward-cifar) | - | 2023 | PyTorch | CIFAR-10/100 | 专注 CIFAR 数据集 |
| [visvig/forward-forward-algorithm](https://github.com/visvig/forward-forward-algorithm) | - | 2023 | PyTorch | MNIST/CIFAR-10 | 含 ConvNet 实验 |
| [dslisleedh/FF-jax](https://github.com/dslisleedh/FF-jax) | - | 2023 | JAX | MNIST | JAX 实现，适合 TPU |
| [rmwkwok/forward_forward_algorithm](https://github.com/rmwkwok/forward_forward_algorithm) | - | 2023 | TensorFlow | MNIST | TF 实现 |

---

## 🔍 核心仓库详细分析

### 1. mpezeshki/pytorch_forward_forward ⭐⭐⭐⭐⭐

**最流行的基础实现**

```
Stars: ~1,500
Language: Python (PyTorch)
Last Update: 2023-01
```

#### 负样本策略
- **Label Embedding**: 将正确/错误标签嵌入图像前 10 个像素
- 正样本: `merge(image, correct_label)`
- 负样本: `merge(image, random_wrong_label)`

#### 支持的数据集
- MNIST only

#### 代码结构
```python
# 核心代码简洁
class Layer(nn.Linear):
    def forward(self, x):
        x_normalized = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return torch.relu(self.bn(super().forward(x_normalized)))
    
    def train(self, x_pos, x_neg):
        # 对比式本地学习
        g_pos = (self.forward(x_pos) ** 2).mean(1)
        g_neg = (self.forward(x_neg) ** 2).mean(1)
        loss = torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold
        ]))).mean()
```

#### 性能
- Train error: 6.75%
- Test error: 6.84%

#### 可复用性评估
| 方面 | 评分 | 说明 |
|------|------|------|
| 代码质量 | ⭐⭐⭐⭐ | 简洁清晰，单文件 |
| 模块化 | ⭐⭐⭐ | 基础模块化 |
| 文档 | ⭐⭐⭐ | README 清晰 |
| 可扩展 | ⭐⭐⭐ | 需要修改支持 CNN |

**推荐借鉴：** Layer 归一化、goodness 计算、基础损失函数

---

### 2. loeweX/Forward-Forward ⭐⭐⭐⭐

**高质量重实现，性能更好**

```
Stars: ~200+
Language: Python (PyTorch)
Last Update: 2023
```

#### 负样本策略
- 同 mpezeshki，但有更好的超参数调优
- One-hot 标签嵌入前 10 像素

#### 性能对比
| 实现 | Test Error |
|------|------------|
| Hinton 原论文 | 1.36% |
| 官方 Matlab | 1.47% |
| loeweX | **1.45%** |

#### 代码特点
- Conda 环境配置完善
- 支持 CUDA 版本配置
- 有完整的训练脚本

**推荐借鉴：** 训练流程、超参数设置

---

### 3. andreaspapac/CwComp ⭐⭐⭐⭐⭐

**AAAI 2024 顶会论文实现，卷积+竞争学习**

```
Paper: AAAI 2024 (Oral + Poster)
Language: Python (PyTorch)
Last Update: 2024
License: MIT
```

#### 核心创新：Channel-wise Competitive (CwC) Loss
- **消除负样本需求**！使用竞争学习替代正负对比
- 引入 CFSE (Channel-wise Feature Separator and Extractor) 模块

#### 负样本策略
- **无需负样本**：使用 channel-wise competitive loss
- 每个通道对应一个类别，通过竞争学习

#### 支持的数据集 & 性能
| Dataset | Test Error | 较 FF 提升 |
|---------|------------|----------|
| MNIST | **0.58%** | 显著 |
| Fashion-MNIST | **7.69%** | 显著 |
| CIFAR-10 | **21.89%** | 显著 |
| CIFAR-100 | **48.77%** | 显著 |

#### 代码结构
```
CwComp/
├── train_main.py      # 训练入口
├── predict_main.py    # 预测和可视化
├── layer_cnn.py       # CNN 层和损失函数 ⭐
├── layer_fc.py        # 全连接层
├── datasets.py        # 数据集处理
└── configure.py       # 配置
```

#### 可复用性评估
| 方面 | 评分 | 说明 |
|------|------|------|
| 代码质量 | ⭐⭐⭐⭐⭐ | 模块化优秀 |
| 创新性 | ⭐⭐⭐⭐⭐ | 无需负样本！ |
| 文档 | ⭐⭐⭐⭐ | 详细 README |
| 可扩展 | ⭐⭐⭐⭐⭐ | 支持多数据集 |

**强烈推荐借鉴：**
- `layer_cnn.py` 中的 CwC Loss 实现
- CFSE 模块设计
- ILT (Iterative Layer Training) 策略

---

### 4. neurophysics-cnrsthales/contrastive-forward-forward ⭐⭐⭐⭐⭐

**Nature Communications 2025 官方实现 (SCFF)**

```
Paper: Nature Communications 16:5978 (2025)
Language: Python (PyTorch)
Python: 3.10.9, CUDA: 11.8
DOI: 10.5281/zenodo.15526033
```

#### 核心创新：Self-Contrastive 自对比
- 灵感来自 SimCLR 等对比学习
- 同一样本的不同增强作为正负对
- **无需标签的无监督学习**

#### 负样本策略
- **Self-Contrastive**: 正样本=弱增强，负样本=强增强
- 适用于多种数据集，无需调整

#### 支持的数据集 & 性能
| Dataset | Accuracy | 方法 |
|---------|----------|------|
| MNIST (MLP) | 98%+ | Greedy |
| MNIST (CNN) | 99%+ | Parallel |
| CIFAR-10 | ~85% | Parallel |
| STL-10 | ~75% | Parallel |
| Tiny ImageNet | ~40% | 2-stage |
| FSDD (Audio) | 支持 | RNN |

#### 训练策略
1. **Greedy Layer-wise**: 逐层贪婪训练
2. **Parallel Training**: 所有层同时训练

#### 代码结构
```
contrastive-forward-forward/
├── SCFF_CIFAR.py           # CIFAR 贪婪训练
├── SCFF_CIFAR_Parallel.py  # CIFAR 并行训练
├── SCFF_STL.py             # STL-10
├── SCFF_MNIST.py           # MNIST MLP
├── SCFF_MNIST_CNN_Parallel.py  # MNIST CNN
├── SCFF_TIMGNET_Parallel.py    # Tiny ImageNet
├── SCFF_FSDD.py            # 音频(RNN)
└── requirements.txt
```

#### 可复用性评估
| 方面 | 评分 | 说明 |
|------|------|------|
| 代码质量 | ⭐⭐⭐⭐ | 清晰但较长 |
| 创新性 | ⭐⭐⭐⭐⭐ | 自对比，Nature 级别 |
| 数据集覆盖 | ⭐⭐⭐⭐⭐ | 最广泛 |
| RNN 支持 | ⭐⭐⭐⭐⭐ | 唯一支持序列数据 |

**强烈推荐借鉴：**
- 自对比负样本生成策略
- Parallel training 实现
- RNN 版本 FF

---

### 5. LumenPallidium/backprop-alts ⭐⭐⭐⭐

**多种 BP 替代方案对比库**

```
Language: Python (PyTorch)
Last Update: 2025-01
```

#### 包含的算法
- Hebbian Learning (FastHebb)
- Predictive Coding (3 种变体)
- Fast Weight Programmers
- Reservoir Computing
- **Forward-Forward** (含 Layer Collaboration)
- PEPITA
- Genetic Algorithms

#### FF 实现特点
- 基于 Hinton 官方 Matlab 源码
- 实现了 Layer Collaboration (arXiv:2305.12393)
- 与其他算法有统一的对比框架

#### 对比结果 (4 层网络, MNIST)
| 算法 | 样本效率 | 时间效率 |
|------|----------|----------|
| Backprop | 最好 | 最快 |
| FF | 中等 | 较慢 |
| Predictive Coding | 好 | 很慢 |
| Reservoir | 中等 | 中等 |

**推荐借鉴：**
- Layer Collaboration 实现
- 统一的对比框架设计

---

## 📚 最新研究进展摘要

### 1. Self-Contrastive Forward-Forward (SCFF)
- **论文**: Nature Communications 16:5978 (2025)
- **arXiv**: 2409.11593
- **核心**: 用自对比学习生成正负样本，无需标签
- **数据集**: MNIST, CIFAR-10, STL-10, Tiny ImageNet, FSDD
- **代码**: [官方仓库](https://github.com/neurophysics-cnrsthales/contrastive-forward-forward)
- **意义**: 解决了 FF 在无监督场景的负样本问题

### 2. Mono-Forward Algorithm
- **论文**: arXiv:2501.09238 (2025-01)
- **核心**: 纯本地学习，单次前向传播
- **创新**: 消除负样本需求，使用本地误差信号
- **性能**: 在 MNIST, FMNIST, CIFAR-10/100 上匹配或超越 BP
- **优势**: 内存使用更均匀，更好的并行性

### 3. Distance-Forward Learning
- **论文**: arXiv:2408.14925 (2024-08)
- **核心**: 用距离度量学习重构 FF
- **创新**: 
  - 基于质心的度量学习
  - Goodness-based N-pair margin loss
  - Layer-collaboration 策略
- **性能**:
  - MNIST: 99.7%
  - CIFAR-10: 88.2%
  - CIFAR-100: 59%
  - SVHN: 95.9%
  - ImageNette: 82.5%
- **意义**: 目前 FF 在视觉任务的 SOTA

### 4. Scalable Forward-Forward
- **论文**: arXiv:2501.03176 (2025-01)
- **核心**: 扩展 FF 到现代 CNN 架构
- **支持架构**: MobileNetV3, ResNet18
- **创新**: 
  - 新的卷积层损失计算方式
  - Hybrid 设计：block 内用 BP，block 间用 FF
- **性能**: 与标准 BP 相当，甚至在某些情况下更好
- **意义**: 首次将 FF 扩展到大规模现代架构

### 5. Convolutional Channel-wise Competitive (CwComp)
- **论文**: AAAI 2024 (Oral + Poster)
- **arXiv**: 2312.12668
- **核心**: Channel-wise 竞争学习消除负样本需求
- **性能**: MNIST 0.58%, CIFAR-10 21.89%
- **代码**: [CwComp](https://github.com/andreaspapac/CwComp)

---

## 🔧 与我们已有代码的对比

当前实现 (`experiments/ff_baseline.py`) 分析：

| 特性 | 我们的实现 | 最佳开源实现 |
|------|-----------|-------------|
| Layer 归一化 | ✅ L2 norm | ✅ 相同 |
| Goodness 计算 | ✅ 平方和 | ✅ 相同 |
| 损失函数 | ✅ Softplus | ✅ 相同 |
| 负样本策略 | 待扩展 | CwC(无需), SCFF(自对比) |
| CNN 支持 | ❌ | ✅ CwComp, SCFF |
| 数据集 | MNIST | CIFAR-10/100, STL-10 等 |
| Layer Collaboration | ❌ | ✅ backprop-alts |
| Parallel Training | ❌ | ✅ SCFF |

---

## 📋 推荐借鉴的代码模块

### 优先级 1 (立即借鉴)

1. **CwComp - Channel-wise Competitive Loss**
   - 文件: `layer_cnn.py`
   - 理由: 消除负样本需求，性能最好
   ```python
   # 核心思想：每个通道对应一个类别
   # 通过竞争学习自动区分
   ```

2. **SCFF - Self-Contrastive 数据增强**
   - 文件: `SCFF_CIFAR.py`
   - 理由: 无监督场景最佳方案
   ```python
   # 正样本：弱增强 (crop, flip)
   # 负样本：强增强 (color jitter, blur)
   ```

### 优先级 2 (近期借鉴)

3. **CwComp - CFSE 模块**
   - 理由: CNN 特征分离的关键模块

4. **SCFF - Parallel Training**
   - 理由: 加速训练

5. **backprop-alts - Layer Collaboration**
   - 理由: 减少 greedy learning 的信息损失

### 优先级 3 (长期参考)

6. **Scalable FF 的 Hybrid 设计**
   - 理由: 大规模模型适用

7. **Distance-Forward 的度量学习框架**
   - 理由: 理论更完善

---

## 🎯 行动建议

### 短期 (1-2 周)
1. 将 CwC Loss 整合到 `ff_baseline.py`
2. 添加 SCFF 的自对比数据增强
3. 扩展支持 CIFAR-10

### 中期 (1 个月)
4. 实现 CNN 版本的 FF (参考 CwComp)
5. 添加 Layer Collaboration
6. 对比各负样本策略性能

### 长期 (2-3 个月)
7. 探索 Scalable FF 的 hybrid 架构
8. 研究 Distance-Forward 的度量学习方法
9. 在更大数据集上验证

---

## 📎 附录：关键论文链接

1. [Hinton 原始论文](https://arxiv.org/abs/2212.13345) - The Forward-Forward Algorithm
2. [SCFF](https://www.nature.com/articles/s41467-025-61037-0) - Nature Communications 2025
3. [CwComp](https://arxiv.org/abs/2312.12668) - AAAI 2024
4. [Distance-Forward](https://arxiv.org/abs/2408.14925) - arXiv 2024
5. [Scalable FF](https://arxiv.org/abs/2501.03176) - arXiv 2025
6. [Mono-Forward](https://arxiv.org/abs/2501.09238) - arXiv 2025
7. [Layer Collaboration](https://arxiv.org/abs/2305.12393) - arXiv 2023
