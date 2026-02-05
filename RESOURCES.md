# FF 研究资源汇总

> 由小队调研完成，2026-02-05

---

## 📚 文献调研完成清单

| 文档 | 内容 | 关键发现 |
|------|------|---------|
| `brenig2023_analysis.md` | FF迁移学习失败分析 | 迁移性能落后BP 38.9%，根因是逐层损失丢弃信息 |
| `lorberbom2024_layer_collab.md` | Layer Collaboration机制 | MNIST误差 3.3%→2.1%，**但未测迁移学习** |
| `opensource_survey.md` | 9个开源仓库分析 | CwComp(AAAI24)+SCFF(Nature25)最重要 |
| `latest_research_2024_2025.md` | 14篇最新论文 | 负样本系统对比仍是空白！ |
| `comprehensive_survey.md` | 综合综述 | 完整FF发展历史 |
| `predictive_coding_feedback.md` | 预测编码联系 | 理论背景 |
| `adversarial_robustness.md` | 对抗鲁棒性 | FF可能天然更鲁棒 |

---

## 🔑 核心发现

### 1. 研究空白（我们的机会）

| 空白 | 说明 | 我们填补 |
|------|------|---------|
| **负样本策略系统对比** | 没人做过 10+ 策略的 head-to-head 对比 | ✅ 正在进行 |
| **Layer Collab + 迁移学习** | AAAI 2024 没测迁移 | ✅ 正在设计 |
| **负样本属性分析** | 什么使负样本"好"？ | ✅ 框架已建立 |

### 2. 关键数据（已验证）

| 指标 | 值 | 意义 |
|------|---|------|
| FF vs BP Layer 2 CKA | **0.038** | 高层完全不同 |
| FF L0↔L2 Self-CKA | **0.025** | 灾难性层间断裂 |
| BP 最小跨层 CKA | 0.36 | 14× 高于 FF |
| FF Self-CKA 平均 | 0.264 | 层间信息断裂 |
| BP Self-CKA 平均 | 0.592 | 信息流畅通 |

---

## 📦 可复用代码资源

### 官方实现
| 仓库 | 来源 | 用途 |
|------|------|------|
| [CwComp](https://github.com/andreaspapac/CwComp) | AAAI 2024 | 无负样本变体 |
| [SCFF](https://github.com/neurophysics-cnrsthales/contrastive-forward-forward) | Nature 2025 | 自对比方法 |

### 社区实现
| 仓库 | Stars | 特点 |
|------|-------|------|
| [mpezeshki/pytorch_forward_forward](https://github.com/mpezeshki/pytorch_forward_forward) | ~1.5k | 最流行基础实现 |
| [loeweX/Forward-Forward](https://github.com/loeweX/Forward-Forward) | ~200 | 代码质量高 |

---

## 🧪 已实现的实验代码

| 代码 | 路径 | 功能 |
|------|------|------|
| CKA 分析 | `analysis/cka_analysis.py` | 表征相似度分析 |
| Linear Probe | `analysis/linear_probe.py` | 特征质量评估 |
| 负样本属性 | `analysis/metrics.py` | hardness/diversity/distribution |
| 策略对比 | `experiments/strategy_comparison.py` | 10策略对比 |
| 迁移实验 | `experiments/transfer_experiment.py` | Layer Collab 迁移 |
| FF Baseline | `experiments/ff_baseline.py` | 基础FF训练 |

---

## 📈 实验状态

### 正在运行
- **策略对比**: label_embedding(38.81%), image_mixing(9.8%失败), 继续中...
- **Layer Collab**: Original(51.74%), Layer Collab(67.43% epoch1)

### 待运行
- 完整迁移学习实验 (MNIST→Fashion-MNIST)
- CIFAR-10 扩展

---

## 📝 关键论文

1. **Hinton 2022** - 原始 FF 论文 (arXiv:2212.13345)
2. **Brenig 2023** - 迁移学习失败分析 (arXiv:2309.11955)
3. **Lorberbom 2024** - Layer Collaboration (AAAI 2024)
4. **SCFF 2025** - 自对比方法 (Nature Communications)
5. **CwComp 2024** - 无负样本变体 (AAAI 2024)
