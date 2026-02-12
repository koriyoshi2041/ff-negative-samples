# Forward-Forward Algorithm: Speaker Notes
# Transfer Learning Paradox and Bio-Inspired Solutions

**Presentation Duration:** ~25 minutes (27 slides)
**Target Audience:** ML researchers, neuroscience-inspired computing enthusiasts

---

## Slide 1: Title Slide

### English

Welcome everyone. Today I'm presenting our systematic investigation into why the Forward-Forward algorithm—Hinton's biologically plausible alternative to backpropagation—hasn't become the new paradigm despite its elegant design.

We'll explore the transfer learning paradox, test bio-inspired variants, and reveal the solution that actually works.

### 中文

欢迎大家。今天我将介绍我们对 Forward-Forward 算法的系统性研究——这是 Hinton 提出的生物学上合理的反向传播替代方案——为什么尽管设计优雅，它却没有成为新的范式。

我们将探索迁移学习悖论、测试生物启发的变体，并揭示真正有效的解决方案。

---

## Slide 2: Our Key Finding (Hero Figure)

### English

*[Point to the figure]*

Let me start with our key finding. This figure shows the transfer learning results from MNIST to Fashion-MNIST.

Notice that CwC-FF—Channel-wise Competitive FF—achieves 89% transfer accuracy. This is the only biologically plausible method that outperforms random initialization at 72%.

Standard FF? Only 54%—that's WORSE than random. We'll explore why this happens and how to fix it.

### 中文

*[指向图片]*

让我从我们的关键发现开始。这张图显示了从 MNIST 到 Fashion-MNIST 的迁移学习结果。

注意 CwC-FF——通道竞争 FF——达到了 89% 的迁移准确率。这是唯一一个超过随机初始化 72% 的生物学合理方法。

标准 FF 呢？只有 54%——这比随机还差。我们将探索为什么会这样以及如何修复它。

---

## Slide 3: Outline

### English

Here's our roadmap. We'll start with the problem—why does FF matter and what barriers does it face? Then we'll explain how FF works, show our experimental results, explore bio-inspired variants that mostly failed, present the CwC-FF solution, and conclude with insights for the field.

### 中文

这是我们的路线图。我们将从问题开始——为什么 FF 重要以及它面临什么障碍？然后我们将解释 FF 如何工作，展示我们的实验结果，探索大多失败的生物启发变体，介绍 CwC-FF 解决方案，并以对该领域的洞察作为结论。

---

## Slide 4: The Three Barriers (Figure)

### English

*[Point to the figure]*

Despite Hinton's prestige and FF's elegant design, three barriers have prevented its adoption:

1. **Performance Gap**: FF achieves 94.5% vs BP's 99.2% on MNIST—a 4.7% gap that matters at scale
2. **Efficiency**: FF needs 30-240× more compute than backprop
3. **Transfer Failure**: The killer—FF features transfer WORSE than random

Today we'll focus on understanding and solving the transfer problem.

### 中文

*[指向图片]*

尽管 Hinton 享有盛誉且 FF 设计优雅，三个障碍阻止了它的采用：

1. **性能差距**：FF 在 MNIST 上达到 94.5%，而 BP 达到 99.2%——在规模化时这 4.7% 的差距很重要
2. **效率**：FF 需要比反向传播多 30-240 倍的计算
3. **迁移失败**：致命问题——FF 特征迁移比随机还差

今天我们将专注于理解和解决迁移问题。

---

## Slide 5: The Backpropagation Dilemma

### English

Why do we even care about alternatives to backpropagation?

Backprop works incredibly well—we all know that. But it has fundamental problems from a biological standpoint.

*[Go through the three points]*

1. **Weight Transport**: BP needs symmetric forward/backward weights. Neurons don't have this.
2. **Global Errors**: BP propagates errors through the entire network. Neurons only have local information.
3. **Two-Phase**: BP requires forward pass, storage, then backward pass. Real neurons learn continuously.

Hinton's solution? Replace the forward and backward passes with two forward passes. No backward pass needed at all.

### 中文

为什么我们要关心反向传播的替代方案？

反向传播效果非常好——我们都知道。但从生物学角度来看，它有根本性的问题。

*[逐一讲解三个要点]*

1. **权重传输**：BP 需要对称的前向/后向权重。神经元没有这个。
2. **全局误差**：BP 通过整个网络传播误差。神经元只有局部信息。
3. **两阶段**：BP 需要前向传递、存储，然后后向传递。真正的神经元是连续学习的。

Hinton 的解决方案？用两次前向传递替代前向和后向传递。完全不需要后向传递。

---

## Slide 6: The Forward-Forward Algorithm

### English

Let me explain how FF works. It's elegantly simple.

*[Point to left column]*

Instead of forward+backward, you do two forward passes:
- **Positive pass**: Real data with correct label → increase "goodness"
- **Negative pass**: Real data with wrong label → decrease "goodness"

The goodness function is just the mean of squared activations.

*[Point to right column]*

Here's the critical design choice: **label embedding**. FF embeds the class label directly into the first 10 pixels of the input.

For digit "3", pixel index 3 is set to the maximum value. This seems elegant, but—and this is crucial—it's the ROOT CAUSE of transfer failure.

### 中文

让我解释 FF 是如何工作的。它优雅而简单。

*[指向左列]*

与前向+后向不同，你做两次前向传递：
- **正向传递**：真实数据加正确标签 → 增加"goodness"
- **负向传递**：真实数据加错误标签 → 减少"goodness"

goodness 函数就是激活值平方的均值。

*[指向右列]*

这是关键的设计选择：**标签嵌入**。FF 将类别标签直接嵌入到输入的前 10 个像素中。

对于数字"3"，像素索引 3 被设置为最大值。这看起来很优雅，但——这是关键——这是迁移失败的根本原因。

---

## Slide 7: Research Questions

### English

We set out to answer four research questions:

1. **RQ1**: Which negative sampling strategy works best?
2. **RQ2**: Can FF features transfer across tasks?
3. **RQ3**: Why does standard FF transfer poorly?
4. **RQ4**: Can bio-inspired variants improve FF?

*[Point to preview block]*

Spoiler alert: Standard FF achieves 94.5% on MNIST but transfers WORSE than random initialization!

### 中文

我们设定了四个研究问题：

1. **RQ1**：哪种负样本策略效果最好？
2. **RQ2**：FF 特征能否跨任务迁移？
3. **RQ3**：为什么标准 FF 迁移效果差？
4. **RQ4**：生物启发变体能否改进 FF？

*[指向预览块]*

剧透：标准 FF 在 MNIST 上达到 94.5%，但迁移比随机初始化还差！

---

## Slide 8: RQ1 - Negative Sampling (Figure)

### English

*[Point to the figure]*

We tested 6 different negative sampling strategies, each trained for 1000 epochs per layer—a fair comparison.

The results are clear: Hinton's original "wrong label" strategy wins at 94.5%. The more sophisticated strategies—hybrid mixing, noise augmentation, masking—all perform worse.

Simple is better for training. But remember, this will come back to bite us in transfer learning.

### 中文

*[指向图片]*

我们测试了 6 种不同的负样本策略，每种在每层训练 1000 个 epoch——公平比较。

结果很清楚：Hinton 的原始"错误标签"策略以 94.5% 获胜。更复杂的策略——混合混合、噪声增强、掩码——都表现更差。

训练时简单更好。但记住，这在迁移学习中会反噬我们。

---

## Slide 9: RQ1 - The 6 Strategies

### English

*[Walk through each strategy]*

Here are the 6 strategies we tested:

1. **wrong_label**: Just swap the label—Hinton's original
2. **class_confusion**: Different image, same label
3. **same_class_diff_img**: Different image, wrong label
4. **hybrid_mix**: Blend two images with wrong label
5. **noise_augmented**: Add Gaussian noise
6. **masked**: Random pixel masking

The simple approach wins. Complex negative generation actually hurts learning.

### 中文

*[逐一讲解每种策略]*

这是我们测试的 6 种策略：

1. **wrong_label**：只交换标签——Hinton 的原始方法
2. **class_confusion**：不同图像，相同标签
3. **same_class_diff_img**：不同图像，错误标签
4. **hybrid_mix**：混合两张图像加错误标签
5. **noise_augmented**：添加高斯噪声
6. **masked**：随机像素掩码

简单方法获胜。复杂的负样本生成实际上会损害学习。

---

## Slide 10: RQ2 - The Transfer Paradox (Figure)

### English

*[Slow down here—this is the key finding]*

Now here's where things get really interesting—and troubling.

*[Point to the figure]*

This is the transfer learning paradox visualized. We trained on MNIST, froze features, and tested on Fashion-MNIST.

Standard FF: 54% transfer accuracy.
Random initialization: 72% transfer accuracy.

FF is WORSE than starting from scratch! The pretrained features actively hurt performance.

### 中文

*[这里放慢——这是关键发现]*

现在事情变得真正有趣——也令人不安。

*[指向图片]*

这是可视化的迁移学习悖论。我们在 MNIST 上训练，冻结特征，在 Fashion-MNIST 上测试。

标准 FF：54% 迁移准确率。
随机初始化：72% 迁移准确率。

FF 比从头开始还差！预训练的特征主动损害了性能。

---

## Slide 11: RQ2 - Transfer Results Table

### English

*[Walk through the table]*

Let me show you the full comparison:

- **CwC-FF**: 98.71% source, 89.05% transfer, +17.2% vs random
- **Backprop**: 95.08% source, 75.49% transfer, +3.6% vs random
- **Random**: 71.89% baseline
- **Standard FF**: 89.90% source, 54.19% transfer, -17.7% vs random

*[Emphasize the paradox]*

Standard FF features are WORSE than random. This is not "a bit worse"—this is fundamentally broken for transfer.

### 中文

*[逐一讲解表格]*

让我展示完整的比较：

- **CwC-FF**：源 98.71%，迁移 89.05%，比随机高 17.2%
- **Backprop**：源 95.08%，迁移 75.49%，比随机高 3.6%
- **Random**：71.89% 基线
- **Standard FF**：源 89.90%，迁移 54.19%，比随机低 17.7%

*[强调悖论]*

标准 FF 特征比随机还差。这不是"差一点"——对于迁移来说这是根本性的问题。

---

## Slide 12: RQ2 - t-SNE Visualization

### English

*[Point to the figure]*

This t-SNE visualization makes the problem clear.

On the left: FF features on Fashion-MNIST. See how scattered the clusters are? The features don't separate the classes well.

On the right: BP features. Much more organized clusters with clear separation.

FF features learned for MNIST don't transfer—they're task-specific, not general visual features.

### 中文

*[指向图片]*

这个 t-SNE 可视化清楚地展示了问题。

左边：Fashion-MNIST 上的 FF 特征。看到聚类有多分散吗？特征没有很好地分离类别。

右边：BP 特征。聚类更有组织，有清晰的分离。

为 MNIST 学习的 FF 特征无法迁移——它们是任务特定的，不是通用的视觉特征。

---

## Slide 13: RQ3 - Label Embedding Root Cause (Figure)

### English

*[Point to the figure—this is the key explanation]*

This figure explains WHY FF fails at transfer. It's all about label embedding.

*[Walk through the 4 panels]*

1. **What is Label Embedding**: The first 10 pixels contain the label, not image data
2. **Why It Breaks Transfer**: MNIST label 0 = digit zero, but Fashion-MNIST label 0 = T-shirt!
3. **What Features Learn**: Standard FF learns "label detectors", CwC-FF learns actual visual features
4. **Results**: Standard FF 54%, CwC-FF 89%

The label embedding creates a shortcut that destroys transfer.

### 中文

*[指向图片——这是关键解释]*

这张图解释了为什么 FF 在迁移时失败。一切都与标签嵌入有关。

*[逐一讲解 4 个面板]*

1. **什么是标签嵌入**：前 10 个像素包含标签，不是图像数据
2. **为什么破坏迁移**：MNIST 标签 0 = 数字零，但 Fashion-MNIST 标签 0 = T 恤！
3. **特征学到什么**：标准 FF 学习"标签检测器"，CwC-FF 学习实际的视觉特征
4. **结果**：标准 FF 54%，CwC-FF 89%

标签嵌入创造了一个破坏迁移的捷径。

---

## Slide 14: RQ3 - Why Label Embedding Breaks Transfer

### English

*[Use a concrete example]*

Let me make this concrete.

During MNIST training, pixel[3] being bright means "digit 3". The network learns: "pixel[3] bright + curves = positive sample."

Now we transfer to Fashion-MNIST. Pixel[3] bright now means "Dress", not digit 3! But the network still expects curves when it sees pixel[3] lit up.

*[Point to the two blocks]*

**Standard FF**: Features = f(image, LABEL) → Useless when labels change meaning!

**CwC-FF**: Features = f(image) → Transfer beautifully!

### 中文

*[使用具体例子]*

让我具体说明。

在 MNIST 训练期间，pixel[3] 亮意味着"数字 3"。网络学习到："pixel[3] 亮 + 曲线 = 正样本。"

现在我们迁移到 Fashion-MNIST。Pixel[3] 亮现在意味着"连衣裙"，不是数字 3！但当网络看到 pixel[3] 亮起时，它仍然期望曲线。

*[指向两个块]*

**标准 FF**：特征 = f(图像, 标签) → 当标签含义改变时无用！

**CwC-FF**：特征 = f(图像) → 完美迁移！

---

## Slide 15: RQ4 - Bio-Inspired Overview (Figure)

### English

*[Point to the figure]*

At this point we thought: maybe FF needs to be MORE biologically plausible to transfer well.

So we implemented 5 bio-inspired variants based on cutting-edge neuroscience research.

*[Point to results in figure]*

The results? Most failed. Three-factor learning gave +1.5% marginal improvement. Prospective FF and PCL-FF failed catastrophically.

Only CwC-FF actually works—and ironically, it's not trying to be more biological. It's fixing the architectural problem.

### 中文

*[指向图片]*

这时我们想：也许 FF 需要更具生物学合理性才能更好地迁移。

所以我们基于前沿神经科学研究实现了 5 种生物启发变体。

*[指向图中的结果]*

结果如何？大多数失败了。三因子学习给出了 +1.5% 的轻微改善。前瞻性 FF 和 PCL-FF 灾难性地失败了。

只有 CwC-FF 真正有效——讽刺的是，它并没有试图变得更具生物学性。它在修复架构问题。

---

## Slide 16: Three-Factor Hebbian Learning

### English

Three-factor learning comes from the neuromodulator literature. Real synapses are modulated by dopamine, acetylcholine, and norepinephrine.

*[Point to equation]*

The learning rule: ΔW = f(pre) × f(post) × M(t), where M is a modulator signal.

*[Point to results table]*

We tested three modulation schemes:
- Top-down modulation: +1.5% transfer (marginal)
- None (baseline): 62.8%
- Layer agreement: -3.0% (hurt!)
- Reward prediction: 18.4% (failed!)

**Verdict**: Modulation doesn't fix label coupling. You're modulating a signal that's teaching the wrong thing.

### 中文

三因子学习来自神经调质文献。真正的突触受到多巴胺、乙酰胆碱和去甲肾上腺素的调节。

*[指向公式]*

学习规则：ΔW = f(pre) × f(post) × M(t)，其中 M 是调节信号。

*[指向结果表]*

我们测试了三种调节方案：
- 自上而下调节：迁移 +1.5%（轻微）
- 无（基线）：62.8%
- 层协议：-3.0%（有害！）
- 奖励预测：18.4%（失败！）

**结论**：调节不能修复标签耦合。你在调节一个教错误东西的信号。

---

## Slide 17: Prospective FF

### English

Prospective FF comes from Song et al.'s Nature Neuroscience 2024 paper on anticipatory neural activity.

*[Explain the mechanism]*

The idea: two-phase learning. First infer what the target activity should be, then consolidate weights to produce that activity.

*[Point to results]*

But look at what happens with more iterations:
- 1 iteration: +5.3% transfer gain
- 10 iterations: +1.2%
- 100 iterations: -13.2%!

More iterations = STRONGER label coupling = WORSE transfer!

The mechanism amplifies label-specific features instead of fixing the problem.

### 中文

前瞻性 FF 来自 Song 等人 2024 年 Nature Neuroscience 论文中关于预期神经活动的研究。

*[解释机制]*

想法：两阶段学习。首先推断目标活动应该是什么，然后整合权重来产生该活动。

*[指向结果]*

但看看更多迭代时会发生什么：
- 1 次迭代：迁移增益 +5.3%
- 10 次迭代：+1.2%
- 100 次迭代：-13.2%！

更多迭代 = 更强的标签耦合 = 更差的迁移！

该机制放大了标签特定的特征，而不是修复问题。

---

## Slide 18: PCL-FF (Death Cascade)

### English

PCL-FF is inspired by predictive coding in cortical circuits.

*[Point to the mechanism]*

The idea: only prediction errors propagate forward, creating sparsity. But the sparsity penalty created a "death cascade."

*[Explain the cascade]*

The sparsity penalty incentivizes h = 0. Combined with ReLU, this creates a positive feedback loop:
- More zeros → lower loss
- Network learns "dead neurons = good"

*[Point to results]*

Standard FF: 90% accuracy, 8% dead neurons
PCL-FF: 17.5% accuracy, 100% dead neurons

Complete failure. The death cascade killed all neurons by epoch 500.

### 中文

PCL-FF 受到皮层回路中预测编码的启发。

*[指向机制]*

想法：只有预测误差向前传播，创造稀疏性。但稀疏性惩罚创造了"死亡级联"。

*[解释级联]*

稀疏性惩罚激励 h = 0。结合 ReLU，这创造了一个正反馈循环：
- 更多零 → 更低损失
- 网络学习"死神经元 = 好"

*[指向结果]*

标准 FF：90% 准确率，8% 死神经元
PCL-FF：17.5% 准确率，100% 死神经元

完全失败。到第 500 个 epoch，死亡级联杀死了所有神经元。

---

## Slide 19: Lessons from Failures (Figure)

### English

*[Point to the figure]*

This figure summarizes what we learned from our failures.

**Key insight**: Bio-inspired modifications address SYMPTOMS, not the ROOT CAUSE.

The fundamental problem is label embedding. Adding dopamine-like modulation, predictive coding, or sparsity doesn't fix the core architectural issue.

The only solution that works—CwC-FF—removes labels from the input entirely.

### 中文

*[指向图片]*

这张图总结了我们从失败中学到的东西。

**关键洞察**：生物启发的修改解决的是症状，而不是根本原因。

根本问题是标签嵌入。添加类似多巴胺的调节、预测编码或稀疏性不能修复核心架构问题。

唯一有效的解决方案——CwC-FF——完全从输入中移除标签。

---

## Slide 20: CwC-FF - The Solution

### English

*[Build anticipation]*

So after all those failures, let me introduce the solution: Channel-wise Competitive FF.

*[Point to comparison]*

**Standard FF**: Input = [label, image], features coupled to labels

**CwC-FF**: Input = [image] only—NO LABELS!

Instead of label embedding, CwC-FF uses channel competition:
- Channels compete within layers
- Winners get positive signal
- Losers get negative signal
- No labels needed at all!

This is the key architectural change that makes transfer work.

### 中文

*[积累期待]*

所以在所有这些失败之后，让我介绍解决方案：通道竞争 FF。

*[指向比较]*

**标准 FF**：输入 = [标签, 图像]，特征与标签耦合

**CwC-FF**：输入 = [仅图像]——没有标签！

CwC-FF 不使用标签嵌入，而是使用通道竞争：
- 通道在层内竞争
- 赢家获得正信号
- 输家获得负信号
- 完全不需要标签！

这是使迁移有效的关键架构变化。

---

## Slide 21: CwC-FF Results (Figure + Table)

### English

*[Point to radar chart]*

Look at this comparison. CwC-FF dominates across all dimensions.

*[Walk through the table]*

- CwC-FF: 98.71% source, 89.05% transfer, +17.2% vs random, high bio-plausibility
- Backprop: 99.2% source, 75.49% transfer, none bio-plausible
- Standard FF: 94.50% source, 54.19% transfer

CwC-FF gives us the BEST of both worlds: better than backprop at transfer while remaining biologically plausible!

### 中文

*[指向雷达图]*

看这个比较。CwC-FF 在所有维度上都占优势。

*[逐一讲解表格]*

- CwC-FF：源 98.71%，迁移 89.05%，比随机高 17.2%，生物合理性高
- Backprop：源 99.2%，迁移 75.49%，无生物合理性
- 标准 FF：源 94.50%，迁移 54.19%

CwC-FF 给我们两全其美：在迁移方面比反向传播更好，同时保持生物学合理性！

---

## Slide 22: Summary of Findings (Figure)

### English

*[Point to summary figure]*

This figure captures our complete findings:

1. Standard FF achieves high source accuracy but fails at transfer
2. Bio-inspired modifications don't solve the core problem
3. CwC-FF's label-free approach is the solution

The transfer paradox has been explained and solved.

### 中文

*[指向总结图]*

这张图概括了我们的完整发现：

1. 标准 FF 达到高源准确率但迁移失败
2. 生物启发的修改不能解决核心问题
3. CwC-FF 的无标签方法是解决方案

迁移悖论已被解释和解决。

---

## Slide 23: Key Insights

### English

*[Count on fingers]*

Four key insights:

1. **Simple negative sampling wins for training but loses for transfer.** The same label embedding that gives 94.5% accuracy creates the shortcut that destroys transfer.

2. **Bio-inspired modifications don't help.** Three-factor learning, predictive coding, sparsity—all real brain features, none fix the problem.

3. **Label-free learning is key.** CwC-FF removes labels entirely, forcing the network to learn actual visual features.

4. **There IS a solution.** CwC-FF: 98.7% source, 89% transfer—best of both worlds!

### 中文

*[用手指数]*

四个关键洞察：

1. **简单的负样本策略赢得训练但输掉迁移。** 带来 94.5% 准确率的相同标签嵌入创造了破坏迁移的捷径。

2. **生物启发的修改没有帮助。** 三因子学习、预测编码、稀疏性——都是真实的大脑特征，没有一个能修复问题。

3. **无标签学习是关键。** CwC-FF 完全移除标签，迫使网络学习实际的视觉特征。

4. **有解决方案。** CwC-FF：源 98.7%，迁移 89%——两全其美！

---

## Slide 24: Conclusion

### English

So why hasn't FF become the new paradigm?

*[Point to limitations]*

**Limitations**: 4.7% accuracy gap, catastrophic transfer failure, label embedding creates shortcuts, bio-inspired fixes don't help.

*[Point to path forward]*

**The Path Forward**: CwC-FF solves the transfer issue. Remove labels from input. Channel competition works. There's potential for neuromorphic hardware.

*[Read the take-home message]*

**Take-home message**: Biological plausibility alone doesn't guarantee good ML properties. Understanding failure modes leads to principled solutions.

### 中文

那么为什么 FF 没有成为新的范式？

*[指向限制]*

**限制**：4.7% 的准确率差距，灾难性的迁移失败，标签嵌入创造捷径，生物启发的修复没有帮助。

*[指向前进道路]*

**前进道路**：CwC-FF 解决了迁移问题。从输入中移除标签。通道竞争有效。有神经形态硬件的潜力。

*[读取核心信息]*

**核心信息**：仅生物学合理性不能保证良好的 ML 属性。理解失败模式导致原则性的解决方案。

---

## Slide 25: Future Work

### English

Several directions for future work:

1. **Scale CwC-FF** to CIFAR-10 and ImageNet
2. **Hybrid approaches**: FF for early layers + BP for final layers
3. **Neuromorphic implementation** on Intel Loihi or IBM TrueNorth
4. **Unsupervised extensions**: Remove supervision entirely

### 中文

未来工作的几个方向：

1. **扩展 CwC-FF** 到 CIFAR-10 和 ImageNet
2. **混合方法**：早期层用 FF + 最终层用 BP
3. **神经形态实现** 在 Intel Loihi 或 IBM TrueNorth 上
4. **无监督扩展**：完全移除监督

---

## Slide 26: References

### English

I want to acknowledge the key papers that informed this work. Hinton's original 2022 paper, Brenig et al. for first showing FF has transfer problems, Lorberbom for Layer Collaboration, and most importantly Papachristodoulou et al. for CwC-FF—the solution that actually works.

### 中文

我想感谢为这项工作提供信息的关键论文。Hinton 2022 年的原始论文，Brenig 等人首先展示 FF 有迁移问题，Lorberbom 的层协作，最重要的是 Papachristodoulou 等人的 CwC-FF——真正有效的解决方案。

---

## Slide 27: Thank You

### English

Thank you for your attention. Questions?

The code and all experiments are available at github.com/koriyoshi2041/ff-negative-samples.

This was tested on Apple M4 Air with PyTorch 2.0+.

### 中文

感谢您的关注。有问题吗？

代码和所有实验可在 github.com/koriyoshi2041/ff-negative-samples 获取。

这是在 Apple M4 Air 上使用 PyTorch 2.0+ 测试的。

---

# Q&A Preparation

## Anticipated Questions

### Q: Why didn't you test on larger datasets like ImageNet?

**EN:** We focused on MNIST/Fashion-MNIST to thoroughly understand the mechanism before scaling. ASGE from ICASSP 2026 has started pushing to ImageNet scale—combining CwC-FF with their techniques is a natural next step.

**中文:** 我们专注于 MNIST/Fashion-MNIST 以在扩展之前彻底理解机制。ICASSP 2026 的 ASGE 已经开始推向 ImageNet 规模——将 CwC-FF 与他们的技术结合是自然的下一步。

### Q: Isn't CwC-FF just contrastive learning?

**EN:** There's a connection. Both use competition for positive/negative signals. But CwC-FF is more localized—competition happens within channels of the same layer, not between different augmentations. This makes it compatible with FF's layer-local learning.

**中文:** 有联系。两者都使用竞争来产生正负信号。但 CwC-FF 更局部化——竞争发生在同一层的通道之间，而不是不同增强之间。这使它与 FF 的层局部学习兼容。

### Q: Is FF actually used in any real applications?

**EN:** Not widely yet. The main interest is in neuromorphic computing—chips like Intel Loihi and IBM TrueNorth that could benefit from BP-free training. There's also interest from edge computing where memory is precious.

**中文:** 还没有广泛使用。主要兴趣在神经形态计算——像 Intel Loihi 和 IBM TrueNorth 这样的芯片可以从无 BP 训练中受益。边缘计算也有兴趣，因为那里内存很珍贵。

---

# Timing Guide

| Slide | Topic | Target Time |
|-------|-------|-------------|
| 1-3 | Introduction | 2 min |
| 4-5 | The Problem | 2 min |
| 6-7 | FF Algorithm | 2 min |
| 8-9 | RQ1: Negative Sampling | 2 min |
| 10-12 | RQ2: Transfer Paradox | 3 min |
| 13-14 | RQ3: Root Cause | 3 min |
| 15-19 | RQ4: Bio-Inspired | 5 min |
| 20-21 | CwC-FF Solution | 2 min |
| 22-25 | Discussion | 3 min |
| 26-27 | Wrap-up | 1 min |
| **Total** | | **~25 min** |

**Buffer for Q&A:** 5-10 minutes

---

# Presentation Tips

1. **The transfer paradox (Slides 10-11) is your "wow" moment**—slow down, let the numbers sink in

2. **Use the figures actively**—point to specific parts as you explain

3. **The bio-inspired failures build tension**—they set up CwC-FF as the revelation

4. **Practice the Chinese/English transitions** if presenting to bilingual audience

5. **Have backup answers** for "why not ImageNet" and "isn't this just contrastive learning"
