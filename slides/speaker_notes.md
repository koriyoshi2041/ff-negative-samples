# Forward-Forward Algorithm: Speaker Notes
# Transfer Learning Paradox and Bio-Inspired Solutions

**Presentation Duration:** ~20 minutes
**Target Audience:** ML researchers, neuroscience-inspired computing enthusiasts

---

## Slide 1: Title Slide

### English

Welcome everyone. Today I'm going to share something that genuinely surprised us during our research.

We've been investigating the Forward-Forward algorithm - Hinton's biologically plausible alternative to backpropagation. And we discovered a paradox that I think gets at something fundamental about how neural networks learn.

The question we started with was simple: Can a brain-inspired learning algorithm actually learn transferable features? The answer... well, it's complicated, and that's what makes this interesting.

*[Pause for 2-3 seconds to let the tension build]*

Let me take you through what we found.

### 中文

欢迎大家。今天我要分享的是我们研究中发现的一个真正令人惊讶的现象。

我们一直在研究 Forward-Forward 算法——Hinton 提出的一种生物学上合理的反向传播替代方案。我们发现了一个悖论，我认为它触及了神经网络学习的一些根本性问题。

我们最初的问题很简单：一个受大脑启发的学习算法能否真正学到可迁移的特征？答案......嗯，很复杂，这正是它有趣的地方。

*[停顿2-3秒，让悬念积累]*

让我带大家看看我们的发现。

---

## Slide 2: Motivation - Why Biological Plausibility?

### English

So why do we care about biologically plausible learning?

Backpropagation works incredibly well - we all know that. But it has some serious issues from a biological standpoint.

First, the weight transport problem. Backprop needs the exact same weights going backward as going forward. Neurons in your brain don't do that - they don't have some magical symmetric connection going back.

Second, the timing issue. Backprop waits for a complete forward pass, stores everything, then runs the gradient backward. Real neurons don't wait - they're firing and learning continuously.

Third - and this is big for industry - backprop needs to store all intermediate activations. That means massive memory costs. Your brain operates on about 20 watts. Training GPT-4 took... considerably more.

*[Small pause for audience reaction]*

Hinton proposed Forward-Forward in late 2022 as an alternative. The idea: what if we could learn with just forward passes?

### 中文

那么我们为什么要关注生物学上合理的学习呢？

反向传播效果非常好——这我们都知道。但从生物学角度看，它有一些严重的问题。

首先是权重传输问题。反向传播需要反向使用完全相同的权重。你大脑中的神经元不会这样做——它们没有某种神奇的对称连接返回去。

第二是时序问题。反向传播要等待完整的前向传递，存储所有内容，然后反向运行梯度。真正的神经元不会等待——它们持续地激发和学习。

第三——这对工业界来说很重要——反向传播需要存储所有中间激活值。这意味着巨大的内存成本。你的大脑大约用20瓦运行。训练GPT-4花费的......要多得多。

*[小停顿让观众反应]*

Hinton 在2022年底提出了 Forward-Forward 作为替代方案。核心想法：如果我们只用前向传递就能学习呢？

---

## Slide 3: The Forward-Forward Algorithm

### English

Let me explain how Forward-Forward works. It's elegantly simple.

Instead of one forward pass and one backward pass, you do two forward passes. That's it.

The first pass uses real data - we call this the positive pass. We want the network to produce high activation when it sees real data.

The second pass uses fake data - the negative pass. We want the network to produce low activation for fake data.

The learning signal comes from something called "goodness" - essentially the sum of squared activations in each layer. Each layer learns locally: increase goodness for positive samples, decrease it for negative samples.

There's a threshold - typically around 2.0 - that separates positive from negative.

*[Point to the visualization if there is one]*

The beautiful thing: each layer learns independently. No need to propagate errors backward through the whole network. This is what makes it biologically plausible.

### 中文

让我解释一下 Forward-Forward 是如何工作的。它优雅而简单。

与其做一次前向传递和一次后向传递，你只做两次前向传递。就是这样。

第一次传递使用真实数据——我们称之为正向传递。我们希望网络在看到真实数据时产生高激活。

第二次传递使用假数据——负向传递。我们希望网络对假数据产生低激活。

学习信号来自于一个叫做"goodness"的东西——本质上是每层激活值的平方和。每层独立学习：对正样本增加 goodness，对负样本减少 goodness。

有一个阈值——通常在2.0左右——来分隔正样本和负样本。

*[如果有可视化就指向它]*

美妙之处在于：每层独立学习。不需要通过整个网络反向传播误差。这正是它在生物学上合理的原因。

---

## Slide 4: Our Research Questions

### English

So here's what we set out to investigate.

We had two main research questions. First: which negative sampling strategy works best? There are many ways to create fake data - Hinton originally just embedded a wrong label, but you could also mix images, add noise, use contrastive methods...

Second, and this turned out to be the bigger surprise: how well do FF features transfer? In the backprop world, we know features learned on ImageNet transfer beautifully to other tasks. Does FF do the same?

*[Let this sink in]*

We expected FF might be a bit worse than backprop at transfer. We did NOT expect what we actually found.

### 中文

这就是我们开始研究的问题。

我们有两个主要研究问题。第一：哪种负样本策略效果最好？有很多方法可以创建假数据——Hinton 最初只是嵌入一个错误的标签，但你也可以混合图像、添加噪声、使用对比方法……

第二，这结果是更大的惊喜：FF 特征的迁移效果如何？在反向传播的世界里，我们知道在 ImageNet 上学到的特征可以漂亮地迁移到其他任务。FF 也是这样吗？

*[让这个问题沉淀一下]*

我们预期 FF 在迁移学习上可能比反向传播差一点。我们没有预料到我们实际发现的结果。

---

## Slide 5: Negative Sampling Results - Simple is Better

### English

Let's start with the negative sampling results.

We tested 10 different strategies - everything from Hinton's original label embedding, to sophisticated contrastive methods, to adversarial approaches.

*[Point to the results table]*

And the winner? Hinton's original method. Just take the correct image, embed the wrong label. That's it.

The sophisticated methods - mixing images, adding noise, adversarial samples - they either matched performance or were worse.

This tells us something important: for supervised FF learning, the negative sample doesn't need to be sophisticated. It just needs to be "wrong" in a consistent way that the network can learn to recognize.

*[Brief pause]*

But here's the thing - this result has a dark side, which we'll get to in a moment.

### 中文

让我们从负样本结果开始。

我们测试了10种不同的策略——从 Hinton 的原始标签嵌入，到复杂的对比方法，再到对抗性方法。

*[指向结果表格]*

获胜者是？Hinton 的原始方法。只需拿正确的图像，嵌入错误的标签。就是这样。

复杂的方法——混合图像、添加噪声、对抗性样本——它们要么性能相当，要么更差。

这告诉我们一些重要的事情：对于监督式 FF 学习，负样本不需要很复杂。它只需要以一种网络可以学会识别的一致方式是"错误的"即可。

*[短暂停顿]*

但问题是——这个结果有一个阴暗面，我们马上就会讲到。

---

## Slide 6: The Transfer Paradox - THIS IS THE KEY FINDING

### English

*[Slow down here - this is the "wow" moment]*

Now here's where things get really interesting. And honestly, really troubling.

We ran standard transfer learning experiments. Train on one dataset, freeze the features, test on another. Classic protocol.

Backprop baseline: features transfer well, as expected. Around 85% accuracy on the target task.

Forward-Forward: 42%.

*[Pause for effect]*

But here's the thing that really shocked us. We also tested random, untrained networks. Networks that never saw any data.

Random network: 45%.

*[Let this sink in for 3-4 seconds]*

Let me say that again. Forward-Forward features transfer WORSE than random initialization. The network actively learned representations that are harmful for downstream tasks.

This isn't "FF is a bit worse." This is "FF is fundamentally broken for transfer learning."

### 中文

*[这里放慢语速——这是"哇"的时刻]*

现在事情变得真正有趣了。说实话，也真的令人不安。

我们进行了标准的迁移学习实验。在一个数据集上训练，冻结特征，在另一个数据集上测试。经典的实验方案。

反向传播基线：特征迁移效果良好，正如预期。在目标任务上约85%的准确率。

Forward-Forward：42%。

*[停顿以产生效果]*

但真正让我们震惊的是这个。我们还测试了随机的、未经训练的网络。从未见过任何数据的网络。

随机网络：45%。

*[让这个信息沉淀3-4秒]*

让我再说一遍。Forward-Forward 特征的迁移效果比随机初始化还要差。网络主动学习了对下游任务有害的表征。

这不是"FF 差一点"。这是"FF 在迁移学习上根本就是坏的"。

---

## Slide 7: Root Cause - The Label Embedding Problem

### English

So we spent a lot of time figuring out why. And the answer lies in how labels are embedded.

*[Use a concrete example here]*

Think about this: In standard FF, you embed the label in the first 10 pixels of the image. A "0" might make pixel 0 bright. A "1" makes pixel 1 bright.

Now imagine we're training on MNIST - handwritten digits. The network learns: "When pixel 0 is bright AND the image looks like a zero, that's positive."

But then we transfer to Fashion-MNIST. Now pixel 0 being bright means "T-shirt." But the network has already associated "pixel 0 bright" with "round shape like the digit zero."

*[Let this sink in]*

The label embedding creates a shortcut. The network doesn't need to learn robust visual features. It just needs to learn: "Do the label pixels match the pattern I expect?" That's enough to get high goodness for positives and low for negatives.

This is a classic shortcut learning problem, but it's baked into the core of how FF works.

### 中文

所以我们花了很多时间弄清楚原因。答案在于标签是如何嵌入的。

*[这里用一个具体的例子]*

想想这个：在标准 FF 中，你把标签嵌入到图像的前10个像素中。"0"可能让像素0变亮。"1"让像素1变亮。

现在想象我们在 MNIST 上训练——手写数字。网络学习到："当像素0亮起并且图像看起来像零，那就是正样本。"

但然后我们迁移到 Fashion-MNIST。现在像素0亮起意味着"T恤"。但网络已经将"像素0亮起"与"像数字零那样的圆形"关联起来了。

*[让这个信息沉淀]*

标签嵌入创造了一个捷径。网络不需要学习鲁棒的视觉特征。它只需要学习："标签像素是否与我期望的模式匹配？"这就足以让正样本获得高 goodness，负样本获得低 goodness。

这是一个经典的捷径学习问题，但它被嵌入到 FF 工作的核心中。

---

## Slide 8: Bio-Inspired Attempts - We Tried Neuroscience

### English

At this point, we thought: maybe the problem is that FF isn't biologically plausible enough. Maybe we need to look at how real brains avoid these problems.

So we went to the neuroscience literature. We implemented three biologically-inspired modifications.

First: Three-Factor Learning. This comes from the neuromodulator literature. Real synapses are modulated by signals like dopamine and norepinephrine. We added a third factor to modulate the local learning signal.

Second: Prospective Forward-Forward. Based on predictive coding theory. The idea is to learn representations that predict future states, not just current classification.

Third: Predictive Coding Light FF. Inspired by the predictive processing framework. The brain constantly predicts its inputs, and learning comes from prediction errors.

*[Pause]*

All of these are grounded in real neuroscience. All of them have been successful in other contexts.

### 中文

到这一点，我们想：也许问题在于 FF 还不够生物学合理。也许我们需要看看真正的大脑是如何避免这些问题的。

所以我们查阅了神经科学文献。我们实现了三种受生物学启发的修改。

第一：三因子学习。这来自神经调质文献。真正的突触受到多巴胺和去甲肾上腺素等信号的调节。我们添加了第三个因子来调节局部学习信号。

第二：前瞻性 Forward-Forward。基于预测编码理论。核心想法是学习预测未来状态的表征，而不仅仅是当前分类。

第三：预测编码轻量版 FF。受预测处理框架启发。大脑不断预测其输入，学习来自预测误差。

*[停顿]*

所有这些都基于真实的神经科学。所有这些在其他情境中都是成功的。

---

## Slide 9: Three-Factor Learning Results

### English

Let's go through what happened with Three-Factor Learning.

The idea is that learning should be modulated by a global signal - like dopamine in the brain. When something unexpected happens, dopamine spikes, and that tells all synapses "pay attention, this is important."

We tried three modulation schemes:
- Top-down: higher layers modulate lower layers
- Bottom-up: input quality modulates all layers
- Surprise-based: prediction error modulates learning

*[Point to results]*

Top-down modulation showed marginal improvement - about 47% transfer instead of 42%. Better than baseline, but still worse than random.

Bottom-up and surprise-based? No improvement at all.

The core problem is still there: the label embedding creates a shortcut that modulation can't fix. You're modulating a learning signal that's fundamentally teaching the wrong thing.

### 中文

让我们看看三因子学习的结果。

核心想法是学习应该被全局信号调节——就像大脑中的多巴胺。当意外发生时，多巴胺飙升，这告诉所有突触"注意，这很重要。"

我们尝试了三种调节方案：
- 自上而下：高层调节低层
- 自下而上：输入质量调节所有层
- 惊讶基础：预测误差调节学习

*[指向结果]*

自上而下调节显示了轻微改善——大约47%的迁移准确率，而不是42%。比基线好，但仍然比随机差。

自下而上和惊讶基础？完全没有改善。

核心问题仍然存在：标签嵌入创造了一个调节无法修复的捷径。你在调节一个根本就在教错误事情的学习信号。

---

## Slide 10: Prospective FF Results - More Iterations, Worse Results

### English

Next, Prospective Forward-Forward. This is based on a Nature Neuroscience paper from 2024.

The idea is intriguing: instead of just learning current representations, learn to predict what representations SHOULD be. It's a two-phase approach - one phase predicts, one phase corrects.

We thought this might help because it forces the network to develop more structured internal representations.

*[Point to learning curve]*

What we found was counterintuitive. With few iterations, Prospective FF matches baseline FF. But as we increase the prediction iterations - which should give better predictions - transfer performance gets WORSE.

5 iterations: 40% transfer
10 iterations: 38% transfer
20 iterations: 35% transfer

The prediction mechanism is reinforcing the label-dependent representations. More prediction iterations mean stronger reinforcement of the shortcut.

### 中文

接下来是前瞻性 Forward-Forward。这基于2024年 Nature Neuroscience 的一篇论文。

这个想法很有趣：不是只学习当前表征，而是学习预测表征应该是什么。这是一种两阶段方法——一个阶段预测，一个阶段纠正。

我们认为这可能有帮助，因为它迫使网络发展更结构化的内部表征。

*[指向学习曲线]*

我们发现的结果是反直觉的。迭代次数少时，前瞻性 FF 与基线 FF 相当。但当我们增加预测迭代次数——这应该会给出更好的预测时——迁移性能反而变得更差。

5次迭代：40%迁移
10次迭代：38%迁移
20次迭代：35%迁移

预测机制正在强化标签依赖的表征。更多的预测迭代意味着对捷径更强的强化。

---

## Slide 11: PCL-FF Results - The Death Cascade

### English

Finally, Predictive Coding Light FF. This adds sparse coding to FF, inspired by how real neurons have sparse activation patterns.

The theory is sound: sparse representations are often more interpretable and more transferable. Many brain areas show sparse coding.

*[Point to visualization if available]*

But we ran into what I call the "death cascade."

Sparse coding in FF causes low-activity neurons to receive even less gradient signal. So they become even less active. Then they get even less signal. And eventually, they effectively die - zero activation, zero gradient, no recovery.

We watched as layer after layer experienced this cascade. By the end of training, sometimes 60% of neurons were effectively dead.

*[Pause]*

Dead neurons don't transfer. Transfer accuracy: 38%.

The biological plausibility of sparsity is undermined by the local learning rule. There's no global signal to rescue dying neurons.

### 中文

最后是预测编码轻量版 FF。这为 FF 添加了稀疏编码，受到真实神经元具有稀疏激活模式的启发。

理论是合理的：稀疏表征通常更可解释且更可迁移。许多大脑区域显示稀疏编码。

*[如果有可视化就指向它]*

但我们遇到了我称之为"死亡级联"的问题。

FF 中的稀疏编码导致低活动神经元接收到更少的梯度信号。所以它们变得更不活跃。然后它们得到更少的信号。最终，它们实际上死亡了——零激活，零梯度，无法恢复。

我们看着一层接一层地经历这种级联。到训练结束时，有时60%的神经元实际上已经死亡。

*[停顿]*

死亡的神经元无法迁移。迁移准确率：38%。

稀疏性的生物学合理性被局部学习规则破坏了。没有全局信号来拯救濒死的神经元。

---

## Slide 12: Layer Collaboration - Modest Improvement

### English

We also tried Layer Collaboration, based on Lorberbom's AAAI 2024 paper.

The insight there is that in standard FF, each layer optimizes independently. They don't know about each other. Layer 1 doesn't know Layer 3 exists.

Layer Collaboration adds a "gamma" term - each layer sees the total goodness from other layers, which provides some coordination.

*[Show the comparison]*

Results: 51% transfer accuracy. Better than baseline FF (42%), finally beats random (45%).

But it's still far from BP (85%). Layer collaboration helps layers work together, but it doesn't fix the fundamental label embedding problem.

The layers are now coordinating, but they're coordinating to learn the same label-dependent shortcut. Coordinated failure is still failure.

### 中文

我们还尝试了层协作，基于 Lorberbom 2024年 AAAI 论文。

那里的洞察是：在标准 FF 中，每层独立优化。它们不知道彼此的存在。第一层不知道第三层存在。

层协作添加了一个"gamma"项——每层看到其他层的总 goodness，这提供了一些协调。

*[展示比较]*

结果：51%迁移准确率。比基线 FF（42%）好，终于超过了随机（45%）。

但仍然远离 BP（85%）。层协作帮助各层一起工作，但它没有修复根本的标签嵌入问题。

各层现在在协调了，但它们在协调学习相同的标签依赖捷径。协调的失败仍然是失败。

---

## Slide 13: CwC-FF - The Solution That Works

### English

*[Build up the anticipation]*

So we've tried sophisticated negative samples, three different neuroscience-inspired methods, and layer collaboration. All failures or marginal improvements.

Then we found CwC-FF: Channel-wise Competitive Forward-Forward.

*[Emphasize this point]*

The key insight: remove the labels entirely from the input.

Instead of embedding labels in the image, CwC-FF uses channel competition. Different channels in the same layer compete to represent the input. Positive "goodness" is defined by channel agreement. Negative "goodness" is defined by channel disagreement.

No labels in the input means no label-dependent shortcut. The network HAS to learn actual visual features.

*[Point to architecture diagram if available]*

This is a single forward pass - even more efficient than standard FF. And it's completely unsupervised.

### 中文

*[积累期待]*

所以我们尝试了复杂的负样本、三种不同的神经科学启发方法和层协作。全部失败或只有轻微改善。

然后我们发现了 CwC-FF：通道竞争 Forward-Forward。

*[强调这一点]*

关键洞察：从输入中完全移除标签。

CwC-FF 不是把标签嵌入到图像中，而是使用通道竞争。同一层的不同通道竞争来表示输入。正向"goodness"由通道一致性定义。负向"goodness"由通道不一致性定义。

输入中没有标签意味着没有标签依赖的捷径。网络必须学习实际的视觉特征。

*[如果有架构图就指向它]*

这是单次前向传递——比标准 FF 更高效。而且它是完全无监督的。

---

## Slide 14: CwC-FF Results - The Only Method That Works

### English

*[This is the payoff - speak with confidence]*

Here are the results.

CwC-FF transfer accuracy: 89%.

*[Let that number land]*

That's not a typo. 89%. Compared to 42% for standard FF. Compared to 45% for random initialization.

*[Walk through the comparison]*

- Standard FF: 42% - worse than random
- Bio-inspired modifications: 35-47% - still struggling
- Layer Collaboration: 51% - modest improvement
- CwC-FF: 89% - actually competitive with backprop at 85%

CwC-FF isn't just better. It's the ONLY FF variant we tested that actually achieves meaningful transfer learning.

The lesson: the label embedding isn't just a detail. It's the root cause of the transfer failure. Remove it, and FF works.

### 中文

*[这是回报——自信地说]*

这是结果。

CwC-FF 迁移准确率：89%。

*[让这个数字沉淀]*

这不是打字错误。89%。相比标准 FF 的42%。相比随机初始化的45%。

*[走过比较]*

- 标准 FF：42% - 比随机还差
- 生物启发修改：35-47% - 仍在挣扎
- 层协作：51% - 适度改善
- CwC-FF：89% - 实际上与85%的反向传播具有竞争力

CwC-FF 不只是更好。它是我们测试的唯一一个真正实现有意义迁移学习的 FF 变体。

教训：标签嵌入不只是一个细节。它是迁移失败的根本原因。移除它，FF 就能工作。

---

## Slide 15: Key Insights - Three Takeaways

### English

Let me summarize with three key insights.

*[Count on fingers as you go through these]*

**First: Simple negative sampling wins for training, but loses for transfer.**
Hinton's label embedding gives the best accuracy on the source task. But it creates a shortcut that destroys transferability. There's a fundamental tension here.

**Second: Bio-inspired solutions fail when they don't address the root cause.**
We tried dopamine-like modulation, predictive coding, sparse representations. All are real features of biological neural systems. None helped because they don't address the label embedding problem. It's not enough to be biologically inspired - you need to solve the right problem.

**Third: Label-free learning is key for transferable representations.**
The only successful method - CwC-FF - removes labels from the input entirely. This forces the network to learn actual visual features rather than label-pixel correlations. This might be a general principle: supervised signals in the input create non-transferable shortcuts.

### 中文

让我用三个关键洞察来总结。

*[边讲边用手指数]*

**第一：简单的负样本策略赢得训练，但输掉迁移。**
Hinton 的标签嵌入在源任务上给出最佳准确率。但它创造了一个破坏可迁移性的捷径。这里有一个根本性的张力。

**第二：当生物启发解决方案不解决根本原因时，它们会失败。**
我们尝试了类似多巴胺的调节、预测编码、稀疏表征。所有这些都是生物神经系统的真实特征。没有一个有帮助，因为它们不解决标签嵌入问题。仅仅受生物学启发是不够的——你需要解决正确的问题。

**第三：无标签学习是可迁移表征的关键。**
唯一成功的方法——CwC-FF——完全从输入中移除标签。这迫使网络学习实际的视觉特征，而不是标签-像素相关性。这可能是一个普遍原则：输入中的监督信号创造不可迁移的捷径。

---

## Slide 16: Conclusion - What This Means for FF's Future

### English

So what does this mean for the future of Forward-Forward?

The algorithm isn't dead. But the original formulation with label embedding has a fundamental flaw for any application requiring transfer.

If you want FF for single-task learning in resource-constrained environments - edge devices, neuromorphic chips - the original method works fine. Use the simple label embedding.

But if you want FF for foundation models, for pre-training that transfers to downstream tasks, you MUST use label-free variants like CwC-FF.

*[Pause]*

There's also a deeper implication. Biological plausibility isn't just about "no backward pass." The brain does many things that we might not think are essential - sparse coding, neuromodulation, predictive processing - but our experiments show these don't automatically improve transfer.

What matters is whether the learning objective encourages genuinely useful representations. The brain solves this somehow. We're still figuring out how.

*[Final pause]*

Questions?

### 中文

那么这对 Forward-Forward 的未来意味着什么？

这个算法没有死亡。但原始的标签嵌入公式对于任何需要迁移的应用都有一个根本性的缺陷。

如果你想在资源受限的环境中使用 FF 进行单任务学习——边缘设备、神经形态芯片——原始方法工作得很好。使用简单的标签嵌入。

但如果你想让 FF 用于基础模型，用于可迁移到下游任务的预训练，你必须使用像 CwC-FF 这样的无标签变体。

*[停顿]*

还有一个更深层的含义。生物学合理性不只是关于"没有反向传递"。大脑做很多我们可能认为不是必需的事情——稀疏编码、神经调节、预测处理——但我们的实验表明这些不会自动改善迁移。

重要的是学习目标是否鼓励真正有用的表征。大脑以某种方式解决了这个问题。我们仍在弄清楚如何做到。

*[最后的停顿]*

有问题吗？

---

## Slide 17: References

### English

I want to acknowledge the key papers that informed this work.

Hinton's original 2022 paper that introduced Forward-Forward.

Brenig et al. 2023 - they first showed FF has transfer problems, which prompted our deeper investigation.

Lorberbom et al. at AAAI 2024 for Layer Collaboration.

Papachristodoulou et al. for CwC-FF, which turned out to be the solution.

And the many neuroscience papers on three-factor learning, predictive coding, and sparse representations that inspired our attempts.

*[If there's time]*

The code and detailed results will be available on our repository. Happy to discuss any of these methods in more detail.

Thank you.

### 中文

我想感谢为这项工作提供信息的关键论文。

Hinton 2022年的原始论文介绍了 Forward-Forward。

Brenig 等人 2023年——他们首先展示了 FF 有迁移问题，这促使我们进行更深入的调查。

Lorberbom 等人在 AAAI 2024 的层协作工作。

Papachristodoulou 等人的 CwC-FF，它最终成为解决方案。

以及许多关于三因子学习、预测编码和稀疏表征的神经科学论文，它们启发了我们的尝试。

*[如果有时间]*

代码和详细结果将在我们的仓库中提供。很乐意更详细地讨论这些方法中的任何一个。

谢谢。

---

# Appendix: Q&A Preparation

## Anticipated Questions and Answers

### Q: Why didn't you test on larger datasets like ImageNet?

**EN:** Great question. We focused on MNIST/Fashion-MNIST and CIFAR because the compute requirements for FF are still significant, and we wanted to thoroughly explore the method space first. ASGE from ICASSP 2026 has started pushing to ImageNet scale, and combining CwC-FF with their techniques is a natural next step.

**中文:** 很好的问题。我们专注于 MNIST/Fashion-MNIST 和 CIFAR，因为 FF 的计算要求仍然很大，我们想先彻底探索方法空间。ICASSP 2026 的 ASGE 已经开始推向 ImageNet 规模，将 CwC-FF 与他们的技术结合是自然的下一步。

### Q: Isn't CwC-FF just contrastive learning?

**EN:** There's a connection, yes. Both use competition to define positive and negative signals. But CwC-FF is more localized - competition happens within channels of the same layer, not between different augmentations of the same image. This makes it more compatible with FF's layer-local learning. You could see it as bringing contrastive principles into the FF framework.

**中文:** 是的，有联系。两者都使用竞争来定义正负信号。但 CwC-FF 更局部化——竞争发生在同一层的通道之间，而不是同一图像的不同增强之间。这使它与 FF 的层局部学习更兼容。你可以把它看作是将对比原则带入 FF 框架。

### Q: What about CNNs and Transformers?

**EN:** Our experiments used MLPs to isolate the learning algorithm effects. Recent work like DeeperForward and ASGE has shown FF can work with CNNs. Transformers are trickier because of the attention mechanism, but there's early work there too. The label embedding problem we identified should apply regardless of architecture.

**中文:** 我们的实验使用 MLP 来隔离学习算法的效果。像 DeeperForward 和 ASGE 这样的最新工作已经表明 FF 可以与 CNN 一起工作。Transformer 因为注意力机制更棘手，但那里也有早期工作。我们发现的标签嵌入问题应该适用于任何架构。

### Q: Is FF actually used in any real applications?

**EN:** Not widely yet. The main interest is in neuromorphic computing - chips that mimic brain architecture. Intel's Loihi, IBM's TrueNorth, and similar projects could benefit from BP-free training. There's also interest from the edge computing community where memory is precious. But for mainstream deep learning, BP still dominates.

**中文:** 还没有广泛使用。主要兴趣在神经形态计算——模仿大脑架构的芯片。Intel 的 Loihi、IBM 的 TrueNorth 和类似项目可以从无反向传播训练中受益。边缘计算社区也有兴趣，因为那里内存很珍贵。但对于主流深度学习，反向传播仍然占主导地位。

---

# Timing Guide

| Slide | Topic | Target Time | Cumulative |
|-------|-------|-------------|------------|
| 1 | Title | 1 min | 1 min |
| 2 | Motivation | 1.5 min | 2.5 min |
| 3 | FF Algorithm | 1.5 min | 4 min |
| 4 | Research Questions | 1 min | 5 min |
| 5 | Negative Sampling | 1.5 min | 6.5 min |
| 6 | Transfer Paradox | 2 min | 8.5 min |
| 7 | Root Cause | 1.5 min | 10 min |
| 8 | Bio-Inspired Intro | 1 min | 11 min |
| 9 | Three-Factor | 1 min | 12 min |
| 10 | Prospective FF | 1 min | 13 min |
| 11 | PCL-FF | 1 min | 14 min |
| 12 | Layer Collab | 1 min | 15 min |
| 13 | CwC-FF Solution | 1.5 min | 16.5 min |
| 14 | CwC-FF Results | 1 min | 17.5 min |
| 15 | Key Insights | 1.5 min | 19 min |
| 16 | Conclusion | 1 min | 20 min |
| 17 | References | 0.5 min | 20.5 min |

**Buffer for Q&A:** 5-10 minutes

---

# Presentation Tips

1. **The transfer paradox (Slide 6) is your "wow" moment** - slow down, make eye contact, let the numbers sink in

2. **Use hand gestures** when counting the three insights (Slide 15)

3. **The bio-inspired failures are important context** - don't rush through them, they show rigor

4. **CwC-FF should feel like a revelation** - build anticipation through the failures

5. **Practice the bilingual transitions** - some audiences may want to switch mid-presentation

6. **Have backup slides** with detailed results tables in case of technical questions
