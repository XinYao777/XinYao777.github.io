---
title: "Why RL Matters：SFT、RL 与 Sequence Distillation 的分工"
date: 2026-08-24T14:10:00+08:00
draft: false
tags: ["RL", "蒸馏", "笔记"]
categories: ["AI 技术思考"]
math: true
summary: "读青稞AI《从 RL 到蒸馏：Why RL Matters？》后的理论化归纳。把后训练拆成 Demonstrate（SFT）→ Explore（RL）→ Transfer（Sequence Distillation）三段：RL 承担“发现”的成本，蒸馏承担“复制”的成本。"
---

> 原文：[《从 RL 到蒸馏，再到软蒸馏：Why RL Matters？》](https://zhuanlan.zhihu.com/p/2067725096886785009)（作者 ybq，知乎；[青稞AI 转载](https://mp.weixin.qq.com/s/S29z51EByrjWJgkvr6vwhQ)）

这篇文章真正讨论的核心，其实不是「RL 和蒸馏谁更强」，而是一个更基础的问题：**在大模型后训练阶段，好的能力究竟是如何产生、如何迁移、又如何适配到不同模型上的？**

把文章的工程经验稍作理论化，整个过程可以概括成三个环节：

- **SFT** 负责提供一个好的策略初始化；
- **RL** 负责在模型自身的策略分布中搜索更高奖励的行为；
- **Sequence Distillation** 负责把已经搜索到的高质量行为迁移给其他模型。

于是全文隐含的主线是：

$$\text{SFT} \rightarrow \text{RL Exploration} \rightarrow \text{Sequence Distillation} \rightarrow \text{Student Adaptation}$$

其中最重要的区别是：**RL 主要承担「发现」的成本，蒸馏主要承担「复制」的成本。**

## 一、好的 CoT Pattern 到底是什么

对于 `997 × 1003`，存在三种推理方式：

- **good**：`997 × 1003 = (1000 − 3)(1000 + 3) = 1000² − 3² = 999991`
- **mediocre**：`997 × 1003 = 1003 × (900 + 90 + 7) = 902700 + 90270 + 7021 = 999991`
- **bad**：`997 × 1003 = 999991`

三个答案都可能正确，但泛化能力显然不同。文章把这种差异统称为 CoT Pattern，而从理论上还可以拆成三层：

1. **Reasoning Strategy**：采用什么求解策略，例如平方差公式 $(a-b)(a+b) = a^2 - b^2$；
2. **Reasoning Trajectory**：真正执行出来的推理路径（`997×1003 → (1000−3)(1000+3) → 1000²−3² → 999991`）；
3. **Surface Realization**：这个策略最终被表达成怎样的 token sequence——强模型可以只写一行，弱模型可能要写出更多中间步骤。

所以「好的 CoT Pattern」更严格地说是：**好的 reasoning strategy + 一条高质量 reasoning trajectory + 一种适合当前模型能力的表达方式**。这三者并不等价。

## 二、SFT 真正承担的作用

文章说 SFT 给 CoT Pattern「定型」。理论化的表达更准确：**SFT 在改变模型的初始策略分布。** 设模型策略为 $\pi_\theta(y\mid x)$，SFT 的目标是

$$\mathcal{L}_{\text{SFT}} = -\,\mathbb{E}_{(x,\,y^*)} \sum_t \log \pi_\theta\!\left(y^*_t \mid x,\, y^*_{<t}\right)$$

它做的事非常直接：**提高训练数据中这些 demonstration trajectory 的概率。**

关键在于：如果某种优质 strategy 在 pretrained model 中出现概率极低（比如 $P(y_{\text{good}}\mid x)=10^{-5}$），那么即使它存在于模型能力空间中，RL 也几乎无法通过有限 rollout 找到它。设优质 trajectory 的采样概率为 $p$，每题 rollout $N$ 次，至少发现一次的概率为

$$P(\text{discover}) = 1 - (1-p)^N$$

- $p = 0.1,\ N = 32$：约 **96.6%**；
- $p = 10^{-5},\ N = 128$：只有约 **0.128%**。

所以 SFT 对 RL 最重要的作用，不是严格意义上的「决定上限」，而是**把好的行为从「几乎探索不到」移动到「RL 可以探索到」的区域**——SFT shapes the exploration support。

这也解释了：一个好的 SFT checkpoint 不能只看结束时的准确率，更要看它给 RL 留下了怎样的策略分布——好行为有没有足够概率被采样、模型是否还保留足够的策略熵、潜在优质 strategy 有没有被提前激活。**SFT 不只是「把准确率做高」的阶段，它还是 RL 的初始化阶段。**

## 三、为什么 SFT 起点低的模型 RL 后反而更强

用极强 teacher 的 trajectory 去训较弱 student，SFT 后效果不一定好。原因是：**teacher 最优的 trajectory，不一定是 student 最容易执行的 trajectory。**

设 teacher 可实现的策略集合为 $\Pi_T$，student 能稳定实现的为 $\Pi_S$，通常 $\Pi_S \subset \Pi_T$，于是 teacher 的最优策略 $\pi_T^*$ 未必落在 student 容易实现的区域。例如强模型可以直接写 `99×99 = 99×(100−1) = 9801`（内部稳定补全被省略的运算），弱模型则需要把 `99×100 − 99×1 = 9900 − 99 = 9801` 全写出来。抽象 strategy 相同，但 trajectory 和 surface realization 不同。

所以文章说的「RL 对 CoT Pattern 做本土化改造」，理论化即：**RL 在 student 可实现的 policy class 中寻找高 reward 的 realization**——

$$\pi_S^* = \arg\max_{\pi \in \Pi_S} \mathbb{E}[R]$$

teacher 给 student 的不是最终答案，而更像是「优秀策略所在的大致方向」；student 仍需在自己的能力约束下找到最合适的实现形式。因此仅比较 SFT 初始准确率，有时会误判一个模型后续 RL 的潜力。

## 四、RL 到底为什么重要

SFT 与 RL 一个本质区别：**SFT 告诉模型「应该生成什么」，RL 告诉模型「什么结果是好的」。** SFT 提高 $\pi_\theta(y^*\mid x)$；RL 的目标更接近

$$\pi^* = \arg\max_{\pi}\ \mathbb{E}_{y\sim\pi}\big[R(x,y)\big]$$

模型不必严格复现某条人工指定的 trajectory，只需找到高 reward 的行为。而我们真正关心的目标往往是复合的——答案正确、格式合法、语言一致、长度不过长、不重复反思、工具调用不过多，于是 reward 可以写成

$$R = R_{\text{correct}} + \alpha R_{\text{format}} + \beta R_{\text{language}} - \lambda L_{\text{token}} - \mu C_{\text{tool}}$$

我们没有规定「必须按这条 CoT 做」，而是说「满足这些目标即可，实现方式你自己找」。因此可以把 RL 抽象成 **Constraint-guided Policy Search**——这是它相对 SFT 极重要的差异。

## 五、为什么 RL 很适合做「本土化」：训练分布视角

还有一个比 CoT Pattern 更基础的视角——**训练分布**。SFT 的状态来自 demonstration，即 $s_t = (x,\, y^*_{<t})$，模型看到的 prefix 是人工或 teacher 写出的正确 prefix，可近似认为 $s \sim d_{\text{teacher}}$。但实际部署时模型面对的是自己生成的 prefix，$s \sim d_\pi$。一旦某一步 $y_{<t} \neq y^*_{<t}$，就可能进入训练数据从未出现过的状态——这就是经典的 distribution mismatch，也与 exposure bias 相关。

而 RL 的训练轨迹来自模型自己，$y \sim \pi_\theta(\cdot\mid x)$，因此 $s \sim d_{\pi_\theta}$：

- **SFT learns on demonstration states.**
- **RL learns on student-generated states.**

所以「本土化」除了「找到适合自己的推理长度和方式」外，还有更深一层含义：**RL 可以在 student 自己真正会访问到的状态分布上做 policy improvement。**

## 六、为什么 RL 对可控性非常有效

文章列了很多实际问题：生成多个 `<think>`、中英文混杂、时而不反思时而连续反思几十次、输出长度不稳定、Agent 工具调用次数不合理。

文章说 SFT「不具备打压 Pattern 的能力」，这个说法略绝对。严格讲，交叉熵对 logit 的梯度为

$$\frac{\partial L}{\partial z_j} = p_j - \mathbb{1}[\,j = y^*\,]$$

目标 token 概率上升、竞争 token 概率相对下降，所以 SFT 并非只能「正向学习」。真正的问题是：**SFT 通常没有主动采样模型自己的 bad trajectory。** 如果「反思 → 再反思 → 无限循环」这种 prefix 从未出现在 SFT 数据中，SFT 就很难针对这个具体 failure mode 精确优化。

RL 则不同：模型先自己 rollout $y_{\text{bad}} \sim \pi_\theta$，再由 verifier / reward 给出 $R(y_{\text{bad}}) < 0$。即——**SFT 只能间接 suppress bad behaviors，RL 更容易显式优化 student 自己产生的 bad behaviors。** 这就是为什么长度控制、格式控制、思考深度、工具调用次数等问题，往往非常适合在 RL 阶段解决；标配的「思考档位」也是通过 RL 给不同档位不同上下文窗口实现的。

## 七、RL 更重要的价值：探索

SFT 只有有限条 demonstration，$D_{\text{SFT}} = \{y^*_1, \dots, y^*_M\}$；而 RL 可以不断 $y \sim \pi_\theta$，访问越来越多 trajectory。当 rollout 足够多，就可能出现训练数据里没有的行为 $y_{\text{new}}$ 且 $R(y_{\text{new}}) > R(y_{\text{SFT}})$，此时好的 RL 算法（如 GRPO）需要抓住这条高质量数据，给足学习信号让模型记住它。

所以 RL 至少可拆成两部分：

- **Adaptation**：teacher / SFT 已给出好的 strategy，RL 找到最适合当前模型能力的实现方式；
- **Exploration**：模型通过大量 sampling，有机会发现 demonstration 中没有的、更高 reward 的 trajectory。

原文所谓「RL 有一点基因突变 + 进化的味道」，翻译成理论语言就是 **Policy-space Exploration + Selection**。

## 八、为什么仅仅做蒸馏，就可能得到很强的模型

首先明确：本文说的「蒸馏」主要指 **Sequence-level Distillation**，而不是狭义的 logit KD。强 teacher 生成完整 response $x \to y_T$，student 用这些 response 做 SFT：

$$\mathcal{L}_{\text{distill}} = -\sum_t \log \pi_S\!\left(y^T_t \mid x,\, y^T_{<t}\right)$$

从优化算法看，它就是 SFT；从数据来源和知识迁移关系看，它是 Sequence Distillation。所以一个重要区分是：**SFT 描述「怎么优化」，Distillation 描述「知识从哪里来」。** 人工数据做交叉熵是 Human SFT；teacher 生成数据做同样的交叉熵，训练形式仍是 SFT，但性质是 Sequence Distillation。

因此文章说的「很多小作坊只做 SFT 不做 RL」，更准确的说法是：**很多团队只做 Teacher Response Distillation + SFT，而不自己承担大规模 RL Exploration。**

## 九、为什么 Sequence Distillation 能复制大量 RL 收益

假设一个很强的 teacher 已经过 RL。对一个 prompt $x$，它做了大量 rollout $\{y_1,\dots,y_K\}$，通过 verifier / reward 得到 $R(y_1),\dots,R(y_K)$，找到高质量 trajectory $y^*$。这个过程支付了昂贵的搜索成本，粗略写成

$$C_{\text{RL}} = C_{\text{rollout}} + C_{\text{reward}} + C_{\text{policy opt}}$$

但 student 不需要重走一遍，可以直接拿 $(x, y^*)$ 做监督学习 $-\log \pi_S(y^*\mid x)$。于是原本的「我该怎样从大量可能的 trajectory 中找到一个好的？」（搜索问题）被转换成「请模仿这条已经找到的好 trajectory」（模仿问题），计算成本完全不同。

所以 RL 与 Sequence Distillation 的关系可以极简概括为：**RL 负责搜索，Sequence Distillation 负责复制搜索结果。** 这正是「蒸馏可以窃取 RL 收益」的真义——它并非说 RL 没意义，恰恰相反：**如果没有某个模型先支付 RL 的 exploration cost，就没有这些高质量 trajectory 可供蒸馏。**

## 十、为什么资源有限的团队只做蒸馏也很合理

如果目标不是探索 frontier，而是快速获得某个已存在的能力，那么重新支付一遍 RL exploration cost 往往并不经济。应用团队完全可以走

$$\text{Strong Teacher} \to \text{Generate} \to \text{Filter} \to \text{Sequence Distillation} \to \text{Student SFT}$$

而无需自己 Massive Rollout → Reward → RL Optimization。这本质是 **Exploration Cost Amortization**：一个强 teacher 支付一次昂贵探索成本 $C_{\text{exploration}}$，随后大量 student（$\pi_{S_1}, \pi_{S_2}, \dots$）复用它产生的高质量 trajectory，平摊到每个 student 上的探索成本不断下降。

所以对 capability catch-up 而言，Sequence Distillation 的性价比可能极高。这些团队并没有证明 $\text{SFT} \approx \text{RL}$，而是在利用「有人已经替他们支付过 RL 的搜索成本」。

## 十一、那为什么蒸馏之后还需要 RL

因为 teacher 最优 trajectory 不一定等于 student 最优 trajectory。Sequence Distillation 解决的是 **Transfer**——把 teacher 已发现的高价值策略迁移给 student；但 student 仍有自己的模型容量、pretraining prior、representation、计算能力、推理稳定性、工具使用能力。teacher 给的 trajectory 可能过度压缩，也可能过于复杂。student 通过蒸馏知道「原来这个问题可以这样解决」，但仍需继续寻找「我怎样执行这个策略最稳定」。

于是自然的训练路径是：

$$\text{Teacher RL} \to \text{Sequence Distillation} \to \text{Student RL}$$

第一阶段 teacher 帮 student 解决 **What to do**，第二阶段 student RL 解决 **How should I do it**。用更抽象的语言：Sequence Distillation 负责 Transfer，Student RL 负责 Adaptation。这也印证了文章的工程观察——把 1T 模型 RL 探索出的优质 Pattern 蒸馏给 100B 模型，再让 100B 通过 RL 对这些 Pattern 做适配。

## 十二、RL 和蒸馏到底是什么关系

至此可以把整个问题压缩成一个清晰框架：

- **SFT — Policy Initialization**：提高优质 demonstration trajectory 的概率，让 RL 更容易探索到好策略；
- **RL — Exploration + Adaptation**：在 student 自己的状态分布中按 reward 搜索更优行为，并找到适合当前模型能力的实现方式；
- **Sequence Distillation — Capability Transfer**：把已经过昂贵搜索得到的高质量 trajectory 直接迁移给另一个模型。

三者不是竞争关系，而是一条自然的生产链：

$$\text{SFT} \to \text{RL Search} \to \text{High-quality Trajectories} \to \text{Sequence Distillation} \to \text{Student Adaptation}$$

## 十三、由此解释的产业现象（含软蒸馏）

对应用型团队，最重要的问题是「如何用最低成本把某个能力做到足够好」，答案很可能就是 **Strong Teacher + Sequence Distillation**，因为已有 frontier model 已帮你完成大量 exploration。

比抄 response 更高级的做法，是**借 sota 模型的能力优化自己的模型**——用 sota 合成能力边界上的数据、诊断现有模型的 pattern 缺陷、优化 RL 的 verifier，这就是原文所谓的**软蒸馏**。它同样是 Transfer 的一种，只是不落在 response 拷贝上。软蒸馏普遍存在，国内厂商几乎都不敢说优化过程里没有 GPT / Claude 的帮助——只要国外模型领先，国内模型就有源源不断的优化手段。

但对真正想推进 frontier 的团队，问题不同：如果所有团队都只做 distillation，新的高价值 trajectory 从哪里来？**Distillation 的前提永远是「存在一个更强的 teacher」。** 所以 distillation 擅长 Capability Transfer，而 RL 更重要的意义在于 Policy Improvement 与 Exploration。

## 十四、重新理解 Why RL Matters

因此 RL 的重要性，不应表述为「只有 RL 才能产生 reasoning」，也不能简单理解成「RL 一定比 SFT 强」。更准确的理解是：

- **SFT shapes where the policy starts.**
- **RL searches where the policy can improve.**
- **Sequence Distillation amortizes the cost of discovering that improvement.**

即：SFT 决定模型从哪里开始搜索；RL 在自己的策略空间中寻找更好的行为；Sequence Distillation 把别人已支付巨大搜索成本得到的成果低成本迁移过来。整个 post-training 的核心逻辑最终抽象成三个动词：

$$\textbf{Demonstrate}\ (\text{SFT}) \quad \textbf{Explore}\ (\text{RL}) \quad \textbf{Transfer}\ (\text{Sequence Distillation})$$

如果只是追赶已存在的能力，Transfer 往往是成本最低的方式；但如果希望继续产生新的能力和新的高价值策略，就必须有人继续承担 Exploration 的成本。这才是「从 RL 到蒸馏」背后真正的 **Why RL Matters**。
