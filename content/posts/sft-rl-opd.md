---
title: "SFT → RL → OPD：从分布视角重新理解大模型后训练"
date: 2026-08-24T17:40:00+08:00
draft: false
tags: ["OPD", "后训练", "笔记"]
categories: ["AI 技术思考"]
math: true
summary: "后训练方法通常按“用什么 loss”来分类，但更关键的问题是：训练数据来自哪里、模型在分布的哪些区域被更新。沿着“状态分布 × 学习信号”这个坐标系，SFT、RL、OPD 的差异一下子清晰起来。"
---

> 原文：[SFT, RL, and On-Policy Distillation Through a Distributional Lens](https://nrehiew.github.io/blog/sft_rl_opd/)（nrehiew）

我们通常按照训练目标区分大模型后训练方法：

- SFT：拟合标准答案；
- RL：最大化奖励；
- OPD：匹配教师分布。

但这种分类只看到了「用什么 loss」，没有回答另一个可能更重要的问题：

> 训练数据来自哪里？模型究竟在分布的哪些区域被更新？

如果把语言模型理解为一个序列概率分布：

$$\pi_\theta(y\mid x) = \prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{\lt t}),$$

那么后训练的本质，就是重新分配不同生成轨迹上的概率质量。

![SFT / RL / OPD 三种后训练方法的分布示意对比](/images/opd/distributions_cover.png)

> 图片来源：nrehiew，*SFT, RL, and On-Policy Distillation Through a Distributional Lens*

沿着这一视角，SFT、RL 和 OPD 的根本差异可以拆成两个维度：

$$\boxed{\ \text{Post-training} = \text{Data Distribution} + \text{Learning Signal}\ }$$

也就是：

1. **模型在哪里学习；**
2. **什么信号告诉模型应该如何改变。**

## 一、SFT：被外部数据分布牵引

设监督数据集为：

$$\mathcal D=\{(x,y^\star)\},$$

SFT 的目标是：

$$\mathcal L_{\mathrm{SFT}} = -\,\mathbb E_{(x,y^\star)\sim\mathcal D}\left[\sum_t \log\pi_\theta\left(y_t^\star\mid x,y_{\lt t}^\star\right)\right].$$

训练状态为：

$$s_t^\star=(x,y_{\lt t}^\star).$$

这些状态由外部数据决定，而不是由当前模型决定。模型无论原来会生成什么，都要被拉向数据集中给定的 token。

![普通 SFT 的分布更新示意](/images/opd/sft_distribution.svg)

> 普通 SFT：交叉熵对数据集里所有 token 无差别施加梯度（灰色虚线 *Dense SFT Update Pressure*），既产生指向新任务的有益更新（绿色 *Useful Update*），也产生侵蚀基座旧能力的附带更新（橙色 *Collateral Update*）——这正是灾难性遗忘的来源。图片来源同上。

在经验分布意义上，交叉熵等价于最小化正向 KL：

$$D_{\mathrm{KL}}\left(p_{\mathcal D}\,\Vert\,\pi_\theta\right).$$

因此，SFT 的分布结构可以概括为：

$$\boxed{\ s_t\sim d^{\mathcal D},\qquad \text{signal}=y_t^\star\ }$$

SFT 的优势非常明确：监督密集、优化稳定，尤其适合冷启动、格式塑造和基础指令遵循。

但它也存在一个结构性问题：所有示范 token 都会受到直接监督。对于任务关键 token（如「因为 $a^2+b^2=c^2$」）SFT 会提高其概率；对于偶然出现的风格 token（如「显然」「因此我们容易得到」）SFT 同样会提高其概率。

损失函数本身并不知道：

> **这个 token 是决定答案正确性的关键步骤，还是数据集中的表达习惯？**

因此，当训练数据分布与原模型相距较远时，SFT 可能产生范围较广的参数更新，并通过参数共享间接影响原有能力。

需要特别强调：这并不意味着「正向 KL 必然造成遗忘」。更准确地说，遗忘来自以下因素的共同作用：

> **外部状态分布 ＋ 密集 token 监督 ＋ 缺少保留原策略的内在约束**

## 二、RL：从模型自己的行为出发寻找高奖励方向

在线强化学习首先让当前策略生成回答：

$$y\sim\pi_\theta(\cdot\mid x),$$

然后由奖励函数进行评价：

$$R(x,y).$$

策略梯度可以抽象写成：

$$\nabla_\theta J(\theta) = \mathbb E_{y\sim\pi_\theta}\left[\sum_t A_t\,\nabla_\theta\log\pi_\theta(y_t\mid s_t)\right].$$

其关键并不只是出现了 reward，而是训练状态来自当前策略：

$$\boxed{\ s_t\sim d^{\pi_\theta}\ }$$

因此，RL 主要在模型当前能够访问的轨迹附近调整概率质量：

**当前会生成的行为 → 其中更高奖励的行为**

![RL(PPO) 的分布更新示意](/images/opd/rl_distribution.svg)

> RL(PPO)：只在当前策略自己采样出的 on-policy 样本（绿点）上施加梯度，把概率质量向高奖励区域（橙圈样本）收缩（蓝色虚线 *Updated Policy*）。更新局限在模型已能到达的高概率区域，因此较少误伤无关旧能力。图片来源同上。

它不会像 SFT 一样直接指定一个可能与模型相距很远的完整目标序列分布，而是在当前分布附近寻找能够提高奖励的方向。这产生了一种局部性：

$$\boxed{\ \text{on-policy sampling}\Rightarrow\text{updates concentrated on locally visited behavior}\ }$$

《[RL's Razor](https://arxiv.org/abs/2509.04259)》将这种现象概括为：在多种都能解决任务的策略中，在线 RL 会隐式偏向与原策略 KL 距离较小的解。

![On-policy 训练在策略空间中就近收敛到最优解](/images/opd/on_policy.svg)

> 策略空间视角：on-policy 采样把更新约束在模型可行策略集合 $\Pi$ 内部，在所有最优策略 $P^\star$ 中隐式收敛到离初始策略 $\pi_0$ 最近的最优解 $\pi^\star$（*nearest optimal*）。不被拽向遥远的外部目标，因此更能保留基座能力。图片来源同上。

但这里需要保持严谨。不能把它无条件写成所有 RL 算法显式求解的目标：

$$\pi^\star = \arg\min_{\pi:R(\pi)=R^\star} D_{\mathrm{KL}}(\pi\,\Vert\,\pi_0)$$

更稳妥的表述是：

$$\boxed{\ \text{on-policy learning dynamics tend to favor locally reachable, smaller-shift solutions}\ }$$

这是一种由采样和优化动力学引入的隐式偏置，而不是对任意 RL 配置均成立的绝对保证。

## 三、OPD：student 决定在哪里学，teacher 决定向哪里改

On-Policy Distillation 位于 SFT 和 RL 之间。它首先由 student rollout：

$$y\sim\pi_S(\cdot\mid x),$$

形成状态：

$$s_t=(x,\,y_{\lt t}),\qquad s_t\sim d^{\pi_S}.$$

然后 teacher 在 student 已经到达的状态上给出下一 token 分布：

$$\pi_T(\cdot\mid s_t).$$

以 reverse KL 为例，目标可以写成：

$$\mathcal L_{\mathrm{OPD}} = \mathbb E_{s_t\sim d^{\pi_S}}\left[D_{\mathrm{KL}}\left(\pi_S(\cdot\mid s_t)\,\Vert\,\pi_T(\cdot\mid s_t)\right)\right].$$

于是 OPD 同时具有两种性质：

- 像 RL 一样，数据来自当前 student；
- 像蒸馏一样，使用 teacher 的稠密 token 分布监督。

其最核心的分工是：

$$\boxed{\ \text{Student controls support}\ }\qquad\boxed{\ \text{Teacher controls local direction}\ }$$

通俗地说：

> student 决定走到哪里；teacher 只负责告诉它，在已经走到的位置下一步应该怎么走。

这也是 OPD 与普通离线蒸馏最本质的区别。

## 四、「teacher 的坏行为没有进入梯度」应该如何严谨表达？

这里容易出现一个概念混淆。teacher 本身不是某个 state，因此不能写成：

$$\text{teacher}\in\text{bad region}.$$

更准确的做法是定义 teacher 的坏行为区域 $\mathcal B_T$：即所有「teacher 分布 $\pi_T(\cdot\mid s)$ 在状态 $s$ 上给出错误或退化监督」的状态 $s$ 组成的集合。

然后考察 student 生成的状态是否落入该区域：

$$\boxed{\ s_t\in\mathcal B_T\ }$$

也就是：**student 走进了 teacher 的 bad region。** teacher 的坏行为进入总体训练梯度的程度，取决于：

$$\Pr_{s_t\sim d^{\pi_S}}\!\left(s_t\in\mathcal B_T\right).$$

如果 $\Pr_{s_t\sim d^{\pi_S}}\!\left(s_t\in\mathcal B_T\right)\approx 0$，那么 teacher 即使在其他任务区域发生了明显退化，那些退化行为也很少被查询，自然不会大量进入 OPD 梯度。因此：

> **teacher 整体发生遗忘 ⇏ student 会完整继承 teacher 的遗忘**

真正决定哪些 teacher 行为会被蒸馏的，是 student 的状态分布与 teacher bad region 的交集 $d^{\pi_S}\cap\mathcal B_T$。

## 五、为什么要引入 OPSD？

OPSD，即 On-Policy Self-Distillation，是理解 OPD 信用分配问题的一种理想「实验探针」。

在普通 OPD 中，teacher 与 student 往往是不同模型。两者 KL 较大，可能来自：

> **能力差异 ＋ 知识差异 ＋ 风格差异 ＋ 表达习惯差异**

因此，我们无法判断这种 teacher-student disagreement 究竟是不是真正的 task importance。

OPSD 则让同一个模型同时扮演 teacher 和 student（$\theta_T=\theta_S$）。区别只在于 teacher 可以看到额外的正确答案或推理轨迹 $z$：

$$\pi_T(\cdot\mid s_t,z),$$

而 student 只能看到原始问题 $\pi_S(\cdot\mid s_t)$。训练目标为：

$$D_{\mathrm{KL}}\left(\pi_S(\cdot\mid s_t)\,\Vert\,\pi_T(\cdot\mid s_t,z)\right).$$

这样就基本排除了「teacher 模型更大、更强」这一混杂因素。

![OPSD 的逐 token KL 分析](/images/opd/opsd_token.png)

> OPSD 的逐 token 分析：较高的 teacher-student KL 常落在 *wait*、*alright* 等风格/转折 token 上，而 *exponent*、*logarithm* 等数学关键 token 的 KL 反而较低——说明「KL 大 ≠ 该 token 对任务更重要」。图片来源同上。

然而，[OPSD 论文](https://arxiv.org/abs/2601.18734)的逐 token 分析发现，较高 KL 经常出现在 "wait""alright" 等风格或转折 token 上，而 "power""exponent""logarithm" 等数学 token 的 KL 反而较低。这暴露出一个关键问题：

> **$D_{\mathrm{KL}}$ 大 ⇏ 该 token 对任务更重要**

teacher-student KL 只能说明「teacher 和 student 在这里意见不同」，不能直接说明「修改这个位置能够使最终答案更正确」。因此，OPSD 不是在证明 OPD 无效，而是在揭示 OPD 的信用分配偏差：

$$\boxed{\ \text{Teacher disagreement}\neq\text{Task credit}\ }$$

这也是 OPSD 需要进行逐 token clipping 的原因：它要避免模型过度优化高 KL、但任务价值较低的风格 token。

## 六、RL 与 OPD：相同的「在哪里学」，不同的「根据什么学」

RL 与 OPD 都是 on-policy，但监督信号完全不同。RL 的更新权重近似为优势函数 $A(s_t,a_t)$，表示某个动作相对于基线带来了多少额外回报。OPD 的更新信号则来自：

$$\log\frac{\pi_T(a_t\mid s_t)}{\pi_S(a_t\mid s_t)},$$

表示 teacher 相比 student 更偏好这个 token 的程度。因此：

$$\boxed{\ \text{RL signal} = \text{reward advantage}\ }\qquad\boxed{\ \text{OPD signal} = \text{teacher disagreement}\ }$$

前者与任务结果的关系更直接，但通常较稀疏；后者能够提供逐 token 的稠密监督，但可能包含 teacher 的风格偏好、错误和不确定性。这形成了后训练中的核心权衡：

| 方法 | 状态分布 | 信号密度 | 信号含义 |
| --- | --- | --- | --- |
| SFT | 外部固定数据 | 稠密 | 模仿标准序列 |
| RLVR | On-policy | 稀疏 | 最大化可验证结果 |
| OPD | On-policy | 稠密 | 匹配 teacher 分布 |
| 理想方法 | On-policy | 稠密 | 低偏差、任务对齐的 credit |

## 七、为什么 OPD student 可能比 teacher 更好？

原文在 Minimal Code Editing 任务上进行了一个很有启发性的实验。作者分别训练了一个 SFT teacher 和一个 RL teacher。SFT teacher 出现了更明显的通用代码能力遗忘，而 RL teacher 的泛化与能力保持更好。

按照传统蒸馏直觉，来自 RL teacher 的 student 应当明显优于来自 SFT teacher 的 student。但实验结果并非如此：

| 模型 | Pass@1 ↑ | 归一化编辑距离 ↓ | LiveCodeBench ↑ |
| --- | ---: | ---: | ---: |
| SFT teacher | 0.775 | 0.450 | 0.286 |
| RL teacher | 0.792 | 0.063 | 0.320 |
| OPD + SFT teacher | 0.800 | 0.059 | 0.297 |
| OPD + RL teacher | 0.787 | 0.055 | 0.314 |

两个 OPD student 的表现十分接近。更意外的是，使用已经发生遗忘的 SFT teacher，student 仍然没有完整继承其遗忘。这说明：

> **teacher 提供什么信号很重要，但 student 在什么状态上询问 teacher 同样重要。**

OPD student 甚至可能在特定 benchmark 上超过 teacher，因为 teacher 不再负责生成完整轨迹，而只是在 student 真正访问的状态上提供局部建议。这类似于：

> 教练自己的比赛成绩不一定最高，但他仍然可以在运动员最容易犯错的位置提供有效指导。

不过，「超过 teacher」通常只意味着 $\operatorname{Eval}(\pi_S) > \operatorname{Eval}(\pi_T)$ 在特定评价任务上成立，而不是 student 在所有能力维度上都强于 teacher。

## 八、为什么 On-policy 能改善泛化？

普通离线蒸馏的训练状态来自 teacher（$s_t\sim d^{\pi_T}$）。但部署时，student 实际访问的是 $s_t\sim d^{\pi_S}$。自回归生成中，一个早期 token 偏差就可能让后续 prefix 逐渐离开 teacher 的训练分布，产生：

$$\boxed{\ \text{covariate shift}\ }\qquad\boxed{\ \text{compounding error}\ }$$

OPD 直接在 student 自己生成的 prefix 上询问 teacher（$s_t\sim d^{\pi_S},\ \pi_T(\cdot\mid s_t)$），因此 $\text{training states}\approx\text{inference states}$。

这与 DAgger 的交互式模仿学习思想高度一致，也正是 [GKD](https://arxiv.org/pdf/2306.13649) 将语言模型蒸馏重新解释为 interactive expert imitation learning 的原因：

> **OPD ≈ 面向自回归语言模型的分布式 DAgger**

student 去自己会犯错的地方，再让 teacher 在那里提供监督。

## 九、文章真正提出的 Δ 是什么？

传统理解通常将后训练方法划分为：

$$\boxed{\ \text{SFT vs RL} = \text{supervised loss vs reward optimization}\ }$$

而这篇文章提出了一个更具有解释力的坐标系：

> **算法差异 ＝ 状态分布 × 学习信号**

由此，RL 的特殊性可能并不完全来自 policy gradient 本身。因为 OPD 没有直接最大化任务 reward，却保留了 $y\sim\pi_S$ 这一 on-policy 结构，并表现出部分与 RL 相似的性质：

- 较少遗忘；
- 较好的分布匹配；
- 较强的任务外泛化；
- student 偶尔超过 teacher。

所以 OPD 在文章中不仅是一种训练算法，更是一个「实验探针」：

> **移除 RL reward，保留 on-policy sampling，观察 RL-like properties 是否仍然存在**

实验结果促使作者提出：

> **on-policy data 可能是 RL 与 OPD 的承重部件**

但这一结论目前仍应被视为综合研究假说，而不是已经证明的普适定理。

## 十、下一代后训练算法缺少什么？

现在可以把后训练拆成三个问题：

1. **Where do we learn？** $\quad\text{Offline}\ \text{or}\ \text{On-policy}$
2. **What tells us what to learn？** $\quad\text{Demonstration}\ /\ \text{Reward}\ /\ \text{Teacher distribution}$
3. **How precisely is credit assigned？** $\quad\text{Sequence-level sparse}\ \text{or}\ \text{Token-level dense}$

于是三种方法可以概括为：

$$\text{SFT}:\ \boxed{\ \text{Offline}+\text{Dense}+\text{Imitation}\ }$$

$$\text{RLVR}:\ \boxed{\ \text{On-policy}+\text{Sparse}+\text{Task-aligned reward}\ }$$

$$\text{OPD}:\ \boxed{\ \text{On-policy}+\text{Dense}+\text{Teacher-biased signal}\ }$$

因此，理想的下一代算法需要同时满足：

$$\boxed{\ \text{On-policy}+\text{Dense credit}+\text{Low-bias / Task-aligned}\ }$$

它既能像 RL 一样知道「什么行为真正提高了任务成功率」，又能像蒸馏一样知道「具体在哪个位置、哪个 token 上应该修改」，同时保持训练分布与当前模型一致。

## 结语：后训练不仅是学习正确答案，更是在控制模型如何移动

从分布视角看，SFT、RL 与 OPD 并不是三种彼此割裂的方法。它们分别回答了后训练的三个不同问题：

- SFT：目标行为长什么样；
- RL：哪些完整行为能够获得更高回报；
- OPD：在 student 自己访问的位置，teacher 会如何调整下一步分布。

真正影响遗忘、泛化和能力迁移的，不只是监督信息量，也不只是 loss 的名称，而是：

> **模型在哪里被训练 × 训练信号指向哪里 × 信用分配是否准确**

因此，SFT → RL → OPD 所体现的并不是简单的算法替换，而是一种后训练范式的逐步演进：

> **拟合外部答案 → 在自身分布上优化结果 → 在自身分布上吸收稠密专家信息**

而尚未解决的最终问题是：

> **如何在 on-policy 数据上，获得稠密、低偏差且真正任务对齐的 credit？**

这可能才是下一代大模型后训练算法真正需要突破的方向。
