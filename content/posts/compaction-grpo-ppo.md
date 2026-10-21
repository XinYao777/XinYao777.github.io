---
title: "Compaction 之后为什么要 split，又为什么不必放弃 GRPO？"
date: 2026-10-20T21:40:00+08:00
draft: false
tags: ["强化学习", "GRPO", "PPO", "Agent", "Compaction"]
categories: ["AI 技术思考"]
math: true
summary: "长任务经 compaction 产生不同数量、不同长度的训练段，并不使 GRPO 在数学上失效。从真实输入、概率比、训练记录、分组统计与损失聚合出发，澄清 split 的实现含义，以及选择 critic-based PPO 的合理动机与边界。"
---

GLM-5.2 的官方介绍将几个现象连接起来：长任务产生超长轨迹；compaction 将轨迹分成数量和长度各异的子段；训练因此从 group-wise optimization 转向 critic-based PPO。[1]

这很容易让人形成一个判断：**compaction 破坏了 GRPO 所要求的 group，因此必须 split，并且必须使用 PPO。**

这个判断过强。需要分别回答三个问题：模型真实看到了什么？训练系统如何组织这些记录？优势和损失如何计算？

本文主要采用一个明确设定：**压缩由环境或外部模块执行，其摘要是 actor 的输入，不是 actor 的待优化输出。**最后再区分这一设定与 CompactionRL 的联合摘要训练。[2]

## 一、三种长度与两种单位

| 概念 | 含义 | compaction 是否必然缩短它 |
|---|---|---|
| 完整任务的执行跨度 | 从开始到结束的全部动作与交互 | 否，压缩后任务继续执行 |
| 当前上下文长度 | 某次模型调用实际可见的输入 | 通常缩短，这是压缩的直接目的 |
| 单条训练序列长度 | 一条记录中组织的上下文及输出 | 按压缩边界切段后，通常可限制在工作窗口内 |

同时区分：**完整 rollout 是一次任务执行；segment 是这次执行中的一段训练记录。**一条 rollout 可以对应多个 segment，但切出更多记录并不意味着采样了更多次独立执行。

训练段更短，不能直接推出任务的信用分配跨度更短；训练记录更多，也不能直接推出 GRPO 的组成员更多。

## 二、没有 compaction，超长轨迹能否使用 GRPO？

可以。设从同一个初始任务 $x$ 采样 $G$ 次完整执行：

$$
\tau_i\sim\pi_{\mathrm{old}}(\cdot\mid x),\qquad R_i=R(\tau_i).
$$

以仅有最终结果奖励的常见 GRPO 形式为例：[3]

$$
\widehat A_i=\frac{R_i-\overline R}{\sigma_R+\epsilon},
\qquad \widehat A_{i,t}=\widehat A_i.
$$

各次执行不必等长，不同执行的第 $t$ 个动作也不必具有相同语义。它比较的是从同一个任务出发的完整结果。

超长轨迹会增加采样成本，使整条执行共享结果优势的信用分配更加粗糙。但它没有让上述组统计变得无法计算。

**共享最终奖励也不意味着策略梯度失效。**在固定环境、无折扣且忽略 clipping 与 KL 的简化条件下：

$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[R(\tau)],
$$

$$
\nabla_\theta J
=\mathbb E_\tau\left[
R(\tau)\sum_{t\in\mathcal A(\tau)}
\nabla_\theta\log\pi_\theta(a_t\mid c_t)
\right].
$$

$\mathcal A(\tau)$ 仅包含 actor 生成的 token，$c_t$ 是当时真实可见的上下文。环境转移可以包括固定的压缩操作。

以上只说明最终奖励可以产生梯度，不是在声称带组归一化、长度归一化和 clipping 的 GRPO 就是该目标的无偏估计。

## 三、compaction 实际发生了什么？

用一次压缩说明：

$$
A\sim\pi_{\mathrm{old}}(\cdot\mid x),\qquad
m=M(x,A),\qquad
B\sim\pi_{\mathrm{old}}(\cdot\mid x,m).
$$

$A$、$B$ 分别表示压缩前后生成的内容，$M$ 是环境侧压缩模块，$m$ 是摘要。为简洁起见，式中省略工具反馈和保留的近期交互；实际记录必须保留这些输入。

压缩后，模型看到的是 $x,m$，而非全部原历史。任务环境继续存在，例如已经修改的文件不会因上下文压缩而恢复原状。

训练 actor 时，$m$ 作为输入参与前向计算，但不计算其 policy loss。需要计算的是：

$$
\rho_{A,t}=\frac{\pi_\theta(A_t\mid x,A_{<t})}
{\pi_{\mathrm{old}}(A_t\mid x,A_{<t})},
$$

$$
\rho_{B,t}=\frac{\pi_\theta(B_t\mid x,m,B_{<t})}
{\pi_{\mathrm{old}}(B_t\mid x,m,B_{<t})}.
$$

**actor loss 所要求的是这些条件概率正确，而不是输入必须存储成几条记录。**分母通常直接使用 rollout 时保存的 old log-prob；分子在更新时重新计算。

## 四、split by compaction 的具体实现

一种直接的数据收集流程是：

1. 按实际调用记录模型输入、生成 token、action mask 与 old log-prob。
2. 环境触发 compaction 时，保存当前段。
3. 以实际压缩后的上下文创建下一段，继续同一个任务。
4. 任务结束后，将最终奖励与完整 rollout 关联。

例如：

~~~text
rollout_id: 42
initial_prompt_id: task_x
final_reward: 1

segment_1:
    context: x
    generated_actions: A
    old_log_probs: log_prob(A | actual_contexts)

segment_2:
    context: [x, m]
    generated_actions: B
    old_log_probs: log_prob(B | actual_contexts)
~~~

这是示意结构；多轮工具交互需要记录各动作实际对应的上下文。被保留到下一段的历史输出，在下一段是 context，不应因重复出现而再次计算同一动作的 loss。

为什么这样组织很自然？因为若直接使用普通 causal attention 处理 $[x,A,m,B]$，$B$ 就可以访问本应被删除的 $A$；若只保存 $[x,m,B]$，又丢失了训练 $A$ 所需的数据。

分成 $[x,A]$ 和 $[x,m,B]$ 两条记录，可以直接使用普通的段内因果注意力，分别复现真实输入。

**它们不必物理上独立计算。**可以分别前向、放入同一 batch，或 packing 到同一张量，用块间隔离的 attention mask、匹配实际调用的位置编号以及正确的 loss mask 计算。

尤其注意：只是把 $m$ 的 loss 设为零，并不能阻止其 hidden states 访问 $A$。恢复后分支的整个多层计算图都应遵循真实可见上下文。

所以，split 的直接作用是方便表示上下文切换，并限制单条训练序列长度。它没有要求创建新的 RL episode，也没有要求重建 GRPO group。如果本来就按每次模型调用记录输入和输出，甚至无需另设一种新的切段机制。

## 五、段数与段长不同，究竟会有什么问题？

设：

$$
\tau_i=(\sigma_{i,1},\ldots,\sigma_{i,K_i}),\qquad
T_i=\sum_{k=1}^{K_i}n_{i,k},
$$

其中 $n_{i,k}$ 仅统计该段待优化的 action token。

假设同一个任务的两次执行为：

| 完整执行 | 最终奖励 | segment 数量 |
|---|---:|---:|
| 第一次 | 1 | 1 |
| 第二次 | 0 | 3 |

完整 rollout 层面的奖励是 $[1,0]$，均值为 $1/2$。若把每段复制的奖励当成新的组成员，奖励就变成 $[1,0,0,0]$，均值为 $1/4$。

这不仅重复计算了相关样本，而且这些 segment 的局部输入通常不同，本来就不是同一局部 prompt 下的多次采样。

**这种做法有问题，但不是必须这样做。**正确保留完整 rollout 分组即可：先从 $G$ 个最终奖励计算 $\widehat A_i$，再用于对应 rollout 的所有 action token。

另外，loss 的聚合方式也需要明确。定义最大化方向的 clipped surrogate：

$$
\ell_{i,k,t}
=\min\left(
\rho_{i,k,t}\widehat A_i,
\operatorname{clip}(\rho_{i,k,t},1-\eta,1+\eta)\widehat A_i
\right).
$$

若对每段先平均，再让所有段等权：

$$
L_{\mathrm{segment}}
=\frac1{\sum_iK_i}\sum_{i,k}
\frac1{n_{i,k}}\sum_t\ell_{i,k,t},
$$

同一条执行切成更多段，通常会获得更大权重；短段中的单个 token 也会获得更大权重。

两种不依赖人为切段数量的聚合方式是：

$$
L_{\mathrm{rollout}}
=\frac1G\sum_i\frac1{T_i}\sum_{k,t}\ell_{i,k,t},
$$

$$
L_{\mathrm{token}}
=\frac{\sum_{i,k,t}\ell_{i,k,t}}{\sum_iT_i}.
$$

| 方式 | 等权单位 | 对较长完整执行的处理 |
|---|---|---|
| rollout 内平均，再对 rollout 平均 | 完整执行 | 每次执行的总权重相同 |
| 全 batch token 平均 | action token | token 更多的执行占比更大 |

在 token、条件上下文、优势与聚合范围不变时，将同一份数据分成两段或三段，都不会改变这两个目标。若改变 minibatch 更新顺序、逐批归一化或梯度裁剪，实际优化轨迹仍可能不同。

**Token 平均消除的是人为段数带来的权重变化，并不保证完整 rollout 等权，也不自动消除全部长度效应。**

变长数据还会带来 padding、packing 与设备负载问题。这些属于工程问题，PPO 同样需要处理。

## 六、中间输入不同，为什么还能属于同一个 group？

这里有两个层级。

**完整任务层级：**不同 rollout 从同一个 $x$ 出发，可以比较最终结果。它们走到的中间上下文不同，不妨碍这种比较。即使没有 compaction，不同工具调用和观察也会让中间输入不同。

**局部上下文层级：**若希望比较“从某个压缩后上下文 $c$ 继续执行”的不同选择，就需要固定 $c$，重新采样多条后续执行：

$$
\tau^{(1)}_{\mathrm{suffix}},\ldots,
\tau^{(G)}_{\mathrm{suffix}}\sim\pi(\cdot\mid c).
$$

涉及真实工具环境时，还需要恢复或复制对应环境状态，不能只复制摘要文本。

不同 rollout 的“第二段”并不自然构成这样的局部 group。它们的摘要、保留历史乃至真实环境状态都可能不同。

因此可以保留原始任务的 group，同时承认自己没有局部 group。**完整 rollout 的结果优势可以广播到各段，但不会因此变成状态条件化的局部优势。**

## 七、PPO 的合理动机在哪里？

结果监督的 rollout-level GRPO 回答：这次完整执行相对同任务的其他执行，结果如何？

Critic-based PPO 希望进一步利用当前上下文的价值预测：

$$
V_\phi(c_t)\approx\mathbb E[R_{\mathrm{future}}\mid c_t].
$$

它可以在不对每个中间上下文额外采样一组后续执行的情况下，提供状态条件化的 baseline，并允许同一任务只采样一次执行。这里的“单次”不意味着只用一条数据训练 critic；价值函数仍依靠整个训练数据分布学习。

| 长任务问题 | compaction / split 的影响 | 引入 critic 的关系 |
|---|---|---|
| 多次完整采样成本高 | compaction 可能减少后续 prefill；split 本身不减少完整执行次数 | 可以取消对每个任务多次采样的组统计要求，但增加 critic 成本 |
| 信用分配粗糙 | 仅切段没有提供新信号，真实时间跨度仍然存在 | 状态价值与优势估计可能改善，但依赖预测质量 |
| 二值奖励组内相同 | 切段不能制造新的结果差异 | critic 不依赖组内方差，但不能凭空创造可靠监督 |
| 策略陈旧与调度 | 没有必然改善，取决于同步、异步和更新机制 | 不是采用 PPO 就自动解决的问题 |

引入 PRM 可以提供额外过程信号，但是否产生有效差异，取决于其评分质量与奖励、优势的组织方式。它也不是 compaction 的自然结果。

在环境压缩丢失信息时，可见上下文不一定是充分的 Markov 状态。价值模型需要预测的是其输入条件下的期望回报，bootstrapping 也可能受部分可观测性与函数逼近误差影响。不能把 token-level advantage 直接等同于正确的因果归因。

此外，split 对 GAE 提出了额外要求。以统一的决策步定义为例：

$$
\delta_t=r_t+\gamma V_\phi(c_{t+1})-V_\phi(c_t),
\qquad
\widehat A_t=\delta_t+\gamma\lambda\widehat A_{t+1}.
$$

若任务尚未结束，切段边界就不能直接当作 terminal。跨段计算或适当 bootstrap 必须与真实后续执行一致；也不能未经说明就在每段结尾重复放置最终奖励。对工具反馈和压缩转移采用什么时间尺度，同样需要定义。

这不是说 PPO 不适合分段，而是说：**它同样需要额外设计，不能仅靠更换算法名称解决分段问题。**

## 八、如何准确理解原文？

官方介绍报告了迁移到 critic-based PPO 的选择。[1] CompactionRL 则明确联合训练执行与摘要，使用 token 级损失归一化以及跨段优势修正，并在实验中采用每个任务一次 rollout。[2]

这与本文的环境压缩设定需要区分：如果摘要来自同一个待训练策略，摘要生成也是 action，应记录其真实输入、old log-prob 并计算相应损失；恢复后再次出现的摘要文本则是 context。这个区别改变待优化 token 的集合，但不改变“segment 不必成为新的 group 成员”的判断。

本文关于保留 rollout-level GRPO 的方案，是对算法可行性的分析，不是声称作者已经实现或比较了该方案。现有描述也不足以证明，在匹配采样与计算预算后，PPO 一定优于所有保留完整分组的实现。

更严谨的解读应当是：**作者为 compaction 后的分段训练选择了状态价值估计，以适配独立处理的中间上下文，并减少对同任务多次采样的依赖。可变段数、可变段长是其工程背景，不能单独构成 GRPO 不可用的证明。**

评估类似论述时，可以检查四件事：

1. Group 定义在完整任务还是局部上下文？
2. Split 是否改变了样本权重？
3. 每个 token 是否使用真实输入计算概率？
4. 段边界是否被误当成任务终止？

这四点明确后，才能进一步讨论 critic 是否值得其训练成本。

## 参考资料

1. [Z.AI：GLM-5.2: Built for Long-Horizon Tasks](https://huggingface.co/blog/zai-org/glm-52-blog)，RL for Long-Horizon Tasks 部分。
2. [CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents](https://arxiv.org/html/2607.05378v1)，第 4 节及训练设置。
3. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)，GRPO。
4. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。

相关阅读：[OPD 乱象：从监督信号、数据分布与学习算法正本清源](/posts/opd-chaos-supervision-engineering/)。
