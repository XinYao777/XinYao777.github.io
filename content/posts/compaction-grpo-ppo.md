---
title: "从超长轨迹到 Compaction：GRPO、log-prob 与 segment 聚合"
date: 2026-10-20T21:40:00+08:00
lastmod: 2026-10-21
draft: false
tags: ["强化学习", "GRPO", "PPO", "Agent", "Compaction"]
categories: ["AI 技术思考"]
math: true
summary: "先比较未经压缩与经过压缩的超长轨迹，明确每一段的真实 input，再讨论 GRPO 的直接扩展、显存约束、log-prob 的分段计算，以及 split 后真正需要处理的 segment loss 聚合。"
---

GLM-5.2 的官方介绍指出：长任务经过 compaction 后，同一个 prompt 下的不同 rollout 会产生数量、长度都不同的 trainable traces，因此采用 critic-based PPO。[1]

要理解这段话，应该先沿着数据实际发生的顺序分析：

> 超长任务如何执行 → compaction 改变了哪些 input → 如何计算正确的 log-prob → 如何聚合各段的训练损失。

这里先固定设定：压缩模块属于环境，摘要作为模型输入，不计入 actor loss。最后再区分摘要也参与训练的情形。

## 一、未经压缩的超长轨迹：每一段的 input 不断累积

设初始任务为 $x$。模型生成一段动作 $A_1$，环境返回观察 $O_1$，随后继续生成 $A_2$，依次执行到任务结束。

为便于说明，把每一轮 assistant 输出视为一段。实际也可以将连续多轮交互归为一段。

| 输出段 | 模型开始生成该段时的 input |
|---|---|
| 第一段 A₁ | x |
| 第二段 A₂ | x、A₁、O₁ |
| 第三段 A₃ | x、A₁、O₁、A₂、O₂ |

用 $H_k$ 表示生成第 $k$ 段之前的上下文，则：

$$
H_1=x,
\qquad
H_{k+1}=H_k\oplus A_k\oplus O_k.
$$

其中 $\oplus$ 表示按实际模板拼接。第 $k$ 段的第 $t$ 个 token 使用的条件分布是：

$$
\pi_\theta(a_{k,t}\mid H_k,a_{k,\lt t}).
$$

这里的 input 是持续追加的前缀。对于使用普通因果注意力、实际输入保持追加关系的调用，可以将完整交互组织成一条训练序列：

~~~text
x → A1 → O1 → A2 → O2 → A3
~~~

环境观察参与后续 token 的条件计算，但只有 assistant 生成的位置计算 actor loss。若框架还会改变模板、插入额外消息或特殊 token，则仍需严格复现实际调用。

任务执行越久，后半段 input 通常越长。这首先是上下文与计算资源的问题，还不是 GRPO 的分组问题。

## 二、经过压缩的超长轨迹：任务继续，input 被替换

假设执行完第二轮后发生一次 compaction。环境将历史压缩为摘要：

$$
m=M(x,A_1,O_1,A_2,O_2).
$$

为简洁起见，假设恢复后的 input 为：

$$
H_3^{\mathrm c}=x\oplus m.
$$

实际系统可能保留近期交互、系统指令和恢复模板，都应计入这个真实 input。

对比两种执行方式：

| 输出段 | 未压缩时的 input | 压缩时的 input |
|---|---|---|
| 第一段 A₁ | x | x |
| 第二段 A₂ | x、A₁、O₁ | x、A₁、O₁ |
| 第三段 A₃ | x、A₁、O₁、A₂、O₂ | x、m |
| 第四段 A₄ | 全部历史继续追加 | x、m、A₃、O₃ |

这张表是整个问题的起点：第三段不再条件于全部原始历史，而是条件于摘要与保留内容。

压缩没有让任务结束。文件修改、浏览器页面和工具状态仍然沿着这次任务继续演化；变化的是模型能够读取的历史表示。

所以，compaction 缩短的是工作上下文，并不必然缩短完成任务所需的动作序列，也没有自动缩短早期动作到最终奖励的因果距离。

## 三、超长轨迹的 GRPO：一个直接扩展

暂时不考虑压缩。对同一个任务 $x$，采样 $G$ 次完整执行：

$$
\tau_i\sim\pi_{\mathrm{old}}(\cdot\mid x),
\qquad R_i=R(\tau_i).
$$

采用结果监督的常见 GRPO 优势：[3]

$$
\widehat A_i=
\frac{R_i-\overline R}{\sigma_R+\epsilon}.
$$

然后让该次完整执行的所有 action token 使用这个优势：

$$
\widehat A_{i,k,t}=\widehat A_i.
$$

这就是从短回答到长任务的一种直接扩展：原来一个回答的 token 共享优势，现在整次多轮执行的 token 共享优势。Group 仍是同一个初始任务下的完整 rollout，而不是不同 rollout 的“第 k 段”。

定义 token 级 probability ratio：

$$
\rho_{i,k,t}(\theta)=
\frac{\pi_\theta(a_{i,k,t}\mid H_{i,k},a_{i,k,\lt t})}
{\pi_{\mathrm{old}}(a_{i,k,t}\mid H_{i,k},a_{i,k,\lt t})}.
$$

相应的 clipped surrogate 为：

$$
\ell_{i,k,t}=
\min\left(
\rho_{i,k,t}\widehat A_i,\,
\operatorname{clip}(\rho_{i,k,t},1-\eta,1+\eta)\widehat A_i
\right).
$$

本文用最大化方向的 $L$ 表示 surrogate；实际最小化的 policy loss 为其负值，并可另加 KL 等项。

理论上，这个扩展可以定义并计算；它没有要求 rollout 等长，也没有要求各次执行的中间步骤语义对齐。

其限制是信用分配粗糙：成功执行中的有效动作、无效尝试和冗余推理，都得到同一个结果优势。任务越长，这个问题可能越突出。但“粗糙”不等于“数学上不可用”。

严格来说，最终奖励策略梯度、GRPO 的组归一化优势、clipping 与长度加权是不同层次；这里讨论的是 GRPO 扩展的可行性，不声称其更新是原始期望奖励梯度的无偏估计。

### 训练时是否容易 OOM？

是，有这种风险，尤其是直接将全部历史作为一个超长序列进行带梯度的前向与反向时。

需要区分 rollout 和训练：

| 阶段 | 主要资源压力 |
|---|---|
| 自回归 rollout | KV cache、并发请求及上下文长度 |
| 训练 actor | 激活、反向计算、logits 或 log-prob 计算的中间张量 |
| 多 rollout 的 batch | 长度差异、padding 和设备间负载 |

若序列长度为 $L$、词表大小为 $V$，直接保存 full-vocabulary logits 的张量规模为 $O(LV)$。普通 dense attention 若显式保存注意力矩阵，相关存储为 $O(L^2)$；FlashAttention 等实现避免显式物化完整矩阵，因此不能笼统说所有实现的显存都按平方增长。稠密注意力的计算成本与其他训练激活仍会随长度增加。

Actor ratio 最终需要实际采样 token 的 log-prob，不必永久保存所有 logits；不过 softmax 归一化及反向仍有计算和存储成本，具体取决于分块、融合等实现。

因此，超长轨迹上运行 GRPO 的一个直接障碍是资源开销。它不是 GRPO 独有的问题：PPO actor 也需要相同的条件概率，critic 还会引入额外成本。

## 四、压缩后如何训练：关键是对正确的 input 计算 log-prob

现在考虑前面已经压缩的执行。对第三段的 token，需要计算：

$$
\log\pi_\theta(a_{3,t}\mid x,m,a_{3,\lt t}).
$$

不能用压缩前全部历史代替它，否则分子的条件分布就与真实 rollout 不一致。

通常在 rollout 时保存：

$$
q^{\mathrm{old}}_{k,t} = \log\pi_{\mathrm{old}}(a_{k,t}\mid H_k^{\mathrm{actual}},a_{k,\lt t}),
$$

训练时重新计算：

$$
q^\theta_{k,t} = \log\pi_\theta(a_{k,t}\mid H_k^{\mathrm{actual}},a_{k,\lt t}),
$$

再得到：

$$
\rho_{k,t}=\exp(q^\theta_{k,t}-q^{\mathrm{old}}_{k,t}).
$$

$H_k^{\mathrm{actual}}$ 就是该段生成时真实记录下来的 input。没有必要重新生成摘要，也不应该让后半段重新看到已经删除的历史。

### 两种可行实现

第一种是专门组织输入与 attention mask。将压缩前分支和压缩后分支放入同一计算载体，隔离后半段不可见的历史，并设置匹配的 position IDs 和 loss mask。

仅屏蔽第三段到原历史的直接注意力还不够：如果摘要位置的 hidden states 已读取原历史，后续 token 仍可能通过这些表示间接访问被删除的信息。应保证恢复后整个分支的计算都只依赖真实恢复后的输入。

第二种是分段计算 log-prob：

~~~text
前段训练输入：
    [x, A1, O1, A2, O2]
    action loss：仅 A1、A2

后段训练输入：
    [x, m, A3, O3, A4]
    action loss：仅 A3、A4
~~~

每段使用普通的因果注意力即可。它们可以分别前向，也可以放入同一 batch，或使用支持隔离注意力的 packing。

这就是 split 的一种简单实现：在上下文被替换的位置，结束一条训练记录，并用实际压缩后的 input 开始下一条。

物理分段能让单次处理的序列更短，但不保证训练总显存一定降低：若同时保留所有段的计算图再统一 backward，激活仍会累积。要降低峰值，通常还需按目标权重逐段 backward 或使用 microbatch，并在正确时机统一 optimizer step。

## 五、分段计算之后，仍然是一条完整超长轨迹

Split 后无需把每一段当成独立 episode。保留完整 rollout 的标识、顺序以及原始任务归属即可：

~~~text
rollout_id: 42
initial_prompt_id: task_x
final_reward: 1

segment_1:
    actual_input: ...
    action_tokens: ...
    old_log_probs: ...

segment_2:
    actual_input: ...
    action_tokens: ...
    old_log_probs: ...
~~~

各段计算得到的 action log-prob，可以按原有执行顺序逻辑拼接：

$$
\mathbf q_i^\theta = \operatorname{concat}
(\mathbf q_{i,1}^\theta,\ldots,\mathbf q_{i,K_i}^\theta).
$$

Old log-prob、ratio、action mask 也采用相同索引关系。

这里拼接的是概率记录及其索引，不是重新将原始历史拼成一个不加区分的 causal-attention 输入。Actor 更新仍需保留正确的梯度路径；“逻辑拼接”也不要求把全部计算图同时留在显存中。

Group 统计依然从 $G$ 次完整执行的最终奖励计算：

$$
\widehat A_i = \operatorname{GroupNorm}(R_1,\ldots,R_G)_i.
$$

然后将 $\widehat A_i$ 用于第 $i$ 次完整执行中的各段。

因此，各段 input 不同，不妨碍它们属于同一次 rollout；不同 rollout 的中间 input 不同，也不妨碍它们在同一初始任务下比较最终结果。

## 六、Split 后真正需要明确的是 segment 聚合

此时，同一个 prompt 下的不同 rollout 可能产生不同数量的 segment，每个 segment 的长度也不同。

设第 $i$ 条 rollout 有 $K_i$ 段，第 $k$ 段有 $n_{i,k}$ 个待优化 action token：

$$
T_i=\sum_{k=1}^{K_i}n_{i,k}.
$$

这里先假设完整 rollout 层面的奖励与优势已经正确计算，不讨论把 segment 错误当成新 group 成员的情形。

真正需要选择的是：各段的 token loss 如何汇总到一次更新？

### 1. 每段先平均，再对所有段平均

$$
L_{\mathrm{segment}} = \frac{1}{\sum_iK_i}
\sum_{i=1}^{G}\sum_{k=1}^{K_i}
\frac{1}{n_{i,k}}\sum_{t=1}^{n_{i,k}}\ell_{i,k,t}.
$$

这种方式令每个 segment 等权。因此：

- 段数更多的 rollout 在整体目标中占据更多份额。
- 较短 segment 中，每个 token 的系数更大。
- 对同一批 token 改变切段位置，目标一般也会改变。

这与 reward 是否重复计算无关，是 loss reduction 的选择造成的。

例如，两次执行都有 1000 个 action token：

| 完整 rollout | segment 划分 | 在 segment 平均中的总系数 | 每个 token 的系数 |
|---|---|---:|---:|
| 第一次 | 1 段，每段 1000 token | 1/5 | 1/5000 |
| 第二次 | 4 段，每段 250 token | 4/5 | 1/1250 |

虽然两次完整执行一样长，仅因切段不同，第二次执行在加权结构中占比就是第一次的四倍。这里比较的是目标中的权重系数，不是说实际梯度范数一定相差四倍。

### 2. 先恢复完整 rollout 的平均，再对 rollout 平均

$$
L_{\mathrm{rollout}} = \frac1G\sum_{i=1}^{G}
\frac1{T_i}
\sum_{k=1}^{K_i}\sum_{t=1}^{n_{i,k}}\ell_{i,k,t}.
$$

若已计算每段均值 $\overline\ell_{i,k}$，等价写法是：

$$
L_{\mathrm{rollout}} = \frac1G\sum_i\sum_k
\frac{n_{i,k}}{T_i}\overline\ell_{i,k}.
$$

每次完整执行等权，段只是计算容器。

### 3. 全 batch 的 action token 统一平均

$$
L_{\mathrm{token}} = \frac{
\sum_i\sum_k\sum_t\ell_{i,k,t}
}{
\sum_iT_i
}.
$$

每个 action token 等权，较长完整执行占据更大份额。在上面的等长例子中，这种方式和 rollout 平均一致；当完整执行长度不同时，两者一般不同。

| 聚合方式 | 等权单位 | 改变人为切段方式会否改变目标 |
|---|---|---|
| Segment 平均 | segment | 一般会 |
| Rollout 内平均，再对 rollout 平均 | 完整 rollout | 不会 |
| 全 batch token 平均 | action token | 不会 |

后两项的“不变”以 token、真实上下文、优势和聚合范围保持不变为前提。若每段独立 optimizer step，或更改逐批归一化与梯度裁剪，实际优化过程仍可能不同。

实践中，即使采用 token 平均，也要避免将不同 token 数的 microbatch mean 不加权地相加，否则又会在实现层面引入额外权重差异。

Token 平均消除的是人为切段带来的系数变化，不代表每条 rollout 等权，也不自动解决所有长度偏差。

## 七、这是否要求从 GRPO 改成 PPO？

不要求。沿用完整 rollout 分组、正确计算各段 log-prob，并明确 loss 聚合方式，就可以保留 GRPO。

未解决的是：所有 action token 仍共享完整执行的结果优势。它没有因 split 而获得更细的信用分配。

引入 critic 的合理动机是，使用当前上下文预测后续回报，构造状态条件化的优势：

$$
V_\phi(c_t)\approx\mathbb E[R_{\mathrm{future}}\mid c_t].
$$

同时，critic-based PPO 不要求同一个任务必须采样多次完整执行，可能降低依赖组比较的采样成本；但要承担价值模型训练的额外成本与估计误差。[4]

对 PPO 而言，split 后仍需要明确如何跨段处理回报和优势。上下文被替换，不意味着任务终止；GAE 不能未经处理就在段边界清零。具体的跨段递推或 bootstrap 必须遵循所定义的时间尺度与后续状态。

因此，下面两个判断应分开：

> 分段后，按 token 聚合损失比简单 segment 平均更符合某些训练目标。

> 使用 critic 估计优势，比整条 rollout 共享一个结果优势更有效。

前者是 loss 权重问题，后者是优势估计问题。两者可以组合，但前者并不必然推出后者；token 级 loss 聚合也可以用于 GRPO。

## 八、回到原文：一个实现选择，不能读成必要定理

官方介绍报告了在这一训练背景下选择 critic-based PPO。[1] CompactionRL 进一步采用执行与摘要联合训练、token 级 loss 归一化和跨段优势修正，实验中每个任务采样一次 rollout。[2]

需要特别区分：在本文设定中，摘要来自环境，不计入 actor loss；若摘要由同一个待训练策略生成并优化，那么摘要生成也是 action，需要记录相应 input 与 old log-prob。这个差别改变待优化 token 的集合，不改变“保留完整 rollout 分组”的可行性。

本文给出的 GRPO 组织方式是可行性分析，不是对作者实现的复述，也不是相同预算下的性能结论。

更准确的理解是：未经压缩的超长轨迹已经可以直接扩展 GRPO，但面临信用分配与资源压力；compaction 改变了各段 input，需要正确计算 log-prob；split 是一种简单实现；split 后要明确 segment 如何聚合。是否进一步采用 critic-based PPO，是另一个需要结合采样预算、价值估计质量和实验收益判断的问题。

## 参考资料

1. [Z.AI：GLM-5.2: Built for Long-Horizon Tasks](https://huggingface.co/blog/zai-org/glm-52-blog)，RL for Long-Horizon Tasks 部分。
2. [CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents](https://arxiv.org/html/2607.05378v1)，第 4 节与训练设置。
3. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)。
4. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。

相关阅读：[OPD 乱象：从监督信号、数据分布与学习算法正本清源](/posts/opd-chaos-supervision-engineering/)。
