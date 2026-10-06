---
title: "从 Figure 2 开始理解 LLM RL 的 Exploration：Outcome-based Exploration 论文阅读"
date: 2026-10-07T00:15:00+08:00
draft: false
tags: ["LLM Reasoning", "Reinforcement Learning", "Exploration", "GRPO", "论文笔记"]
categories: ["论文阅读"]
math: true
summary: "RL 不只是 optimizer，也是一个不断改变采样分布的 data generator。本文从 Figure 2 出发，区分累计 rollout coverage、标准 pass@k、historical coverage 与 current-policy diversity，并把 Historical / Batch Exploration 统一重写为 outcome-level advantage shaping。"
---

> 原文：[Outcome-based Exploration for LLM Reasoning](https://arxiv.org/abs/2509.06941)（arXiv:2509.06941，Yuda Song、Julia Kempe、Remi Munos，2025）

## 一句话概括

RL 能让模型更容易一次答对，却也可能让模型越来越“固执”：已经会的题反复走熟悉路径，不会的题也逐渐失去尝试其他答案的能力。

这篇论文最有意思的地方，不只是给 GRPO 加了一个 exploration bonus，而是换了两个观察角度：

1. **把整个 RL 训练过程看成一个持续变化的 sampling process；**
2. **不在巨大的 token / reasoning-trace space 中数探索，而把 trajectory 投影到更小、更可验证的 outcome space。**

方法本身反而很简单：过去少见的答案多鼓励一点；当前 batch 里重复出现的答案多惩罚一点。

我认为这篇论文真正值得留下的是三件事：

- 一个研究视角：**RL as Sampling**；
- 一个建模抽象：**Trajectory $\rightarrow$ Outcome**；
- 一个工程方法：**Outcome-level Advantage Shaping**。

## 1. Problem：Pass@1 上升，模型却可能更不会“找答案”

先看一个直观过程。面对一道难题，base model 会尝试许多方向，大部分是错的，但偶尔能碰到正确答案。RL 一旦采到正确答案，就会提升这类 trajectory 的概率。

对已经会做的题，这是 exploitation；但同一个参数更新也会影响其他题。如果概率质量不断集中，模型在尚未解决的问题上也可能更少尝试 alternative hypotheses：

$$
\text{exploitation}\uparrow
\qquad\text{while}\qquad
\text{exploration}\downarrow.
$$

这对 test-time scaling 尤其重要。Pass@1 只看一次采样；当我们愿意为一道题采多次、做 reranking 或 search 时，需要当前模型仍保留足够的生成多样性。

> **论文原意：** outcome-reward RL 提高准确率，却伴随 generation diversity degradation；这种退化不仅出现在最终模型的测试集上，也发生在训练过程中，并会从已解决问题传递到未解决问题。
>
> **进一步判断：** “正确答案概率变高”与“候选空间是否仍有宽度”是两个不同目标。只优化前者，不保证模型仍适合作为 test-time search 的 proposal distribution。

## 2. Figure 2：把 RL 训练本身看成 Sampling

通常比较 Base 和 RL，是固定两个 checkpoint，再分别计算标准 pass@$k$。Figure 2 问的是另一个问题：**训练本身每天都在 rollout，那么整段 RL 训练能不能被看成一种 adaptive sampling algorithm？**

![Figure 2：RL training dynamics 与 base model sampling 的累计比较](/images/outcome-exploration/figure-2.png)

> 图源：论文 Figure 2，按原图裁切；论文采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。上排统计“截至目前至少答对过一次”的问题数，下排统计截至目前见过的不同答案数。

### 2.1 $k=nt$ 只是累计 rollout budget

假设每个 epoch 对每道题采 $n$ 条 rollout。训练到第 $t$ 个 epoch 时，每道题一共消耗

$$
\boxed{k=n\times t}
$$

条采样预算。Figure 2 顶部横轴的 $k$ 只是把 epoch 换算成等量的累计 samples，便于和一直从 base model 采样做公平的 budget comparison。

关键在于：两侧的 $k$ 条样本并不来自相同类型的分布。

Base 侧是 fixed-policy sampling：

$$
\tau_1,\ldots,\tau_k
\overset{\mathrm{iid}}{\sim}
\pi_{\mathrm{base}}(\cdot\mid x).
$$

RL 侧则来自一串持续变化的 checkpoint：

$$
\tau_{s,i}\sim\pi_s(\cdot\mid x),
\qquad
s=1,\ldots,t,
\quad i=1,\ldots,n.
$$

因此，RL 曲线在问题 $x$ 上统计的其实是

$$
C_t(x) =
\mathbf 1\!\left[
\exists s\le t,\ i\le n:
r(x,\tau_{s,i})=1
\right],
$$

即：**到当前训练阶段为止，这道题在历史上有没有至少被答对过一次。**

如果记第 $s$ 个 checkpoint 单次答对 $x$ 的概率为 $p_s(x)$，并暂时把各次 rollout 视为条件独立，那么历史累计发现概率是

$$
\boxed{
P_{\mathrm{hist},t}(x) =
1-\prod_{s=1}^{t}\bigl(1-p_s(x)\bigr)^n.
}
$$

这不等于从最终 checkpoint $\pi_t$ 独立采 $nt$ 次：

$$
P_{\pi_t,nt}(x) =
1-\bigl(1-p_t(x)\bigr)^{nt}.
$$

所以 Figure 2 中的 RL 曲线**不是当前模型的标准 pass@$k$ 曲线**。更准确地说，它是跨 checkpoint、跨训练历史的 cumulative coverage；论文说这些量“对应” fixed model 的 pass@$k$ / diff@$k$，但两者不能直接画等号。

### 2.2 Figure 2 真正说明了什么

人话结论是：**RL 前期比 Base 更快找到答案，但越往后发现新可解问题的速度越慢；在相同累计 rollout budget 下，最终甚至不如一直从 Base 采样。**

下排更关键：RL 累计见到的 distinct answers 更少，而且这种差距也出现在“至今从未答对”的问题上。论文把它称为 **transfer of diversity degradation**：在已解决题上集中概率质量的参数更新，迁移到了未解决题，使后者也变窄。

> **论文原意：** Figure 2 衡量 RL training process 的探索效率，并用未解决题上的 diff@$k$ 下降支持 diversity degradation 的跨问题迁移。
>
> **进一步判断：** 这张图最有价值的不是再次证明“RL 后 pass@$k$ 会掉”，而是把 RL 暴露成一个 feedback loop：$\pi_t$ 既是被优化的对象，也是下一轮训练数据的生成器。policy 变窄会让后续 on-policy data 同时变窄。

## 3. Outcome Modeling：不要数 reasoning path，数 outcome

一道数学题可以有几乎无穷多种语言表述、推导顺序和 token trajectory。直接问“模型探索了多少种 reasoning”，既难定义，也难统计。

但可验证推理任务有一个天然的压缩层：许多不同 trajectory 会落到同一个最终答案。定义

$$
x
\xrightarrow{\pi_\theta}
\tau
\xrightarrow{g_x}
a
\xrightarrow{\phi_x}
o,
$$

其中：

- $\tau$ 是完整 rollout（reasoning + final answer）；
- $g_x(\tau)$ 从 rollout 中抽取答案 $a$；
- $\phi_x(a)$ 做等价归一化，例如 $1/2$、$0.5$、$2/4$ 映射到同一 outcome；
- $o$ 是归一化后的 outcome。

于是 policy 在 trajectory space 上的概率质量被聚合成 outcome distribution：

$$
\boxed{
p_\theta(o\mid x) =
\Pr_{\tau\sim\pi_\theta(\cdot\mid x)}
\left[\phi_x(g_x(\tau))=o\right].
}
$$

等价地，

$$
p_\theta(o\mid x) =
\sum_{\tau:\,\phi_x(g_x(\tau))=o}
\pi_\theta(\tau\mid x).
$$

### 3.1 采 $k$ 次，期望能看到多少种答案？

如果真正想问 outcome diversity，可以定义 $k$ 次独立采样中的 expected distinct outcomes：

$$
\boxed{
D_k(x) =
\sum_{o\in\mathcal O_x}
\left[1-\bigl(1-p_\theta(o\mid x)\bigr)^k\right].
}
$$

这个公式并不神秘。对每个 outcome $o$ 定义指示变量 $I_o$：$k$ 次中出现过至少一次则为 1，否则为 0。那么

$$
\Pr(I_o=1) =
1-\bigl(1-p_\theta(o\mid x)\bigr)^k,
$$

而 distinct outcome 数为 $\sum_o I_o$。利用期望的线性性，就得到上面的 $D_k(x)$。

### 3.2 Pass@$k$ 不等于 Diversity

设唯一正确的 outcome 为 $o^\star$，则

$$
\operatorname{Pass@k}(x) =
1-\bigl(1-p_\theta(o^\star\mid x)\bigr)^k.
$$

它和 $D_k$ 有相同的“至少出现一次”结构，但含义不同：

$$
\boxed{
\operatorname{Pass@k} =
\text{correct-outcome coverage}
}
$$

而

$$
\boxed{
D_k =
\text{all-outcome coverage}.
}
$$

两个模型可以有相同的 $p(o^\star\mid x)$，因此拥有相同的 pass@$k$，但错误概率质量的分配完全不同：一个可能集中在单个错误答案上，另一个可能分散到许多 alternative outcomes。Pass@$k$ 是和求解能力相关的 proxy，却不是 outcome diversity 的完整描述。

> **论文原意：** 用最终答案作为 reasoning diversity 的可计算 proxy；实验中即使采样预算很大，每题平均 distinct answers 仍少于 50，因此 outcome space 是 tractable 的。
>
> **进一步判断：** outcome 是 token 与 reward 之间一个很实用的中间抽象层。它牺牲了“同一答案的不同推理路径”这部分信息，却换来了语义等价、可计数和可直接做 gradient shaping 的对象。

## 4. Historical Exploration：对过去少见的答案多给一点梯度

Figure 2 说明 vanilla RL 后期越来越少发现新 outcome。最直接的补救是：**一个答案过去出现得越少，就额外鼓励一次。**

维护问题—答案对的历史访问次数

$$
N_t(x,o) =
\text{截至 step }t\text{，outcome }o\text{ 在问题 }x\text{ 上出现的次数},
$$

论文定义

$$
b_{\mathrm{ucb}}(x,o) =
\min\left\{1,\sqrt{\frac{1}{N_t(x,o)}}\right\}.
$$

### 4.1 它更像 inverse-count novelty，而不是完整 UCB

经典 bandit UCB 通常包含经验回报与置信半径：

$$
\widehat\mu(a) +
c\sqrt{\frac{\log t}{N_t(a)}}.
$$

本文只借用了“访问越少，bonus 越大”的 count-based 形状。更重要的是，数学题 verifier 给出的 reward 基本是 deterministic 的：某个归一化答案一旦被验证为错，再次访问并不会降低多少 reward uncertainty。

所以我更愿意把本文的 $b_{\mathrm{ucb}}$ 称为

$$
\boxed{\text{inverse-count novelty shaping}}
$$

而不是严格意义上的 uncertainty estimation。论文自己也观察到：naive UCB 确实提高了训练集历史 coverage，却没有稳定提高测试性能。**以前没见过，不等于值得反复探索；新鲜的错误答案可以很多。**

### 4.2 从只给正奖励，到同时允许负信号

论文随后引入两种 baseline。

UCB-Mean 使用 leave-one-out batch mean：

$$
A_i^{\mathrm{exp}} =
b_i -
\frac{1}{G-1}\sum_{j\ne i}b_j.
$$

如果 $\bar b=\frac1G\sum_j b_j$，则

$$
A_i^{\mathrm{exp}} =
\frac{G}{G-1}(b_i-\bar b),
$$

本质是 group-relative novelty advantage。

UCB-Con 则减去常数 baseline：

$$
A_i^{\mathrm{exp}} =
b_i-b_0.
$$

例如 $b_0=0.5$ 时，历史访问少于 4 次的 outcome 获得正信号，频繁出现的 outcome 获得负信号。论文还做了两个重要的工程处理：

- 当一个 group 中所有答案都正确、task advantage 全为 0 时，把 exploration bonus 也设为 0，避免探索项反过来惩罚全对 batch；
- mask final-answer tokens，让探索梯度只作用于 reasoning tokens，避免模型通过“随便换最终数字”来 bonus hacking。

实验上，加入 baseline 会稍微牺牲训练历史 coverage，但 UCB-Mean 与 UCB-Con 的测试表现更稳定；UCB-Con 在多数设置上给出更好的 pass@$k$ frontier。

### 4.3 更严谨的重写：它改的是 Advantage，不是 Optimizer

论文的 Eq. (1) / Eq. (2) 把 group-normalized advantage 直接放进一个期望里，省略了 GRPO/PPO 真正连接参数 $\theta$ 的 likelihood ratio（或 log-probability）项。由于组内标准化 advantage 的和本来就接近 0，这种写法如果被当成完整、可微的 optimization objective，会令人困惑。

更清楚的形式是：底层仍然使用普通 GRPO，只额外构造 outcome-level exploration advantage。

先从旧策略采样 $G$ 条 rollout：

$$
\tau_i\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid x),
\qquad i=1,\ldots,G.
$$

由 correctness reward $r_i$ 构造普通 group-relative task advantage：

$$
A_i^{\mathrm{task}} =
\frac{r_i-\bar r}{s_r+\epsilon}.
$$

用 mask $m_{i,t}\in\{0,1\}$ 控制 exploration 只更新 reasoning tokens：

$$
\boxed{
A_{i,t}^{\mathrm{final}} =
A_i^{\mathrm{task}} +
\lambda m_{i,t}A_i^{\mathrm{exp}}.
}
$$

再定义 token-level importance ratio

$$
\rho_{i,t}(\theta) =
\frac{
\pi_\theta(z_{i,t}\mid x,z_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(z_{i,t}\mid x,z_{i,<t})
},
$$

把 $A_{i,t}^{\mathrm{final}}$ 放回正常的 clipped surrogate：

$$
\begin{aligned}
L(\theta) =
\mathbb E\Bigg[
&\frac1G\sum_{i=1}^{G}\frac1{T_i}\sum_{t=1}^{T_i}
\min\Big(
\rho_{i,t}A_{i,t}^{\mathrm{final}},\\
&\operatorname{clip}(\rho_{i,t},1-\epsilon_c,1+\epsilon_c)
A_{i,t}^{\mathrm{final}}
\Big)
-\beta D_{\mathrm{KL}}
\Bigg].
\end{aligned}
$$

这时方法定位就很清楚了：

$$
\boxed{
\text{GRPO optimizer}
\quad+\quad
\text{outcome-level advantage shaping}.
}
$$

> **论文原意：** 在 GRPO correctness signal 上叠加 outcome exploration bonus，并通过 baseline、all-correct gating 与 answer-token masking 改善训练。
>
> **进一步判断：** 论文的 GRPO formalization 过度简写；把方法重写成 advantage construction，比把 Eq. (2) 当作一个全新的 RL objective 更准确。UCB-Con 中的常数 baseline 在理想 policy-gradient expectation 下梯度为 0，它的实际作用更应理解为 finite-batch、clipping 与 gating 条件下的 advantage shaping。

## 5. Batch Exploration：去过很多地方，不等于现在还能去

Historical exploration 回答的是：**训练至今去过哪些 outcome？** 但训练过程中曾见过 100 种答案，不代表最终 policy 现在仍能生成它们。

Batch Exploration 直接处理当前分布：同一个 batch 中，和别人“撞答案”越多，扣分越多。论文定义

$$
\boxed{
A_i^{\mathrm{batch}} =
-\frac1G
\sum_{j\ne i}
\mathbf 1[o_i=o_j].
}
$$

### 5.1 它其实是 outcome collision penalty

若 $o_i,o_j$ 独立采自当前 outcome distribution $p(o\mid x)$，两次采样相同的概率是

$$
\Pr(o_i=o_j) =
\sum_o p(o\mid x)^2.
$$

因此

$$
\mathbb E[A_i^{\mathrm{batch}}] =
-\frac{G-1}{G}
\sum_o p(o\mid x)^2.
$$

$\sum_o p(o)^2$ 是 collision probability；Rényi-2 entropy 为

$$
H_2(p) =
-\log\sum_o p(o)^2.
$$

所以，最大化 batch bonus 等价于压低 outcome collision probability，并与增大 Rényi-2 entropy 单调一致。严格来说，它没有显式的 $\log$，又受到有限 group sampling、GRPO clipping 和 task advantage 的共同影响，因此更准确的表述是：

$$
\boxed{
\text{Batch} =
\text{outcome-level collision penalty}
\approx
\text{Rényi-2 entropy regularization}.
}
$$

### 5.2 Historical Coverage 与 Current Diversity 是两个状态变量

论文实验恰好展示了这种区别：Historical 方法在“累计解决多少题、累计见过多少答案”上更强；Batch 在最终 checkpoint 的 large-$k$ pass@$k$ 和当前 batch diversity 上更稳。

以 batch size 8 的 distinct answers 为例：

| 方法 | 已解决问题 | 未解决问题 | 全部问题 |
| --- | ---: | ---: | ---: |
| GRPO | 2.279 | 4.805 | 2.883 |
| UCB-Con | 2.272 | 4.855 | 2.926 |
| Batch | 2.284 | **5.390** | **3.230** |

Batch 的主要增益出现在**未解决问题**：它不是让已经会做的题无限制造花样，而是在不会做的题上继续保留 alternative outcomes。

可以把两种 exploration 压缩成一句话：

$$
\boxed{
\underbrace{\text{Where have I ever been?}}_{\text{historical coverage}}
\neq
\underbrace{\text{Where can I still go?}}_{\text{current-policy diversity}}.
}
$$

> **论文原意：** Historical 与 Batch 不是替代关系。前者更擅长扩大训练历史覆盖，后者直接促进当前 batch 的多样性，并在训练后期更好地保留 large-$k$ 能力。
>
> **进一步判断：** Batch 是全文最干净的 engineering trick。它不需要维护跨训练历史的 count table，直接对当前 policy 的 outcome collisions 施加负反馈；代价是它可能每个 epoch 都循环同一小组互不相同的答案，因此不保证真正扩大历史覆盖。

## 6. What I Learned：我真正想从这篇论文留下什么

### 6.1 RL 不只是 optimizer，也是 data generator

On-policy RL 的闭环是

$$
\pi_t
\longrightarrow
\text{rollout data}
\longrightarrow
\text{update}
\longrightarrow
\pi_{t+1}.
$$

policy collapse 不只是“当前输出变单一”，还意味着未来用来训练自己的数据同时变窄。Figure 2 的价值，就是让这个动态后果变得可测量。

### 6.2 不要再用一个 diversity 词混合两件事

至少要区分：

- **historical coverage**：整段训练历史曾访问过多少题、多少 outcome；
- **current-policy diversity**：固定当前 checkpoint，现在仍能采出多少 outcome；
- **correct-outcome coverage**：标准 pass@$k$ 关心的正确答案是否至少出现一次。

Figure 2、Historical UCB 与 Batch Exploration 分别对应不同量。把它们混成“diversity”会直接导致对实验的误读。

### 6.3 Outcome space 是一个可迁移的中间抽象层

$$
\text{Token}
\rightarrow
\text{Trajectory}
\rightarrow
\boxed{\text{Outcome}}
\rightarrow
\text{Reward}.
$$

许多 RL 方法直接从 token entropy 跳到最终 reward；这篇论文提醒我，outcome 这一层可能同时更有语义、更可统计，也更适合做 advantage shaping。

它的边界也很清楚：必须能可靠抽取、归一化并比较 outcome。数学题适合；开放式写作、多轮交互或“同一个答案、不同推理质量”的任务则需要更丰富的 outcome representation。

## 结语

如果只把这篇论文记成“给 GRPO 加 UCB”，我认为反而错过了最有价值的部分。更准确的阅读顺序应该是：

$$
\boxed{
\begin{gathered}
\text{Problem}
\rightarrow
\text{Figure 2 / RL as Sampling}\\
\rightarrow
\text{Outcome Modeling}
\rightarrow
\text{Historical Exploration}\\
\rightarrow
\text{Batch Exploration}
\rightarrow
\text{What I Learned}.
\end{gathered}
}
$$

方法层面，它可以统一理解为 outcome-level advantage shaping；研究视角上，它把一个更一般的问题说清楚了：**当 policy 同时负责生成自己的下一批训练数据时，我们不仅要问它现在答得多准，还要问它的未来数据分布正在变宽，还是变窄。**
