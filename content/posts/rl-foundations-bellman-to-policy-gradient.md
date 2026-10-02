---
title: "强化学习基础全梳理：从 Bellman 方程到价值估计与策略梯度"
date: 2026-09-30T18:00:00+08:00
draft: false
tags: ["强化学习", "Bellman 方程", "Policy Gradient", "Actor-Critic", "笔记"]
categories: ["AI 技术思考"]
math: true
ShowToc: true
summary: "从状态、动作、奖励与回报出发，完整推导 V/Q、Bellman 期望与最优方程，串起动态规划、MC、TD、SARSA、Q-learning、策略梯度和 Actor-Critic，并解释 target、bootstrap、步长与 on/off-policy 的工程含义。"
---

强化学习的难点，往往不在某一个公式，而在公式之间的连接：为什么回报的定义会导出 Bellman 方程？为什么价值估计用“目标减去估计值”更新？估计出状态价值后，又如何真正改变行动？

本文沿着一条主线展开：**交互产生轨迹，轨迹定义回报，回报定义价值，Bellman 方程刻画价值的自洽关系，算法估计价值或直接优化策略，改进后的策略再产生新数据。**

为了让数学边界清楚，默认讨论有限状态、有限动作、奖励有界的 MDP。持续任务取 $0\le\gamma<1$；回合任务可以取 $\gamma=1$，但需保证相关回报与期望存在。有限时域任务若要求使用平稳形式，应把剩余时间纳入状态。连续空间的求和可替换为积分，但动作最大值的存在和求解还需要额外条件。

## 1. 从一次交互理解强化学习

一次交互写成：

$$
S_t\xrightarrow{A_t}(R_{t+1},S_{t+1}).
$$

智能体在时刻 $t$ 观察状态 $S_t$，选择动作 $A_t$，环境随后返回奖励 $R_{t+1}$ 和下一状态 $S_{t+1}$。奖励下标是 $t+1$，因为它是执行本次动作之后收到的反馈。

与通常把训练数据视为固定的监督学习不同，强化学习中的动作会改变下一状态，进而改变未来看到的数据和获得的奖励。当前动作的好坏，不能只由眼前奖励决定。

例如，机器人可以直接穿过拥挤走廊，也可以绕路。绕路当下耗时更多，却可能避免碰撞；只有把后续影响算进去，才知道哪个动作更好。

### 1.1 大写、小写、真值和估计值

| 符号 | 含义 |
| --- | --- |
| $S_t,A_t,R_{t+1}$ | 随机变量 |
| $s,a,r$ | 随机变量的具体取值 |
| $\pi$ | 策略，即选择动作的规则 |
| $G_t$ | 从时刻 $t$ 开始的样本回报 |
| $v_\pi,q_\pi$ | 给定策略的真实价值函数 |
| $v_*,q_*$ | 最优价值函数 |
| $V,Q$ | 算法当前维护的价值估计 |
| $U_t$ | 本次更新使用的目标，target |
| $\alpha,\beta$ | 学习率或步长 |
| $\theta,\phi$ | 策略和价值函数的参数 |

这里的大小写是约定，不是所有论文都统一遵守的规则。阅读时首先检查作者定义。特别是 $V(s)$ 没有时间下标，并不代表它是真值；它常常只是省略了迭代下标的估计值。

### 1.2 Policy 不是整个 agent

随机策略定义为：

$$
\pi(a\mid s)=\Pr(A_t=a\mid S_t=s),\qquad
\sum_a\pi(a\mid s)=1.
$$

确定性策略可以写成 $a=\mu(s)$。用 $\mu$ 有助于区分“动作”与“动作概率分布”。

策略只是智能体选择动作的部分。一个完整 agent 还可以包含价值网络、世界模型、经验回放缓存和优化器。

```text
Agent
├── policy：决定怎么行动
├── value / Q：评估长期后果
├── world model：预测环境，可选
├── replay buffer：保存经验，可选
└── optimizer：更新参数
```

## 2. MDP 与轨迹分布：数据从哪里来

使用联合转移核时，一个 MDP 可以记为：

$$
\mathcal M=(\mathcal S,\mathcal A,p,\gamma,d_0),
$$

其中 $d_0$ 是初始状态分布，环境模型为：

$$
p(s',r\mid s,a)
=\Pr(S_{t+1}=s',R_{t+1}=r\mid S_t=s,A_t=a).
$$

奖励已包含在联合转移核中，无须再额外指定独立奖励函数。也可以等价地用状态转移概率与期望奖励表达：

$$
P(s'\mid s,a)=\sum_r p(s',r\mid s,a),
\qquad
\bar r(s,a)=\sum_{s',r}p(s',r\mid s,a)r.
$$

马尔可夫性表示：给定当前状态和动作，下一步的分布不再依赖更早历史。它不意味着真实世界没有记忆，而是要求“状态”包含预测未来所需的信息。如果只观察到局部信息，就可能需要历史、循环网络或信念状态。

长度为 $T$ 的轨迹写成：

$$
\tau=(S_0,A_0,R_1,S_1,\ldots,A_{T-1},R_T,S_T).
$$

对于固定长度的轨迹，其概率为：

$$
p_\pi(\tau)
=d_0(S_0)\prod_{t=0}^{T-1}
\pi(A_t\mid S_t)
p(S_{t+1},R_{t+1}\mid S_t,A_t).
$$

这个式子揭示了一个工程事实：**训练数据分布依赖策略。** 策略改变之后，访问哪些状态、哪些动作经常出现、哪些奖励能被观察到，都会改变。

## 3. Reward、return 与最终优化目标

奖励是一条反馈，回报是多条未来奖励的累计：

$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}
=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
$$

回合在 $T$ 终止时：

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1},\qquad G_T=0.
$$

$\gamma=0$ 表示只看下一步奖励；$\gamma$ 越接近 1，远期奖励的权重越大。持续任务中，有界奖励和 $\gamma<1$ 还保证折扣回报绝对有界。

最重要的递归关系来自直接拆出第一项：

$$
\boxed{G_t=R_{t+1}+\gamma G_{t+1}.}
$$

给定初始状态分布，优化目标是：

$$
J(\pi)=\mathbb E_{\tau\sim p_\pi}[G_0]
=\mathbb E_{S_0\sim d_0}[v_\pi(S_0)].
$$

这里 $G_t$ 是一条轨迹上的随机回报，$J$ 是对轨迹分布取期望后的策略表现。一次高分不等于策略的期望表现提高。

## 4. 状态价值与动作价值：两个观察位置

### 4.1 状态价值：从状态开始，之后都按策略行动

$$
v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s].
$$

它回答：处在状态 $s$，随后一直使用策略 $\pi$，平均能获得多少长期回报？同一个状态，在不同策略下可以有完全不同的价值。

### 4.2 动作价值：第一步固定动作，后续再按策略行动

$$
q_\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

它回答：先在状态 $s$ 做动作 $a$，之后使用 $\pi$，平均回报是多少？即使 $\pi(a\mid s)=0$，也可以通过“第一步强制执行 $a$”的干预式定义解释该动作价值，不必依赖零概率事件上的条件期望。

### 4.3 用动作价值表示状态价值

当前动作由策略随机产生，因此：

$$
\boxed{v_\pi(s)=\sum_a\pi(a\mid s)q_\pi(s,a).}
$$

等价的期望形式是：

$$
v_\pi(s)=\mathbb E_{A\sim\pi(\cdot\mid s)}[q_\pi(s,A)].
$$

如果两个动作价值分别为 10 和 0，策略各以一半概率选择，则状态价值为 5。只知道这个平均值，无法恢复两个动作各自的价值。

### 4.4 用状态价值表示动作价值

固定第一步动作后，随机性来自环境响应：

$$
\boxed{
q_\pi(s,a)
=\mathbb E_{(S',R)\sim p(\cdot,\cdot\mid s,a)}
[R+\gamma v_\pi(S')].
}
$$

完全展开为：

$$
q_\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)
[r+\gamma v_\pi(s')].
$$

这两条相互表示是 Bellman 方程的骨架：**状态节点对动作求平均，动作之后对环境结果求平均。**

## 5. Bellman 期望方程：固定策略下的自洽关系

### 5.1 从回报递归推导

把回报递归代入状态价值定义：

$$
\begin{aligned}
v_\pi(s)
&=\mathbb E_\pi[G_t\mid S_t=s]\\
&=\mathbb E_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s]\\
&=\mathbb E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})\mid S_t=s].
\end{aligned}
$$

最后一步使用条件期望的塔式法则：给定下一状态，后续按同一策略行动的期望回报就是该状态的价值。这里同时依赖状态的马尔可夫性和策略的相应假设。

### 5.2 状态价值的三种等价写法

**一层期望**将所有随机性合并：

$$
v_\pi(s)=\mathbb E_\pi
[R_{t+1}+\gamma v_\pi(S_{t+1})\mid S_t=s].
$$

**两层期望**明确“先选择动作，再由环境响应”：

$$
v_\pi(s)
=\mathbb E_{A\sim\pi(\cdot\mid s)}
\left[
\mathbb E_{(S',R)\sim p(\cdot,\cdot\mid s,A)}
[R+\gamma v_\pi(S')]
\right].
$$

**完全展开**适合查表和实现动态规划：

$$
v_\pi(s)=\sum_a\pi(a\mid s)
\sum_{s',r}p(s',r\mid s,a)
[r+\gamma v_\pi(s')].
$$

三种写法是同一个关系，不是三种算法。

### 5.3 动作价值也有三种写法

一层期望：

$$
q_\pi(s,a)=\mathbb E_\pi
[R_{t+1}+\gamma q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a].
$$

两层期望：

$$
q_\pi(s,a)
=\mathbb E_{(S',R)\sim p(\cdot,\cdot\mid s,a)}
\left[R+\gamma
\mathbb E_{A'\sim\pi(\cdot\mid S')}
[q_\pi(S',A')]
\right].
$$

完全展开：

$$
q_\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma\sum_{a'}\pi(a'\mid s')q_\pi(s',a')\right].
$$

注意期望的顺序：$v_\pi$ 从“尚未选动作”开始，$q_\pi$ 从“本次动作已确定”开始。因此 $q_\pi$ 先平均环境结果，再平均下一步动作。

### 5.4 期望没有下标，是不是没有指定分布？

不是。例如：

$$
q_\pi(s,a)=\mathbb E[R_{t+1}+\gamma v_\pi(S_{t+1})\mid S_t=s,A_t=a]
$$

在 MDP 上下文中，条件已经确定环境分布 $p(\cdot,\cdot\mid s,a)$。后续策略的影响则包含在 $v_\pi$ 中。作者可以省略期望下标，但读者应能回答：**究竟哪些量随机，它们由哪个分布生成？**

## 6. 最优价值与 Bellman 最优方程

定义：

$$
v_*(s)=\max_\pi v_\pi(s),\qquad
q_*(s,a)=\max_\pi q_\pi(s,a).
$$

在有限折扣 MDP 中，存在一个平稳确定性最优策略，能同时在所有状态达到最优价值。

### 6.1 最优状态价值与动作价值的相互表示

$$
\boxed{v_*(s)=\max_a q_*(s,a).}
$$

$$
\boxed{
q_*(s,a)=\sum_{s',r}p(s',r\mid s,a)
[r+\gamma v_*(s')].
}
$$

代入即可得到状态版本：

$$
v_*(s)=\max_a\sum_{s',r}p(s',r\mid s,a)
[r+\gamma v_*(s')].
$$

动作版本则是：

$$
q_*(s,a)=\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma\max_{a'}q_*(s',a')\right].
$$

相应的期望形式为：

$$
v_*(s)=\max_a\mathbb E_{(S',R)\sim p(\cdot,\cdot\mid s,a)}
[R+\gamma v_*(S')],
$$

$$
q_*(s,a)=\mathbb E_{(S',R)\sim p(\cdot,\cdot\mid s,a)}
\left[R+\gamma\max_{a'}q_*(S',a')\right].
$$

### 6.2 为什么对完整策略取最大，变成对一个动作取最大？

定义一步前瞻分数：

$$
B(s,a)=\sum_{s',r}p(s',r\mid s,a)[r+\gamma v_*(s')].
$$

对任意策略，后续价值不超过最优价值，所以：

$$
v_\pi(s)\le\sum_a\pi(a\mid s)B(s,a)\le\max_a B(s,a).
$$

因此 $v_*(s)\le\max_a B(s,a)$。反过来，第一步执行使 $B$ 最大的动作，后续使用一个在所有状态都最优的策略，就能达到 $\max_a B(s,a)$。这个“先执行一个动作、之后继续最优”的策略也属于允许比较的策略集合，于是反向不等式成立。

两边合起来得到 Bellman 最优方程。它不是把 $\max_\pi$ 直接替换成 $\max_a$，而是把完整决策拆为“当前一步”和“未来全部步骤”；未来最优性已经存进 $v_*(s')$。

这也解释了一步贪心为什么不一定短视：这里最大化的是“即时奖励加全部未来价值”，不是只最大化即时奖励。

### 6.3 max 和期望的位置不能随意交换

当前动作要在看到环境随机结果之前选，所以 $v_*$ 中是先对各动作计算期望，再取最大。在 $q_*$ 中，下一动作可以在观察到下一状态之后选择，所以 $\max_{a'}$ 位于环境期望内部。

一般来说：

$$
\max_a\mathbb E[X_a]\le\mathbb E[\max_a X_a].
$$

右边相当于先看到随机结果再挑动作，可能隐含决策时并不存在的信息。

## 7. Bellman 方程如何变成算法

方程描述真值应满足的条件，算法则用当前估计反复构造新的估计。

定义策略 Bellman 算子：

$$
(T_\pi V)(s)=\sum_a\pi(a\mid s)
\sum_{s',r}p(s',r\mid s,a)[r+\gamma V(s')].
$$

定义最优 Bellman 算子：

$$
(T_*V)(s)=\max_a\sum_{s',r}p(s',r\mid s,a)
[r+\gamma V(s')].
$$

真实价值是固定点：

$$
v_\pi=T_\pi v_\pi,\qquad v_*=T_*v_*.
$$

有限折扣 MDP 中，两者在最大范数下都是 $\gamma$ 压缩映射。例如：

$$
\|T_\pi V-T_\pi W\|_\infty
\le\gamma\|V-W\|_\infty.
$$

因此精确的表格迭代会向唯一固定点收敛。这解释了反复 backup 的依据；这个保证不能直接推广到任意神经网络、离策略数据和任意优化器的组合。

### 7.1 Policy evaluation：固定策略，评估它

$$
V_{k+1}=T_\pi V_k.
$$

这里策略不变，目标是 $v_\pi$。已知模型时，可以遍历所有可能结果计算期望，不必真的走完整条轨迹。

矩阵写法也很直观：令

$$
P_\pi(s,s')=\sum_a\pi(a\mid s)P(s'\mid s,a),\qquad
r_\pi(s)=\sum_a\pi(a\mid s)\bar r(s,a),
$$

则：

$$
v_\pi=r_\pi+\gamma P_\pi v_\pi,
\qquad
v_\pi=(I-\gamma P_\pi)^{-1}r_\pi.
$$

线性系统求解给出另一种评估方式，但大规模问题通常不适合显式求逆。

### 7.2 Policy improvement：用价值改进行动

已知 $v_\pi$ 和模型，可以选择：

$$
\mu_{\mathrm{new}}(s)\in\arg\max_a
\sum_{s',r}p(s',r\mid s,a)[r+\gamma v_\pi(s')]
=\arg\max_a q_\pi(s,a).
$$

因为最大值不小于旧策略的加权平均：

$$
q_\pi(s,\mu_{\mathrm{new}}(s))\ge v_\pi(s).
$$

对新策略反复展开该不等式，或利用 Bellman 算子的单调性，就得到策略改进结论：

$$
v_{\mu_{\mathrm{new}}}(s)\ge v_\pi(s).
$$

它依赖准确评估等条件。使用有误差的 $V$ 或 $Q$ 做贪心，不自动保证每一步真实表现都提高。

### 7.3 Policy iteration 与 value iteration

策略迭代交替做两件事：先评估当前策略，再按其价值改进策略。

```text
π₀ → 评估得到 vπ₀ → 贪心改进得到 π₁
   → 评估得到 vπ₁ → 贪心改进得到 π₂ → …
```

价值迭代直接使用：

$$
V_{k+1}=T_*V_k.
$$

它每次更新都包含动作最大化，不等待某个固定策略被完整评估。

| 方法 | 每次更新做什么 | 目标 |
| --- | --- | --- |
| 策略评估 | 按固定策略对动作加权 | 当前策略价值 |
| 策略迭代 | 交替评估与改进 | 最优策略与价值 |
| 价值迭代 | 直接做最优 Bellman backup | 最优价值 |

策略评估与价值迭代虽然都更新 $V$，但使用的算子和要到达的固定点不同。

## 8. 从精确期望到样本：return 与 target

模型未知时，通常不能计算完整的转移期望，但可以观察样本：

$$
(S_t,A_t,R_{t+1},S_{t+1}).
$$

算法构造一个更新目标 $U_t$，再把估计向目标移动：

$$
\boxed{V(S_t)\leftarrow V(S_t)+\alpha[U_t-V(S_t)].}
$$

**Return 是未来奖励的累计；target 是这次训练希望预测值靠近的量。** MC 把完整 return 当作 target，TD 使用“已观察奖励加剩余价值估计”作为 target。target 不是未知真值的另一个名字。

样本更新也并不是把等式中的期望凭空删除，而是以随机样本近似期望更新。单次样本会有噪声，需要跨多次访问积累信息。

## 9. 为什么是 target − estimate，为什么叫步长

### 9.1 从样本均值推导增量更新

假设同一状态已经观察到 $n$ 个回报样本 $U_1,\ldots,U_n$，其平均值为 $V_n$。再观察一个样本：

$$
\begin{aligned}
V_{n+1}
&=\frac{nV_n+U_{n+1}}{n+1}\\
&=V_n+\frac{1}{n+1}(U_{n+1}-V_n).
\end{aligned}
$$

因此“目标减去估计值”本身就来自增量平均。对每个状态使用访问次数倒数作为步长，可以实现各自的样本平均。

### 9.2 从梯度下降推导

把当前标量估计记为 $x$，将本次目标 $U$ 暂时视为常数。平方损失为：

$$
L(x)=\frac12(U-x)^2.
$$

其梯度为：

$$
\frac{\partial L}{\partial x}=x-U.
$$

沿负梯度更新：

$$
x_{\mathrm{new}}
=x-\alpha\frac{\partial L}{\partial x}
=x+\alpha(U-x).
$$

这就是价值估计常见的更新公式。**梯度决定更新方向和尺度，学习率控制沿该方向走多远，所以也叫步长。** 对普通梯度下降，参数变化的长度是 $\alpha\|\nabla L\|$，并非恒等于 $\alpha$。

当 $0\le\alpha\le1$ 时，还能写成加权平均：

$$
x_{\mathrm{new}}=(1-\alpha)x+\alpha U.
$$

$\alpha=0$ 不学习；$\alpha=1$ 完全采用当前目标；$\alpha=0.1$ 则移动当前差距的十分之一。例如 $x=4,U=10$，更新后是 $4.6$。学习率大于 1 时会越过目标，已不再是凸组合。

### 9.3 常数步长意味着更重视新信息

固定 $\alpha$，连续更新 $n$ 次可展开为：

$$
V_n=(1-\alpha)^nV_0
+\sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}U_i.
$$

这是一种指数加权：新样本权重大，旧样本影响逐渐减弱。策略和环境变化时，这有助于跟踪变化；在固定随机问题中，常数步长通常保留波动，不意味着精确收敛到真值。

经典随机逼近常使用：

$$
\sum_n\alpha_n=\infty,\qquad
\sum_n\alpha_n^2<\infty.
$$

它们表达“长期仍有足够更新量，同时噪声影响可控”。具体收敛还需要访问覆盖、采样和任务条件，不能只检查学习率就宣称收敛。

### 9.4 神经网络与 semi-gradient

用参数化价值函数 $V_\phi(s)$ 时：

$$
L(\phi)=\frac12[\operatorname{sg}(U_t)-V_\phi(S_t)]^2,
$$

其中 $\operatorname{sg}$ 表示停止梯度。更新为：

$$
\phi\leftarrow\phi+alpha
[U_t-V_\phi(S_t)]\nabla_\phi V_\phi(S_t).
$$

表格情形中，一个状态值就是一个独立参数，所以退化为前面的标量更新。神经网络共享参数，一次更新还会影响其他状态的预测。

TD 目标含有 $V_\phi(S_{t+1})$，但标准 TD 更新不沿目标这一侧反向传播，因此叫**半梯度**。它是对本次固定目标的回归更新，不能直接说成对完整 Bellman 残差目标做普通梯度下降。若不停止目标梯度，就已经改变了算法。

## 10. MC、TD 与 n-step：三种目标如何构造

### 10.1 Monte Carlo：等到终局，使用完整回报

$$
U_t^{\mathrm{MC}}=G_t,
\qquad
V(S_t)\leftarrow V(S_t)+\alpha[G_t-V(S_t)].
$$

在固定策略、完整回合、适当采样条件下，$G_t$ 的条件期望就是 $v_\pi(S_t)$。它不依赖后续价值估计，但必须拿到完整回报，且长轨迹可能带来较大方差。

### 10.2 TD(0)：一步奖励加下一状态估计

$$
U_t^{\mathrm{TD}}=R_{t+1}+\gamma V(S_{t+1}).
$$

定义 TD error：

$$
\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t).
$$

然后：

$$
V(S_t)\leftarrow V(S_t)+\alpha\delta_t.
$$

直觉是：“原先觉得当前状态值这么多，现在真实走了一步，结合下一状态的估计，应当修正到哪里？”终止状态的后续价值为零，最后一步的目标就是最后奖励。

TD 不必等待整局结束，但目标会继承下一状态估计的误差。MC 与 TD 的偏差、方差比较依赖具体任务；常见直觉是 TD 用 bootstrap 换取更及时、往往方差更低的更新。

### 10.3 n-step TD：先观察多步，再估计剩余部分

若 $t+n<T$：

$$
U_t^{(n)}
=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}
+\gamma^nV(S_{t+n}).
$$

若观察窗口已经到达终点，则只累积到 $T$，不再接价值估计：

$$
U_t^{(n)}=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}=G_t,
\qquad t+n\ge T.
$$

“部分 bootstrap”指前面 $n$ 步已经使用真实奖励，剩余尾部仍由价值函数估计。并不是每个奖励都混了一部分估计。

### 10.4 Bootstrap 与是否采样是两个不同维度

Bootstrap 指用已有价值估计构造新的价值目标。DP 虽然对所有转移做精确期望，但仍使用当前 $V(s')$，所以也 bootstrap。

| 方法 | 如何处理环境结果 | 是否使用后续价值估计 |
| --- | --- | --- |
| 迭代式 DP 策略评估 | 按模型计算完整期望 | 是 |
| MC | 采样完整回报 | 否 |
| TD(0) | 采样一步转移 | 是 |
| n-step TD | 采样多步奖励 | 未到终点时是 |

完整期望与样本更新是一条轴，bootstrap 与否是另一条轴。两者不能混为一谈。

## 11. 从评价状态到选择动作：为什么需要 Q

假设两个动作价值为 10 和 0，均匀策略的状态价值为 5；两个动作价值都是 5，状态价值也为 5。仅凭当前状态的一个标量，分不清该选择哪个动作。

有模型时，可以把 $V$ 与每个动作的后果结合，做一步前瞻。无模型时，学习 $Q(s,a)$ 更直接：

$$
\mu(s)\in\arg\max_a Q(s,a).
$$

但这不意味着所有无模型控制都必须显式学习 Q。后面会看到，Actor-Critic 可以使用状态价值、采样奖励和策略梯度来改进行动。

### 11.1 argmax 是查表，还是可以用于神经网络？

$\arg\max$ 是一个数学操作，与函数如何表示无关。

```python
# 表格表示
action = Q[state].argmax()

# 离散动作 Q 网络：一次输出各动作的价值
q_values = q_network(state_tensor)
action = q_values.argmax(dim=-1)
```

对策略网络的动作概率取最大，则是在选择最可能动作，并不等于对 Q 做贪心，也不是训练策略网络的方法。策略梯度训练时通常需要从动作分布采样。

连续动作下，$\arg\max_a Q(s,a)$ 本身是一个优化问题，无法像小规模离散动作那样直接枚举；这也是引入显式 actor 的一个动机。

### 11.2 探索策略

令 $\mathcal G(s)=\arg\max_a Q(s,a)$ 为所有并列最优动作组成的集合。均匀处理并列项的 epsilon-greedy 策略为：

$$
\pi(a\mid s)=\frac{\epsilon}{|\mathcal A|}
+(1-\epsilon)\frac{\mathbf1\{a\in\mathcal G(s)\}}{|\mathcal G(s)|}.
$$

探索让尚未充分尝试的动作仍有机会产生数据。过早完全贪心，可能把初期的估计误差固化成长期行为。

## 12. SARSA、Expected SARSA 与 Q-learning

### 12.1 SARSA：按实际后续策略评估

观察到下一动作 $A_{t+1}\sim\pi(\cdot\mid S_{t+1})$ 后：

$$
U_t=R_{t+1}+\gamma Q(S_{t+1},A_{t+1}),
$$

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)
+\alpha[U_t-Q(S_t,A_t)].
$$

它使用五元组“状态、动作、奖励、下一状态、下一动作”，因此得名 SARSA。标准 on-policy SARSA 中，产生数据的策略与被评估的策略一致，探索带来的后续风险也进入价值估计。

### 12.2 Expected SARSA：对下一动作求期望

$$
U_t=R_{t+1}+\gamma\sum_{a'}\pi(a'\mid S_{t+1})Q(S_{t+1},a').
$$

环境转移仍是样本，只是下一动作从采样改为对目标策略求平均。是否 on-policy 取决于这个目标策略是否与行为策略一致。

### 12.3 Q-learning：目标中采用贪心后续动作

$$
U_t=R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a'),
$$

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)
+\alpha[U_t-Q(S_t,A_t)].
$$

行为策略可以继续探索，目标却按照当前 Q 的贪心动作构造。这是 Bellman 最优算子的样本版本。

有限表格、奖励有界、充分访问各状态动作对、合适步长等条件下，Q-learning 可以收敛到 $q_*$。神经网络版本不直接继承这个一般性保证。

### 12.4 一个数字例子

设当前 $Q(s,a)=4$，奖励为 1，折扣因子为 0.9，下一状态两个动作的 Q 值是 6 和 2。

若 SARSA 实际选到第二个动作：

$$
U^{\mathrm{SARSA}}=1+0.9\times2=2.8.
$$

若目标策略以 0.8 和 0.2 的概率选择两个动作：

$$
U^{\mathrm{Expected}}=1+0.9(0.8\times6+0.2\times2)=5.68.
$$

Q-learning 的目标为：

$$
U^{\mathrm{Q}}=1+0.9\times6=6.4.
$$

当 $\alpha=0.1$ 时，三种更新后的值依次为 3.88、4.168、4.24。差异来自“假设未来如何行动”，不是来自不同的更新外壳。

### 12.5 最小 Q-learning 更新代码

下面是单次转移的更新片段，假设 `Q` 为二维数组，动作已由带探索的行为策略选择。

```python
def q_learning_update(Q, s, a, reward, next_s, terminated,
                      alpha=0.1, gamma=0.99):
    # 真正的任务终止后，未来回报为零。
    if terminated:
        target = reward
    else:
        target = reward + gamma * Q[next_s].max()
    td_error = target - Q[s, a]
    Q[s, a] += alpha * td_error
    return td_error
```

要区分任务真正终止与外部时间限制截断。若截断只是采样器停止收集，而任务本身仍继续，通常应从最后的有效观测 bootstrap；不能误用自动 reset 后的新回合初始观测。若时间限制本身定义了任务终点，则需按该有限时域任务建模。

### 12.6 从表格 Q 到 DQN

用神经网络 $Q_\phi$ 代替表格后，常见目标为：

$$
U_t=R_{t+1}+\gamma(1-d_t)\max_{a'}Q_{\bar\phi}(S_{t+1},a'),
$$

$$
L_Q(\phi)=\mathbb E_{(s,a,r,s',d)\sim\mathcal D}
\left[\frac12(\operatorname{sg}(U)-Q_\phi(s,a))^2\right].
$$

$d_t$ 表示真正终止，$\bar\phi$ 是滞后更新的目标网络参数，$\mathcal D$ 是经验回放缓存。回放缓解连续采样的相关性并复用数据；目标网络减慢监督目标变化。它们帮助稳定训练，但不等于消除了函数逼近、bootstrap 与离策略学习组合带来的风险。

## 13. Policy Gradient：直接优化策略

价值方法可以通过 Q 的贪心操作改变行动。另一条路线直接参数化策略 $\pi_\theta(a\mid s)$，最大化：

$$
J(\theta)=\mathbb E_{\tau\sim p_\theta}[G_0].
$$

更新方向是梯度上升：

$$
\theta\leftarrow\theta+\alpha\nabla_\theta J(\theta).
$$

### 13.1 为什么会出现 log probability

以下先用有限时域推导；终止后可补零奖励。假设环境动力学、奖励规则和初始状态分布不直接依赖策略参数。

利用对数导数恒等式：

$$
\nabla_\theta p_\theta(\tau)
=p_\theta(\tau)\nabla_\theta\log p_\theta(\tau).
$$

轨迹概率取对数后，只有动作概率依赖 $\theta$：

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(A_t\mid S_t).
$$

因此：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\sum_\tau\nabla_\theta p_\theta(\tau)G_0\\
&=\mathbb E_{\tau\sim p_\theta}
\left[G_0\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(A_t\mid S_t)\right].
\end{aligned}
$$

不需要对环境本身求导，因为策略参数通过轨迹概率影响期望回报。

### 13.2 用 reward-to-go 去掉过去奖励

动作 $A_t$ 不会改变已发生的奖励。对应过去奖励的梯度项在期望中为零，于是：

$$
\boxed{
\nabla_\theta J(\theta)
=\mathbb E_{\tau\sim p_\theta}
\left[\sum_{t=0}^{T-1}\gamma^tG_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)\right].
}
$$

这里外面的 $\gamma^t$ 很重要：$G_t$ 从时刻 $t$ 重新开始计折扣，而目标 $G_0$ 从时刻 0 开始计折扣。对于本文定义的折扣目标，不能没有说明就把这一项删掉。无折扣有限回合取 $\gamma=1$ 时，它自然消失。

对无限时域折扣问题，也可以使用归一化折扣状态访问分布：

$$
d_{\pi,\gamma}(s)=(1-\gamma)
\sum_{t=0}^{\infty}\gamma^t\Pr_\pi(S_t=s).
$$

此时策略梯度定理写成：

$$
\nabla_\theta J(\theta)
=\frac{1}{1-\gamma}
\mathbb E_{S\sim d_{\pi_\theta,\gamma},\,A\sim\pi_\theta(\cdot\mid S)}
[q_{\pi_\theta}(S,A)\nabla_\theta\log\pi_\theta(A\mid S)].
$$

时间权重被吸收到状态分布中，期望符号因此变得紧凑。看见没有下标的策略梯度公式时，应先确认它采用什么目标和访问分布。

### 13.3 梯度的工程直觉

$\nabla_\theta\log\pi_\theta(A_t\mid S_t)$ 指向让已采样动作更可能出现的局部方向。用回报作为权重，就让带来更好结果的样本对更新贡献更多。

但“低回报就降低概率”并不总成立：若没有 baseline，且所有回报都是正数，单个样本的系数依然为正。要表达“比预期差，因此降低倾向”，需要 advantage。

共享参数还会联动影响其他状态的动作分布，因此这种概率变化直觉是局部解释，不是任意有限步长下的严格单调保证。

## 14. Baseline、advantage 与 Actor-Critic

### 14.1 为什么可以减去状态 baseline

对不依赖当前动作的 $b(s)$：

$$
\begin{aligned}
\mathbb E_{A\sim\pi_\theta(\cdot\mid s)}
[b(s)\nabla_\theta\log\pi_\theta(A\mid s)]
&=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta1=0.
\end{aligned}
$$

因此，在策略梯度估计中减去这个 baseline 不改变期望；合适的 baseline 可以降低方差。若 baseline 也通过网络计算，actor 的这一项仍应将其视为固定权重。

常用 $b(s)=v_\pi(s)$，定义优势函数：

$$
A_\pi(s,a)=q_\pi(s,a)-v_\pi(s).
$$

它表示该动作相对当前策略在该状态的平均水平，好多少或差多少，而且：

$$
\sum_a\pi(a\mid s)A_\pi(s,a)=0.
$$

### 14.2 两种 advantage 估计

使用完整回报：

$$
\hat A_t=G_t-V_\phi(S_t).
$$

使用一步 TD error：

$$
\hat A_t=\delta_t
=R_{t+1}+\gamma V_\phi(S_{t+1})-V_\phi(S_t).
$$

若 $V_\phi=v_\pi$，则：

$$
\mathbb E[\delta_t\mid S_t=s,A_t=a]=A_\pi(s,a).
$$

所以单步 TD error 是 advantage 的一个样本估计，不是每次都恰好等于真实 advantage；critic 不准确时还会引入偏差。

### 14.3 Actor-Critic 的两个学习目标

Actor 是策略 $\pi_\theta$，critic 是价值估计 $V_\phi$。

Critic 用 TD 目标回归：

$$
U_t=R_{t+1}+\gamma(1-d_t)V_\phi(S_{t+1}),
\qquad
L_V(\phi)=\frac12[\operatorname{sg}(U_t)-V_\phi(S_t)]^2.
$$

Actor 则用 advantage 加权动作对数概率。与前文有限回合折扣目标一致的轨迹损失为：

$$
L_\pi(\theta)
=-\sum_{t=0}^{T-1}\gamma^t
\log\pi_\theta(A_t\mid S_t)\operatorname{sg}(\hat A_t).
$$

最小化这个损失实现对应的梯度上升。若均匀按时间步采样却直接去掉 $\gamma^t$，就需要另行解释所优化的目标或近似。

下面展示单个 on-policy 转移的核心更新，假设 actor 与 critic 参数独立，`t` 是当前回合内步数：

```python
value = critic(obs).squeeze(-1)
with torch.no_grad():
    next_value = critic(next_obs).squeeze(-1)
    target = reward + gamma * (1.0 - terminated.float()) * next_value
advantage = target - value

critic_loss = 0.5 * advantage.square().mean()
dist = torch.distributions.Categorical(logits=actor(obs))
log_prob = dist.log_prob(action)
actor_loss = -(gamma ** t * log_prob * advantage.detach()).mean()

critic_optimizer.zero_grad()
critic_loss.backward()
critic_optimizer.step()

actor_optimizer.zero_grad()
actor_loss.backward()
actor_optimizer.step()
```

`target` 停止梯度，避免 critic 同时移动预测和目标；`advantage.detach()` 则让 actor 把评价当作权重，不通过它反向修改 critic。

若按完整轨迹估计前面定义的目标，应先对每条轨迹内的时间项求和，再对轨迹取平均；变长轨迹全部摊平后按步平均，会改变相对权重。代码片段展示的是局部更新机制，不是完整训练器。

### 14.4 GAE：把多个 TD error 连起来

常见优势估计是：

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.
$$

这里 $T$ 是当前轨迹段的边界；若是真正终点，末尾价值为零；若只是收集边界，末尾使用 bootstrap 价值。

$\lambda=0$ 时只取一步 TD error；$\lambda=1$ 时和式望远镜相消，得到该轨迹段的多步目标减去当前价值。在完整终止回合中，它等于 $G_t-V(S_t)$。中间取值用于调整对更远期 TD 信号的权重。

## 15. 统一理解 on-policy 与 off-policy

用 $b$ 表示产生数据的行为策略，用 $\pi$ 表示当前评估或优化的目标策略。

**On-policy：数据由正在学习的策略产生。Off-policy：使用一个策略的数据，学习另一个策略。** 这是价值方法和策略梯度共同使用的定义。

| 方法 | 数据从哪里来 | 目标中的策略 |
| --- | --- | --- |
| 标准 SARSA | 当前带探索策略 | 同一策略 |
| Q-learning | 可来自带探索行为策略 | Q 所隐含的贪心策略 |
| REINFORCE | 当前策略的新轨迹 | 当前策略 |
| 常见 on-policy Actor-Critic | 当前策略的新轨迹 | 当前策略 |
| 使用旧数据的策略优化 | 旧策略或其他行为策略 | 当前待优化策略 |

### 15.1 为什么 Q-learning 不总需要重要性采样比率

给定状态动作对 $(s,a)$，下一奖励和状态来自环境 $p(\cdot,\cdot\mid s,a)$，而不是行为策略决定的另一个环境。Q-learning 的一步目标直接对下一动作取最大，无须对行为策略实际选出的下一动作求平均。

因此它可以在充分覆盖等条件下使用其他行为策略的转移。这个结论不能推广为“任何离策略多步回报、任何策略梯度都无需修正”。

### 15.2 策略梯度为什么在意数据来自谁

策略梯度中的期望原本在当前策略的轨迹分布下。如果改用 $b$ 的轨迹，分布就变了。

满足支持覆盖条件时，完整轨迹的重要性权重为：

$$
w(\tau)=\frac{p_\pi(\tau)}{p_b(\tau)}
=\prod_{t=0}^{T-1}\frac{\pi(A_t\mid S_t)}{b(A_t\mid S_t)}.
$$

环境项相消，从而：

$$
\mathbb E_{\tau\sim p_\pi}[f(\tau)]
=\mathbb E_{\tau\sim p_b}[w(\tau)f(\tau)].
$$

轨迹很长时，概率比乘积可能产生极大方差。只乘单步动作比率，通常不足以自动修正整个状态访问分布的差异；具体离策略算法需要结合目标设计、价值估计和相应修正来理解。

### 15.3 PPO 的位置

PPO 常用刚采集的旧策略数据进行若干轮更新，然后重新采样，通常归入 on-policy 算法。它使用：

$$
\rho_t(\theta)=
\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\theta_{\mathrm{old}}}(A_t\mid S_t)},
$$

以及裁剪代理目标：

$$
L^{\mathrm{CLIP}}(\theta)
=\hat{\mathbb E}_t\left[
\min\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)\right].
$$

这里 $\hat{\mathbb E}_t$ 是采样批次的经验平均。裁剪限制的是代理目标中某些继续改变概率的收益，不是对所有状态动作概率比或 KL 距离施加硬约束。存在概率比，也不意味着算法可以无条件复用任意陈旧的数据。[PPO 原论文](https://arxiv.org/abs/1707.06347)给出了这一目标及其迭代采样、优化流程。

## 16. 一个两步问题，把整条链串起来

设初始状态 $s_0$ 有两个动作：立即结束得到 2 分；或者进入 $s_1$，当前奖励为 0。在 $s_1$ 完成任务后得到 10 分并结束。取 $\gamma=0.9$。

```text
s₀ ── 立即结束 / 奖励 2 ──→ terminal
 │
 └── 继续 / 奖励 0 ──→ s₁ ── 完成 / 奖励 10 ──→ terminal
```

因为 $v(s_1)=10$，所以：

$$
q(s_0,\mathrm{stop})=2,\qquad
q(s_0,\mathrm{continue})=0+0.9\times10=9.
$$

若初始策略等概率选两个动作：

$$
v_\pi(s_0)=0.5\times2+0.5\times9=5.5.
$$

改进策略会选择继续，得到最优价值 $v_*(s_0)=9$。这不是偏好延迟奖励，而是比较过完整折扣回报后的结果。

如果不知道模型，就通过采样学习。假设初始所有 $V$ 都是 0，采到“继续、完成”的一条轨迹，使用 $\alpha=1$：

1. MC 在终局计算 $G_1=10,G_0=9$，直接把两个状态更新为 10 和 9。
2. 在线一步 TD 在第一次离开 $s_0$ 时，目标是 $0+0.9\times0=0$；到达终局后才把 $V(s_1)$ 更新为 10。再次访问相同路线时，10 的价值才通过 bootstrap 传回 $s_0$。
3. 若使用回放或反向顺序额外更新，价值可以更快传播；更新顺序因此影响学习速度。

在当前策略下，两个动作的真实 advantage 分别是：

$$
A_\pi(s_0,\mathrm{stop})=2-5.5=-3.5,
$$

$$
A_\pi(s_0,\mathrm{continue})=9-5.5=3.5.
$$

这给策略梯度提供了清楚的相对信号：继续比平均表现更好，立即结束比平均表现更差。价值方法和策略梯度在这个问题上利用的是同一套长期后果，只是改变策略的方式不同。

## 17. 阅读算法时，先识别它在优化什么

面对一篇 RL 论文或一段实现，可以依次回答以下问题：

1. **任务目标是什么？** 有限回合还是持续任务，是否折扣，何时真正终止？
2. **数据来自谁？** 当前策略、旧策略，还是离线数据集？
3. **学习哪个对象？** 当前策略的 V/Q、最优 Q，还是直接学习策略参数？
4. **target 如何构造？** 完整回报、一步或多步 bootstrap，还是模型期望？
5. **梯度流向哪里？** target 是否停止梯度，actor 的 advantage 是否作为固定权重？
6. **策略如何改进？** 贪心或 epsilon-greedy、模型前瞻，还是策略梯度？
7. **如何验证改善？** 关注独立评估回报、成功率和波动，而不只看训练损失。

可以把本文涉及的方法放回同一张图：

```text
交互 → 轨迹 → return → 最大化期望回报
                │
                ├── 价值：vπ / qπ / v* / q*
                │      └── Bellman 自洽关系
                │             ├── 已知模型：DP 评估 / 策略迭代 / 价值迭代
                │             └── 样本学习：MC / TD / n-step
                │                    └── SARSA / Q-learning / DQN
                │                           └── 从 Q 导出策略
                │
                └── 直接优化 πθ：Policy Gradient / REINFORCE
                              ↑
                    Critic 提供 advantage
                              └── Actor-Critic / GAE / PPO
```

价值函数把未来回报压缩成可以估计的对象，Bellman 方程把长期问题拆成局部递归关系，学习算法决定如何用数据逼近这些关系，策略更新则把估计转化为下一轮行动。把这几个层次分清，许多看起来不同的 RL 公式就能放进同一个体系。

## 参考资料与延伸阅读

- Sutton & Barto，[*Reinforcement Learning: An Introduction*, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：第 3—4 章对应 MDP 与动态规划，第 5—7 章对应 MC、TD 与多步方法，第 9、13 章对应函数逼近与策略梯度。
- OpenAI Spinning Up，[Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：补充对数概率梯度、reward-to-go 与 baseline 的推导；该页面先采用有限时域无折扣目标，阅读时应与本文折扣约定区分。
- Schulman 等，[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：GAE 的原始论文。
- Schulman 等，[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：PPO 的裁剪代理目标与训练流程。
