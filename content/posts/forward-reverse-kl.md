---
title: "Forward KL vs Reverse KL：mode-covering 与 mode-seeking 的数学根源"
date: 2026-08-24T17:10:00+08:00
draft: false
tags: ["KL 散度", "蒸馏", "笔记"]
categories: ["AI 技术思考"]
math: true
summary: "“Forward KL = mode-covering，Reverse KL = mode-seeking”不是经验口号，而是直接来自 KL 两边“谁负责加权”的不同。用双峰 Teacher + 单峰 Student 把这种不对称放大到肉眼可见。"
---

「Forward KL = mode-covering，Reverse KL = mode-seeking」不是一句经验口号，它直接来自 KL 两边「谁负责加权」的不同。

先给结论。前向 KL

$$D_{\mathrm{KL}}(P_T \,\|\, P_S) = \mathbb{E}_{y\sim P_T}\!\left[\log\frac{P_T(y)}{P_S(y)}\right]$$

是 **Teacher** 在决定「哪些地方重要」；而反向 KL

$$D_{\mathrm{KL}}(P_S \,\|\, P_T) = \mathbb{E}_{y\sim P_S}\!\left[\log\frac{P_S(y)}{P_T(y)}\right]$$

是 **Student** 在决定「哪些地方重要」。仅这一个交换，就导致了完全不同的行为。

![Teacher（双峰）、Forward KL（mode-covering / zero-avoiding，中间出现 hallucination）、Reverse KL（mode-seeking / zero-forcing，丢掉一个 mode）三种分布对比](/images/forward-reverse-kl.png)

> 图片来源：*A Survey of On-Policy Distillation for Large Language Models*

## 1. Forward KL：为什么它不敢漏掉 Teacher 的 mode

写成积分形式：

$$D_{\mathrm{KL}}(P_T\|P_S) = \int P_T(y)\log\frac{P_T(y)}{P_S(y)}\,dy$$

优化 $P_S$ 时 $P_T$ 固定，因此等价于

$$\min_{P_S}\ -\,\mathbb{E}_{y\sim P_T}\log P_S(y)$$

也就是说：**Teacher 经常出现的地方，Student 必须给足概率。** 假设 Teacher 有两个 mode，$P_T(A)=0.5,\ P_T(B)=0.5$；如果 Student 只覆盖 A，即 $P_S(A)\approx1,\ P_S(B)\approx0$，那么 B 这一块的 KL 项是

$$0.5\log\frac{0.5}{P_S(B)}$$

当 $P_S(B)\to0$ 时 $\log\frac{0.5}{P_S(B)}\to\infty$，于是 $D_{\mathrm{KL}}(P_T\|P_S)\to\infty$。所以 Forward KL 极度害怕

$$\boxed{P_T(y)>0,\quad P_S(y)\approx0}$$

即「Teacher 认为这里有合法答案，但 Student 完全不给概率」。因此 Student 最安全的选择就是：**Teacher 有哪些 mode，我尽量全覆盖**——这就是 mode-covering。

## 2. 为什么又叫 zero-avoiding

Forward KL $D_{\mathrm{KL}}(P_T\|P_S)$ 的对数项分母是 $P_S(y)$。如果 $P_T(y)>0$ 但 $P_S(y)=0$，则 $\log\frac{P_T(y)}{0}=\infty$。所以 Student 会极力避免 $P_S(y)=0$，这就是所谓 **zero-avoiding**——更准确地说：**Student 不敢在 Teacher 有概率质量的地方变成零。**

## 3. 副作用：为什么会「覆盖过头」

假设 Teacher 是两个峰

$$P_T(y) = \tfrac12\mathcal{N}(-a,\sigma^2) + \tfrac12\mathcal{N}(a,\sigma^2)$$

而 Student 能力有限，只允许是一个单峰 Gaussian

$$P_S(y) = \mathcal{N}(m,s^2)$$

```text
Teacher:
      /\              /\
     /  \            /  \
____/    \__________/    \____
       -a              +a
```

Student 不可能同时变成两个峰，那 Forward KL 怎么办？答案很漂亮——**对 Gaussian Student，Forward KL 的最优解实际上是「矩匹配」。** 因为

$$D_{\mathrm{KL}}(P_T\|P_S) = \text{const} - \mathbb{E}_{P_T}\log P_S(y)$$

而 Gaussian 的对数密度

$$\log P_S(y) = -\tfrac12\log(2\pi s^2) - \frac{(y-m)^2}{2s^2}$$

因此优化相当于最小化 $\ \frac12\log s^2 + \dfrac{\mathbb{E}_{P_T}(y-m)^2}{2s^2}$。对 $m$ 求最优得 $m^\star = \mathbb{E}_{P_T}[y]$，由对称性 $m^\star=0$；对 $s^2$ 求最优得 $s^{2\star} = \operatorname{Var}_{P_T}(y) = a^2+\sigma^2$。所以 Student 最后变成

$$\boxed{\,P_S = \mathcal{N}(0,\ a^2+\sigma^2)\,}$$

```text
Forward-KL Student:
          ______
        /        \
_______/          \_______
```

它为了「两个 mode 都不能漏」，不得不把整个区域全部罩住，于是连 Teacher 本身概率很低的中间区域也给了不少概率——这就是图里所谓的 **hallucination zone**。

## 4. Reverse KL：逻辑完全翻转

$$D_{\mathrm{KL}}(P_S\|P_T) = \int P_S(y)\log\frac{P_S(y)}{P_T(y)}\,dy = \mathbb{E}_{y\sim P_S}\big[\log P_S(y)-\log P_T(y)\big]$$

注意现在积分权重变成了 $P_S(y)$。这意味着：**Student 只关心「自己会去的地方」。** 这是理解 Reverse KL 最关键的一句话。

## 5. Student 忽略一个 Teacher mode 会发生什么

假设 Teacher $P_T(A)=0.5,\ P_T(B)=0.5$，Student 决定只去 B，即 $P_S(B)\approx1,\ P_S(A)\approx0$。Reverse KL 在 A 区域的贡献近似

$$P_S(A)\log\frac{P_S(A)}{P_T(A)}$$

由于 $P_S(A)\approx0$，这一项 $\approx0$。这就是关键——Teacher 可能一直喊「A 也是非常好的答案！」，但 Student 回答「可我根本不去 A，那与我有什么关系？」。数学上就是

$$P_S(A)\approx0 \quad\Rightarrow\quad \text{A 对 } D_{\mathrm{KL}}(P_S\|P_T) \text{ 几乎没有贡献}$$

所以 Reverse KL **允许 mode dropping**。

## 6. Reverse KL 真正害怕什么

它真正害怕

$$\boxed{P_S(y)>0,\qquad P_T(y)\approx0}$$

因为 $P_S(y)\log\dfrac{P_S(y)}{P_T(y)}$ 在 $P_T(y)\to0$ 时会非常大。所以 Student 会想：**我最好只待在 Teacher 高概率的区域，千万不要去两个峰之间。** 这就是 **mode-seeking**。

## 7. 为什么叫 zero-forcing

Reverse KL $D_{\mathrm{KL}}(P_S\|P_T)$ 的对数项分母是 $P_T$。如果 $P_T(y)=0$ 而 $P_S(y)>0$，那么 $\log\dfrac{P_S(y)}{0}=\infty$，因此

$$P_T(y)=0 \ \Rightarrow\ P_S(y)\to0$$

Student 被迫在 Teacher 为零的位置也变成零——所以叫 **zero-forcing**。

## 8. 用双峰 Gaussian 更严格地看 Reverse KL

还是 $P_T(y) = \tfrac12\mathcal{N}(-a,\sigma^2) + \tfrac12\mathcal{N}(a,\sigma^2)$，Student $P_S(y)=\mathcal{N}(m,s^2)$，并假设两个 mode 相距非常远 $a\gg\sigma$。如果 Student 选择右边，$m\approx a,\ s^2\approx\sigma^2$，那么在 Student 经常采样的位置 $y\approx a$，Teacher 近似为 $P_T(y)\approx\tfrac12\mathcal{N}(a,\sigma^2)$，于是

$$D_{\mathrm{KL}}(P_S\|P_T) \approx D_{\mathrm{KL}}\!\big(P_S \,\|\, \mathcal{N}(a,\sigma^2)\big) + \log 2$$

因此最优自然就是 $P_S\approx\mathcal{N}(a,\sigma^2)$，或对称地选择左峰。也就是说 $\ \boxed{m^\star\approx +a \ \text{ or } \ -a}\ $，而不是 $m^\star=0$。

## 9. 为什么 Reverse KL 不选中间那个「大胖 Gaussian」

因为如果 Student 放很多概率在 $y\approx0$，Teacher 在这里的概率大约是

$$P_T(0) \propto \exp\!\left(-\frac{a^2}{2\sigma^2}\right)$$

当 $a\gg\sigma$ 时 $P_T(0)\approx0$，于是 Reverse KL 中的 $-\log P_T(0)$ 极大。也就是说：**Student 只要跑到两个 mode 中间，就会受到巨大惩罚。** 因此它宁愿只选择一个 mode，也绝不愿意把两个 mode 平均起来——这就是 mode-seeking 的数学根源。

## 10. 两种 KL 的「心理活动」

**Forward KL** $D_{\mathrm{KL}}(P_T\|P_S) = \mathbb{E}_{\color{blue}{P_T}}[\cdots]$——Teacher 说「我的每一个高概率区域，你都必须照顾到」，于是 Student 说「好，那我全部罩住」。结果是 **Coverage**，代价是**可能覆盖 Teacher 的低密度区域**。

**Reverse KL** $D_{\mathrm{KL}}(P_S\|P_T) = \mathbb{E}_{\color{blue}{P_S}}[\cdots]$——Student 说「我只检查自己实际会去的地方」，只要选择 Mode 2，$P_S(\text{Mode 1})\approx0$，Mode 1 对 objective 几乎消失。结果是 **Seeking**，代价是 **Mode Dropping**。

## 11. 最关键的数学差异

| | Forward KL | Reverse KL |
| --- | --- | --- |
| 形式 | $D(P_T\parallel P_S)$ | $D(P_S\parallel P_T)$ |
| 谁加权 | Teacher | Student |
| 最怕什么 | Teacher 有质量但 Student 没覆盖 | Student 跑到 Teacher 低概率区 |
| $P_T>0,\ P_S\to0$ | $\to\infty$ | 几乎不关心 |
| $P_S>0,\ P_T\to0$ | 基本不直接惩罚 | $\to\infty$ |
| 行为 | mode covering | mode seeking |
| 风险 | 两个 mode 中间也分配质量 | 丢掉合法 mode |

真正该记的是中间两行，而不是死记「Forward = covering，Reverse = seeking」。

## 12. 回到 LLM：这个性质非常重要

假设 Teacher 对一个问题有两套都正确的 reasoning：Mode A 是代数解法，Mode B 是几何解法，而 Student capacity 有限。

**Forward KL 的倾向**：希望 $P_S(A)>0$ 且 $P_S(B)>0$，两个都学。但如果模型表示能力有限，可能出现某种「平均」——A 的前半段 + B 的后半段，这就是图中所谓的 inter-mode region。（当然在真正的离散 token space 中，并不会真的像 Gaussian 一样几何平均，图片只是直觉示意。）

**Reverse KL 的倾向**：Student 可能发现 Mode B 更容易稳定执行，于是 $P_S(B)\uparrow$、$P_S(A)\downarrow$，最后 $P_S(A)\approx0$，只保留一种非常 coherent 的解法。这就是为什么 Reverse KL 常被描述成 **sharp / coherent / mode-seeking**；但缺点同样明显——**可能丢失 Teacher 的 diversity**。

## 13. 一个重要限定：这不是 KL 永恒的「魔法属性」

如果 Student 分布族足够强，能够精确表示 Teacher，即 $P_S=P_T$，那么 $D(P_T\|P_S)=0$ 且 $D(P_S\|P_T)=0$，两者的最优解完全一样：$\boxed{P_S^\star=P_T}$。

所以「Forward KL mode-covering、Reverse KL mode-seeking」真正成立得最明显的场景是

$$\boxed{\text{Teacher multimodal} + \text{Student family capacity constrained}}$$

也就是存在 **distribution / model-family mismatch**。图里之所以故意规定 Teacher = bimodal、Student = single Gaussian，就是为了把这种 asymmetry 放大到肉眼可见。

## 最后：把数学原理压缩成一句话

Forward KL：

$$\boxed{\,D(P_T\|P_S) = \mathbb{E}_{P_T}[-\log P_S] + \text{const}\,}$$

> Teacher 去过的地方，Student 一个都不能漏。

因此 **mode-covering**。

Reverse KL：

$$\boxed{\,D(P_S\|P_T) = \mathbb{E}_{P_S}[-\log P_T] - H(P_S)\,}$$

> Student 只需保证自己去的地方，Teacher 也认为是高概率区域。

因此它可以完全不去其他合法 mode，形成 **mode-seeking**。把这两句话真正理解透，forward / reverse KL 基本就不需要背了。
