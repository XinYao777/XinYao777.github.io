---
title: "大模型眼里的世界，不是由「词」组成的：从中文分词、BPE 到多模态 Tokenizer"
date: 2026-09-15T20:00:00+08:00
draft: false
tags: ["Tokenizer", "BPE", "多模态", "分词", "大模型", "笔记"]
categories: ["AI 技术思考"]
math: true
summary: "如果问大模型处理语言最基本的单位是什么，很多人第一反应是「词」。但现代大模型并不是这样工作的。沿着「人类定义语言单位 → 数据发现统计单位 → 模型学习离散单位 → 模型动态分配计算单位」这条历史线，重新理解 Tokenizer 这一最基础、也最容易被忽略的抽象层：Token 不是现实世界天然存在的「意义原子」，而是模型世界里的「计算原子」。"
---

> 从中文分词、BPE、\(P(x)\) 到多模态 Tokenizer：重新理解大模型最基础、也最容易被忽略的一层抽象。

如果问一个刚接触大模型的人：

> 大语言模型处理语言时，最基本的单位是什么？

很多人的第一反应会是：

**词。**

毕竟人类学习语言时就是这样理解的：

```text
I / love / artificial / intelligence
```

中文似乎也可以写成：

```text
我 / 喜欢 / 人工智能
```

所以我们很容易形成一个朴素的世界观：

> 世界上先存在「词」，Tokenizer 的工作只是把这些词找出来，然后交给大模型。

但现代大模型真正的工作方式并不是这样。

同一句：

```text
人工智能正在改变世界
```

在不同模型眼中，可能是：

```text
[人工智能] [正在] [改变] [世界]
```

也可能是：

```text
[人工] [智能] [正在] [改变] [世界]
```

还可能接近：

```text
[人] [工] [智] [能] [正] [在] [改] [变] [世] [界]
```

甚至在 Byte-level Tokenizer 的底层，一个汉字还可能暂时被表示为若干 UTF-8 byte。

那么问题就来了：

> 到底哪一种才是「正确」的切分？

答案可能令人意外：

\[
\boxed{\text{不存在唯一正确的 Tokenization}}
\]

因为 Token 从来不只是语言学中的「词」。

它同时是：

\[
\text{信息表示单位}
\]

是：

\[
\text{压缩单位}
\]

是：

\[
\text{概率建模单位}
\]

也是：

\[
\text{Transformer 的计算单位}
\]

如果再从自回归生成的角度观察，它甚至可以被理解为：

\[
\boxed{\text{一次决策的动作粒度}}
\]

而到了图像、声音和视频时代，事情变得更加明显：图片没有「词」，声音也没有「词」。

可今天我们仍然在讨论 image token、audio token、video token、action token。

这说明 Token 这个概念，早就超出了自然语言中的「词」。

本文试图沿着一条历史线，把这个问题重新讲清楚：

\[
\boxed{
\text{人类定义语言单位}
\rightarrow
\text{数据发现统计单位}
\rightarrow
\text{模型学习离散单位}
\rightarrow
\text{模型动态分配计算单位}
}
\]

最终我们会得到一个比「Tokenizer 是分词器」更准确的认识：

\[
\boxed{
\text{Token 是模型世界里的计算原子，而不是人类世界里天然存在的意义原子。}
}
\]

## 第一部分：缘起——机器如何理解语言

### 故事要从一个比大模型早 70 年的问题讲起：机器眼里的语言是什么？

1954 年 1 月，Georgetown University 与 IBM 做了一次后来非常著名的机器翻译演示。

系统能够把六十多个俄语句子翻译成英语。

但它的规模今天看非常有限：词汇表只有约 250 个 lexical items，并依靠 6 条 grammar rules 工作。

可以把它粗略想象成：

```text
俄语单词
   ↓
查词典
   ↓
确定词义
   ↓
应用语法规则
   ↓
调整词序
   ↓
英语
```

这里隐含着一个非常强的人类假设：

> **语言是由「词、词性、语法结构」这些人类语言学单位组成的。**

机器的任务不是发现这些结构，而是：人把语言结构告诉机器，机器执行规则。

所以早期机器翻译的基本对象，是：

```text
word
morpheme
part-of-speech
syntax tree
semantic role
```

这些都是人类先定义好的符号。

从今天看，这是一种非常典型的：

\[
\boxed{\text{Human-designed Representation}}
\]

### 机器翻译曾经做过一个宏大的梦：能不能先翻译成「意义」？

早期机器翻译的发展过程中，出现过一张非常经典的图：**Vauquois Triangle。**

它把机器翻译大致分成：

\[
\text{Direct}
\rightarrow
\text{Transfer}
\rightarrow
\text{Interlingua}
\]

三个层次。

在最底层，机器直接做：

\[
\text{Source Word}
\rightarrow
\text{Target Word}
\]

例如：

```text
dog → 狗
apple → 苹果
```

再高级一点，不直接映射词，而是先解析：

```text
John gave Mary a book
```

得到类似：

```text
Subject: John
Verb: give
Indirect Object: Mary
Object: book
```

然后再转换成目标语言结构。

而 Vauquois Triangle 的最高层，是 **Interlingua**：

\[
L_{source}
\rightarrow
Z
\rightarrow
L_{target}
\]

其中：

\[
Z
\]

不属于英语，也不属于中文，而是某种**语言无关的意义表示**。

经典机器翻译理论认为，Direct、Transfer、Interlingua 的区别，本质是对源语言分析深度的不同；Interlingua 位于 Vauquois Triangle 顶部，是希望得到一种独立于具体语言表面的意义表示。

例如：

```text
John gave Mary a book.
```

可以不再保存英语词序，而转换成：

```text
EVENT       = GIVE
AGENT       = JOHN
RECIPIENT   = MARY
OBJECT      = BOOK
TENSE       = PAST
```

然后：

```text
English ─┐
Chinese ─┤
French  ─┼→ Universal Meaning → Any Language
Japanese ─┘
```

这是一个非常重要的思想。

因为七十年后，多模态大模型某种意义上又回到了这个问题：text、image、audio、video 是否可以进入某种：

\[
Z
\]

然后由一个统一模型进行理解、推理和生成？

区别在于：早期的 \(Z\) 是语言学家设计的。今天我们希望：

\[
Z=f_\theta(x)
\]

由数据和神经网络学习出来。

### 但在讨论「意义」之前，人们首先碰到了一个非常现实的问题：什么才叫一个词？

英语会让人产生一个错觉。因为：

```text
I love artificial intelligence
```

天然有空格。我们很容易写成：

```text
[I] [love] [artificial] [intelligence]
```

于是：

\[
\text{space}\approx\text{word boundary}
\]

似乎理所当然。

但中文立刻打破了这个直觉。例如：

```text
南京市长江大桥
```

正确理解通常是：

```text
南京市 / 长江大桥
```

但如果只看字符序列：

```text
南 京 市 长 江 大 桥
```

「市长」本身又是一个合法词。

于是中文 NLP 长时间存在一个基础任务：

\[
\boxed{\text{Chinese Word Segmentation}}
\]

因为：

```text
我喜欢人工智能
```

原始文本并没有告诉机器边界究竟在哪里：

```text
我 / 喜欢 / 人工智能
```

这里已经暴露出一个很深的问题：

\[
\boxed{\text{Word 并不是计算系统天然能够观察到的对象}}
\]

「词」本身就是一种抽象。

### 更麻烦的问题是：语言根本不是一个封闭集合

假设我们决定：

> 好，那就把所有「词」列出来，给每个词一个 ID。

于是：

```text
cat          → 1024
dog          → 815
computer     → 9342
intelligence → 18273
```

似乎问题解决了。但很快出现：

```text
ChatGPT
Schrödinger
electroencephalography
某个人名
某个新药
某个新公司
一个刚创造出来的词
```

自然语言的词集合并不是固定的。

因此如果：

\[
w\notin V
\]

早期神经机器翻译往往只能把它变成：

```text
<UNK>
```

即：

\[
\boxed{\text{Unknown Token}}
\]

这意味着：

```text
Xiuyi
Schrödinger
Obamacare
某个罕见地名
```

可能全都变成同一个符号：

```text
<UNK>
```

信息直接丢失。

Sennrich 等人在 2016 年那篇后来影响极大的论文《Neural Machine Translation of Rare Words with Subword Units》中，正是从神经机器翻译的 open-vocabulary / rare-word 问题出发，把 subword tokenization 推成了主流解决方案。

## 第二部分：从词到子词，再到字节

### 一个最暴力的办法：既然「词」会无限增长，那就不用词

例如：

```text
unbelievable
```

不要表示为：

```text
[unbelievable]
```

而是：

```text
[u][n][b][e][l][i][e][v][a][b][l][e]
```

那么词表只需要包含字符。

英文可能只需要几百种基本字符。中文字符集合虽然大得多，但仍然远小于所有可能词汇。

于是：

\[
\text{OOV}\downarrow
\]

甚至可以接近：

\[
0
\]

但问题从 vocabulary 转移到了 sequence length。

假设：

```text
artificial intelligence
```

按词：

```text
[artificial] [intelligence]
```

只有 2 个 token。按字符：

```text
[a][r][t][i][f][i][c][i][a][l]...
```

可能二十多个。

而 Transformer 的主要计算与序列长度高度相关。最经典的 self-attention：

\[
O(N^2)
\]

因此：

\[
N\uparrow
\]

意味着 Training Cost、Inference Cost、KV Cache 都上升，Effective Context 下降。

于是机器学习重新面对一个工程 trade-off：

\[
\boxed{
\text{Word 太粗}
\qquad
\text{Character 太细}
}
\]

有没有中间方案？答案就是：

\[
\boxed{\text{Subword}}
\]

### Subword 的本质不是「找到正确语言单位」，而是找到一个工程折中

例如：

```text
unbelievable
```

可以表示成：

```text
[un] [believ] [able]
```

常见词：

```text
apple
```

可能整体成为：

```text
[apple]
```

罕见词：

```text
microarchitecture
```

可能变成：

```text
[micro] [arch] [itecture]
```

更罕见的字符串则继续退化：

```text
xyzabc
```

变成：

```text
[x] [y] [z] [a] [b] [c]
```

这形成一个非常漂亮的自适应粒度：

\[
\text{frequent sequence}
\rightarrow
\text{large token}
\]

\[
\text{rare sequence}
\rightarrow
\text{small token}
\]

所以 Subword 实际解决的是：

\[
\boxed{
\text{Vocabulary Size}
\leftrightarrow
\text{Sequence Length}
}
\]

之间的折中。

这时候，一个在今天已经非常熟悉的算法进入了 NLP：**BPE。**

### BPE 最有意思的地方是：它原本根本不是一个 NLP 算法

Byte Pair Encoding 最早由 Philip Gage 在 1994 年作为数据压缩算法提出。

原始算法的思路非常直接：

> 找到最常出现的相邻 byte pair，用一个新的符号替代它，再继续重复。

也就是说：

\[
(a^*,b^*) = \arg\max_{a,b}Count(a,b)
\]

然后：

\[
a^*,b^*
\rightarrow
c
\]

Gage 当时讨论的核心是数据压缩，而不是词法学、词根、词缀或者语言理解。

二十多年以后，这种思想被稍作修改，成为 NLP 中的 Subword BPE。

举一个极简例子。语料：

```text
low
lower
lowest
```

初始拆成：

```text
l o w
l o w e r
l o w e s t
```

假设 `l + o` 频率很高，于是：

```text
l o → lo
```

再统计 `lo + w` 又很高：

```text
lo w → low
```

最后 vocabulary 里可能自然出现：

```text
low
er
est
```

这时候人类会产生一种很强的错觉：

> BPE 好像发现了语言学中的词根和后缀。

但请注意：

\[
\boxed{\text{BPE 根本不知道什么是词根}}
\]

它只知道：

\[
Count(a,b)
\]

所以这一点非常重要：

\[
\boxed{
\text{算法目标}
\neq
\text{人类解释}
}
\]

### 这里必须建立第一个重要框架：一种 Token，可以同时拥有三种完全不同的「意义」

例如 vocabulary 中出现：

```text
ing
```

人类语言学家看到以后说：

> 这是英语中一个非常典型的后缀。

这是：

\[
\boxed{\text{Human Interpretation}}
\]

但 BPE 当初为什么把它组合出来？因为：

```text
i + n
in + g
```

在语料中统计频率足够高。这是：

\[
\boxed{\text{Algorithmic Origin}}
\]

模型训练以后，Embedding(`ing`) 和上下文网络又可能学习到 progressive、gerund、verb morphology 等词法模式。这是：

\[
\boxed{\text{Learned Representation}}
\]

三个东西不能混淆：

\[
\boxed{
\text{人类赋予的意义}
\neq
\text{算法产生它的原因}
\neq
\text{模型最终学到的东西}
}
\]

这个区分不仅适用于 Tokenizer，理解整个深度学习都非常有用。

### WordPiece、BPE、Unigram，到底差在哪里？

这里不必陷入每个实现细节，只需要抓住它们所处的抽象层级。

WordPiece、BPE、Unigram 都是在解决：

\[
\boxed{\text{如何建立 Subword Vocabulary}}
\]

这个「战役级」问题。但它们采用不同的「战术」。

BPE 是：

\[
\text{small units}
\rightarrow
\text{不断 merge}
\rightarrow
\text{larger units}
\]

而 Unigram LM 基本是反方向：

\[
\text{large candidate vocabulary}
\rightarrow
\text{估计 piece 概率}
\rightarrow
\text{不断 prune}
\]

对于一种 segmentation：

\[
S=(t_1,\dots,t_n)
\]

可以定义：

\[
P(S) = \prod_i P(t_i)
\]

然后寻找高概率的 segmentation。

WordPiece 则采用另一种 vocabulary construction / segmentation 准则。Schuster 与 Nakajima 在 2012 年日语和韩语 Voice Search 工作中已经使用了后来被称为 WordPiece 的思想，说明 subword 思路事实上早于 BERT 很多年。

但从战略层看，它们并不是三个完全不同的问题。可以这样理解：

| 抽象层 | 问题 |
|---|---|
| 战略 | 怎样处理开放词汇，又避免序列过长？ |
| 战役 | 使用 Subword |
| 战术 | BPE / WordPiece / Unigram |
| 工程 | vocab size、pre-tokenization、实现速度、语言配比 |

这个层级非常重要。否则很容易陷入：

> BPE 和 WordPiece 哪个「更先进」？

实际上它们只是同一基本矛盾下不同的具体方案。

### SentencePiece 又是另一个很容易被误解的名字

SentencePiece 经常和 BPE、WordPiece、Unigram 并排列出来。严格来说不够准确。

SentencePiece 更像一个完整 tokenizer framework。它的重要贡献之一是：

> 可以直接从 raw sentence 上训练，不需要先依赖语言特定的 word segmentation。

这对于 English、中文、日本語、ไทย、العربية 这样的多语言场景尤其重要。

SentencePiece 可以使用不同模型，包括 BPE 和 Unigram。它强调从 raw sentences 进行语言无关的 subword tokenization。

这实际上体现了 Tokenizer 历史上的另一条趋势：

\[
\boxed{
\text{减少人为语言学 pipeline}
}
\]

过去：

```text
Raw Text
↓
Language-specific preprocessing
↓
Word segmentation
↓
Morphology
↓
Tokenizer
```

后来越来越希望：

```text
Raw Text
↓
Tokenizer
```

### 但是即使 Character 仍然没有彻底解决「开放世界」

因为 Unicode 字符本身也是一个很大的空间。

更重要的是，我们最终会发现：

> 计算机根本不需要把「字符」作为最底层世界。

文本在计算机里最终都可以变成：

\[
\boxed{\text{bytes}}
\]

以 UTF-8 为例：

```text
A
```

是 byte。而中文：

```text
人
```

编码成 UTF-8 后是多个 byte。任何文本最终都可以写成：

\[
b_1,b_2,\dots,b_N
\]

其中：

\[
b_i\in\{0,1,\dots,255\}
\]

这意味着：

\[
\boxed{|V_{\text{base}}|=256}
\]

就可以表示任意 byte sequence。

于是产生了一个极其漂亮的思想：

> 如果最底层单位是 byte，那么理论上根本不需要 `<UNK>`。

### Byte-level BPE 的本质：Universal Alphabet + Learned Compression

以 GPT-2 的 tokenizer 实现为例，它首先把 UTF-8 bytes 映射到可逆的 Unicode 表示，再在这些序列上执行 BPE；OpenAI 原始 GPT-2 encoder 代码甚至在 `bytes_to_unicode()` 的注释中明确讨论了避免大型字符词表和 unknown 的动机。

所以 Byte-BPE 可以抽象成：

\[
\text{Raw Text}
\xrightarrow{\text{UTF-8}}
\text{Bytes}
\xrightarrow{\text{BPE}}
\text{Variable-length Byte Sequences}
\]

例如：

```text
hello
```

UTF-8 中本来就是：

```text
68 65 6C 6C 6F
```

然后经过 BPE，常见组合不断 merge：

```text
h + e → he
l + l → ll
```

进一步可能形成：

```text
hello
```

于是对于高频字符串：

\[
N\downarrow
\]

对于从没见过的字符串，退化回 bytes，仍然可以表示。

因此 Byte-BPE 的本质可以概括成：

\[
\boxed{
\text{Universal Alphabet}
+
\text{Data-driven Compression}
}
\]

## 第三部分：Tokenizer 究竟在优化什么

### 这时候我们必须第一次停下来问：Tokenizer 到底在优化什么？

很多介绍会说：

> 一个好的 tokenizer 就应该尽量让 token 数少。

这听起来非常合理。如果同样一句文本：

Tokenizer A：

\[
N_A=20
\]

Tokenizer B：

\[
N_B=8
\]

显然 B 更便宜。但是：

\[
\boxed{\text{更短的 Token Sequence 是否一定带来更好的模型？}}
\]

答案并不是。

2024 年 EMNLP 的《Tokenization Is More Than Compression》专门系统研究了这个问题。作者设计了 PathPiece，能够在给定 vocabulary 下寻找 token 数量尽可能少的 segmentation，然后训练了大量不同 tokenizer 的模型。

结果并不支持：

\[
\text{fewer tokens}
\Rightarrow
\text{better downstream performance}
\]

这个朴素假设。论文进一步指出，pre-tokenization、vocabulary construction 和 segmentation 等因素同样重要。

更早的《Tokenization and the Noiseless Channel》也得到一个很有意思的结果：在机器翻译实验中，基于 Rényi entropy 的 token distribution efficiency 与 BLEU 的相关性很高，而单纯的压缩长度相关性反而很弱。

于是我们得到一个非常重要的认知：

\[
\boxed{
\text{Tokenizer 不只是 Compression}
}
\]

### 为什么不是「越压缩越好」？因为模型最终不是在存文件，而是在学习

压缩算法最关心：

\[
\text{Code Length}
\]

语言模型却还要关心：

\[
\text{Learnability}
\]

假设 vocabulary 中有一个 token：

```text
supercalifragilisticexpialidocious
```

它确实可以把很长字符串压缩成一个 token。但如果训练语料里只出现三次：

\[
Count(t)=3
\]

那么：

\[
Embedding(t)
\]

很难获得充分训练。

反过来，如果 token 太小：

```text
e
```

出现数十亿次，它虽然非常容易学习，但：

\[
\text{semantic specificity}\downarrow
\]

因此 token frequency distribution 本身存在一个 trade-off。太稀疏：

\[
\text{insufficient learning}
\]

太高频：

\[
\text{poor discriminativeness}
\]

好的 tokenizer 并不是简单追求：

\[
N\rightarrow\min
\]

它需要同时考虑：

\[
\boxed{
\text{Compression}
+
\text{Statistical Efficiency}
+
\text{Representation Quality}
}
\]

## 第四部分：中文——观察 Tokenizer 的显微镜

### 中文，是研究 Tokenizer 最好的「显微镜」

为什么中文特别有意思？因为中文几乎同时暴露了 Tokenizer 的所有矛盾。

首先，中文没有英语那样明显的空格边界：

```text
人工智能正在改变世界
```

到底是：

```text
人工智能 / 正在 / 改变 / 世界
```

还是：

```text
人工 / 智能 / 正在 / 改变 / 世界
```

这是第一层。

第二层：中文的「字」本身通常已经有一定意义：

```text
人
工
智
能
```

第三层：汉字内部还有结构：

```text
氵
艹
口
木
马
```

即偏旁、部首、声旁、字形。

第四层：计算机底层又把一个汉字表示为多个 UTF-8 bytes。

因此中文同时存在 word、character、sub-character、byte 四个非常自然的尺度：

\[
\text{word}
\quad
\text{character}
\quad
\text{sub-character}
\quad
\text{byte}
\]

这也是为什么中文 Tokenization 不是「英文 Tokenization 换个语言」那么简单。

### BERT 时代有一个非常有代表性的选择：中文接近按字切

经典 Chinese BERT 中，每个汉字基本可以被看作一个独立单位。所以：

```text
人工智能
```

接近：

```text
[人][工][智][能]
```

这种方案非常稳健。因为：

\[
\text{OOV}\downarrow
\]

也不需要先解决复杂的中文分词。

但它也带来一个问题。模型必须重新从 `人`、`工`、`智`、`能` 组合出 `人工智能` 这个整体概念。

因此后来又出现 Whole Word Masking。需要特别强调：

> Whole Word Masking 并不意味着「人工智能」变成一个 token。

它仍然可能是：

```text
[人][工][智][能]
```

只是 Mask 时一起处理：

```text
[MASK][MASK][MASK][MASK]
```

希望训练目标迫使模型更多地学习 word-level information。

### 中文甚至还可以进一步往「字里面」走

2023 年 TACL 的《Sub-Character Tokenization for Chinese Pretrained Language Models》提出：

> 传统中文 tokenizer 往往把汉字看成不可再分的基本单位，但汉字内部其实还包含字形和语音信息。

因此他们把汉字先编码成 glyph/stroke-like sequence 与 pronunciation representation，再在这些表示上执行 subword segmentation。

实验显示，这样做可以得到更短的序列，同时 pronunciation-based tokenizer 对同音 typo 更鲁棒。

这件事情非常值得思考。因为：

> 汉字到底是不是「原子」？

对于人类语言学而言，未必。对于传统 BERT 工程而言：

\[
\text{Yes}
\]

对于 SubChar：

\[
\text{No}
\]

所以所谓：

\[
\boxed{\text{atomic unit}}
\]

本身就是一种建模选择。

### 所谓「中文税」，第一层其实只是一个非常朴素的工程问题

假设同样的信息量，英文被 tokenizer 编成：

\[
10\ tokens
\]

中文被编成：

\[
18\ tokens
\]

那么同样：

\[
128K
\]

token 的 context window，对两种语言实际能够容纳的自然语言内容并不完全相等。

而推理时 latency、KV cache、FLOPs，以及按 token 计费的 API cost，也都会受到影响。

所以所谓「中文税」，最直接的意义是：

\[
\boxed{\text{Tokenization Efficiency Tax}}
\]

但这里一定不能把工程结果误解成语言本体。不能因为中文在某个 tokenizer 下需要更多 token，就直接得出：

> 中文天然比英文低效。

真正关系更接近：

\[
\boxed{
\text{Language}
\times
\text{Training Corpus}
\times
\text{Vocabulary Budget}
\times
\text{Tokenizer Algorithm}
}
\]

如果训练语料里有大量高质量中文，`人工智能` 频繁出现，BPE 很可能逐渐学出：

```text
[人工智能]
```

或者：

```text
[人工] [智能]
```

如果中文在 tokenizer 训练数据里非常弱，那么它可能只能退化成：

```text
[人][工][智][能]
```

甚至更细。

所以 Vocabulary 实际上在做一件非常现实的事情：

\[
\boxed{\text{给不同语言和模式分配有限的「压缩预算」}}
\]

### 但是 2025 年的一项中文研究让问题突然变得更深了

如果中文 Tokenizer 的问题仅仅是成本，那么最直接的目标似乎就是：

\[
\boxed{\text{尽量把中文压成更少的 Token}}
\]

然而 2025 年发表于 *Computational Linguistics* 的论文 **Tokenization Changes Meaning in Large Language Models: Evidence from Chinese** 给出了一个非常漂亮的反例。

作者研究的不只是：

> 中文用了多少 token。

而是：

> **Token boundary 是否会改变模型的语义判断？**

答案是：

\[
\boxed{\text{会}}
\]

论文利用了汉字、Unicode、UTF-8 与 BPE 之间一个非常特殊但真实存在的结构关系。

### 一个「本来毫无语言意义」的 Byte Pattern，竟然可以变成语义 Shortcut

例如汉字 `茶` 和 `茎` 都含有 `艹` 这样的语义部件。

UTF-8 编码中，某些具有相同部首的汉字，前几个 bytes 也存在很强统计相关性。

而 BPE 本质上是在做：

\[
\text{frequent byte strings}
\rightarrow
\text{token}
\]

于是某些 token 恰好对应：

```text
多个拥有相同偏旁的汉字所共享的 byte prefix
```

注意：这个 token **不是设计者刻意定义的「艹 token」**。算法根本不知道 `艹` 是什么意思。它只是因为：

\[
Count(byte\ sequence)
\]

足够高，所以成为 token。

但训练完成以后，模型发现这个 token 与某类汉字高度相关。

于是：

\[
\boxed{
\text{Engineering Encoding}
\rightarrow
\text{Statistical Correlation}
\rightarrow
\text{Learned Representation}
}
\]

就发生了。这是一条极其重要的因果链。

### 这篇论文最精彩的地方，是它真的验证了模型会被这种 Token 边界「欺骗」

作者构造了控制实验。两个汉字可以真正具有相同 semantic radical，或者没有相同 radical；与此同时它们又可能共享 initial token，或者不共享。于是形成一个经典的 \(2\times2\) 实验。

理论上，是否相同部首应该由汉字结构决定。但实验发现：

> 如果两个汉字共享某个 token，模型会系统性地更容易判断它们在语言学结构上也更加相似。

GPT-4、GPT-4o 和 Llama 3 都表现出了这种影响。

更有意思的是：

> 在一些实验里，一个汉字被编码成单个、更长的 token，模型反而比多 token 表示表现更差。

作者据此指出，单纯追求更少、更长 token，并不能解决 tokenization 带来的表征问题。

这给出了一个极其反直觉的结论：

\[
\boxed{
\text{Compression Better}
\not\Rightarrow
\text{Representation Better}
}
\]

### 这时候我们终于可以明确区分四件经常被混在一起的东西

假设一个 Token 是：

```text
人工智能
```

第一种问题是：

> 人类认为它是不是一个完整概念？

这是 Human Semantics。

第二种问题是：

> 算法为什么把它变成一个 Token？

可能只是 \(Count(\text{人工智能})\) 很高。这是 Algorithmic Objective。

第三种问题是：

> 模型通过训练以后，这个 Token 的 representation 学到了什么？

这是 Learned Semantics。

第四种问题是：

> 把它整体作为一个 Token，对 latency、sequence length 和模型效果有什么影响？

这是 Engineering Consequence。

所以：

\[
\boxed{
\text{Human Meaning}
\neq
\text{Algorithmic Reason}
\neq
\text{Learned Meaning}
\neq
\text{Engineering Utility}
}
\]

我认为这是理解 Tokenizer 时最值得建立的框架之一。

## 第五部分：概率建模与序贯决策视角

### 现在再问一个更大的问题：为什么语言模型一定要 Tokenize？

到这里，我们一直在讲：一个词应该怎么切。但这仍然停留在 Tokenizer 内部。

真正理解 Token，需要回到语言模型究竟在建模什么。

一个生成式语言模型最根本的目标，是建模某种数据分布：

\[
P(x)
\]

假设 tokenizer：

\[
T
\]

把原始字符串：

\[
x_{\text{raw}}
\]

映射成：

\[
(x_1,x_2,\dots,x_T)
\]

那么模型实际建模的是：

\[
P(x_1,x_2,\dots,x_T)
\]

利用概率链式法则：

\[
\boxed{ P(x_1,\dots,x_T) = \prod_{t=1}^{T} P(x_t|x_{\lt t}) }
\]

注意：这不是 Transformer 发明的。这是概率论的链式分解。

真正的工程选择是：

\[
\boxed{x_t\text{ 到底是什么}}
\]

它可以是 word，可以是 subword，可以是 character，也可以是 byte。

于是我们突然发现：

> Tokenizer 实际上定义了 \(P(x)\) 进行自回归分解时的**离散粒度**。

### 从这里开始，Token 就不再只是「语言单位」

假设：

```text
人工智能
```

Tokenizer A：`[人工智能]`，那么：

\[
T=1
\]

只需要建模一次：

\[
P(\text{人工智能}|context)
\]

Tokenizer B：`[人工][智能]`，则：

\[
P(\text{人工智能}|context) = P(\text{人工}|context) P(\text{智能}|context,\text{人工})
\]

Tokenizer C：`[人][工][智][能]`，则：

\[
P(x) = P(\text{人}|s_0) P(\text{工}|s_1) P(\text{智}|s_2) P(\text{能}|s_3)
\]

原始信息没有变。但概率分解的路径变了。这意味着：

\[
\boxed{
\text{Tokenization defines the temporal granularity of autoregressive modeling}
}
\]

### 更进一步：自回归语言模型可以被写成一个极其简单的序贯决策过程

定义状态：

\[
s_t=x_{\lt t}
\]

也就是当前已经生成出来的完整 Token 前缀。

定义动作：

\[
a_t=x_t
\]

即下一步选择哪个 Token。那么：

\[
\pi_\theta(a_t|s_t) = P_\theta(x_t|x_{\lt t})
\]

状态转移则极其简单：

\[
s_{t+1} = s_t\oplus a_t
\]

其中 \(\oplus\) 表示 append。例如当前状态 `[人工]`，模型执行动作 `[智能]`，新状态就是 `[人工][智能]`。

所以 transition：

\[
P(s_{t+1}|s_t,a_t)
\]

实际上是确定性的。只要：

\[
s_{t+1}=s_t\oplus a_t
\]

概率就是 1。

### 这里要特别严谨：Pretraining 不是强化学习

虽然我们可以把自回归生成表示成 MDP-like sequential decision process，但不能由此说：

> 大模型预训练就是 RL。

标准 next-token pretraining 是：

\[
\max_\theta
\sum_t
\log
P_\theta(x_t^*|x_{\lt t}^*)
\]

训练数据已经告诉模型：

\[
a_t^*=x_t^*
\]

应该输出什么。所以它更接近：

\[
\boxed{\text{Maximum Likelihood / Behavior Cloning}}
\]

而 RL 是：

\[
a_t\sim\pi_\theta(\cdot|s_t)
\]

模型自己 rollout，得到：

\[
\tau=(s_0,a_0,s_1,a_1,\dots)
\]

最后根据：

\[
R(\tau)
\]

优化策略。因此：

\[
\boxed{
\text{Autoregressive Process 可以表述为 MDP}
\neq
\text{Pretraining 是 RL}
}
\]

### 为什么它仍然满足 Markov 形式？

很多人看到：

\[
s_t=x_{\lt t}
\]

会问：

> 语言显然依赖非常久以前的信息，怎么能叫 Markov？

这里的关键是：Markov property 并不是要求：

\[
s_t=x_{t-1}
\]

而是要求 \(s_t\) 包含预测未来所需要的全部 relevant history。

如果我们直接定义：

\[
s_t=(x_1,\dots,x_{t-1})
\]

整个历史已经包含在 state 里。于是：

\[
P(s_{t+1}|s_0,a_0,\dots,s_t,a_t) = P(s_{t+1}|s_t,a_t)
\]

自然成立。也就是说：

\[
\boxed{
\text{把完整 History 纳入 State，就可以把历史依赖过程 Markov 化}
}
\]

真正的难题变成：

> 如何把越来越长的 prefix，有效编码成模型内部的 representation？

Transformer 做的正是：

\[
x_{\lt t}
\rightarrow
h_t
\]

再根据 \(h_t\) 预测：

\[
P(x_t|x_{\lt t})
\]

### 这时 Tokenizer 的战略意义突然变了：它在定义 Action Space

如果：

\[
a_t=x_t
\]

那么模型的 action space 就是：

\[
\mathcal A=\mathcal V
\]

即：

\[
\boxed{\text{Vocabulary 就是 Action Space}}
\]

假设 vocabulary size：

\[
|\mathcal V|=100000
\]

模型每一步实际上是在十万个候选动作上形成：

\[
\pi_\theta(a|s)
\]

然后采样或选择一个。因此 Vocabulary Size 不只是 embedding matrix 多大。它还决定：

\[
\boxed{\text{每一步决策能够选择多大的 Macro Action}}
\]

### Tokenizer 同时定义了第二件事：Horizon

还是 `人工智能`。

Tokenizer A：`[人工智能]`，需要：

\[
H=1
\]

Tokenizer B：`[人工][智能]`，需要：

\[
H=2
\]

Tokenizer C：`[人][工][智][能]`，需要：

\[
H=4
\]

于是经典的：

\[
\text{Vocabulary Size}
\leftrightarrow
\text{Sequence Length}
\]

可以用一个更深的决策论语言重新表达：

\[
\boxed{
\text{Action Space Size}
\leftrightarrow
\text{Decision Horizon}
}
\]

词表越大：

\[
|\mathcal A|\uparrow
\]

每一步可以走得越远：

\[
H\downarrow
\]

词表越小：

\[
|\mathcal A|\downarrow
\]

动作越原子：

\[
H\uparrow
\]

### 这和 Hierarchical RL 中 Primitive Action / Macro Action 极其相似

字符 `[a][r][t][i][f]...` 类似：

\[
\text{primitive actions}
\]

Subword `[artificial]`、`[intelligence]` 更接近：

\[
\text{macro actions}
\]

如果 `[artificial intelligence]` 整个成为一个 Token，则是更大的：

\[
\text{macro action}
\]

这并不意味着 Tokenizer 就是 Hierarchical RL。但这个类比揭示了一个重要结构：

\[
\boxed{
\text{Tokenization 本质上也在选择决策时间尺度}
}
\]

一步迈得小：灵活，但步数多。一步迈得大：步数少，但 action space 更大，而且内部结构被隐藏。

### 这也解释了一个容易被忽略的事实：Action 会直接成为下一时刻的 State

普通控制系统中：

\[
a_t
\]

影响环境。而语言模型的特殊之处在于：

\[
a_t=x_t
\]

直接被写入未来的 state。即：

\[
s_{t+1} = [s_t,a_t]
\]

下一步：

\[
s_{t+2} = [s_t,a_t,a_{t+1}]
\]

所以：

\[
\boxed{\text{Action becomes State}}
\]

这意味着一个早期错误会永久进入后续条件分布。如果模型开始写：

```text
因为 17 × 6 = 92
```

那么后面所有推理都在条件 `17 × 6 = 92` 之上继续。这就是 autoregressive error propagation 的基本结构之一。

### SFT 和 On-policy RL 的差异，在这个状态视角下也异常清晰

SFT 训练时：

\[
s_t^{train}=x_{\lt t}^{GT}
\]

也就是状态来自 ground-truth trajectory。模型学习：

\[
\pi(a_t^*|s_t^{GT})
\]

但推理时：

\[
s_t^{test} = \hat x_{\lt t}
\]

来自模型自己之前生成的 action。所以：

\[
d_{\text{data}}(s)
\neq
d_{\pi}(s)
\]

这就是经典的：

\[
\boxed{\text{State Distribution Mismatch}}
\]

而 on-policy RL、on-policy distillation 等方法的重要价值之一，正是：

\[
s\sim d_{\pi_\theta}
\]

在模型自己真正会访问的状态空间中学习。

因此 Tokenizer → State → Action → On-policy，这些原本看似不同的话题，其实可以被一条统一的序贯建模逻辑连接起来。

### 甚至可以从这个角度重新理解 Chain-of-Thought

假设模型直接做：

\[
Question
\rightarrow
Answer
\]

例如：

```text
Q → 42
```

映射很短。而 Chain-of-Thought 引入：

\[
z_1,z_2,\dots,z_k
\]

于是：

\[
Question
\rightarrow
z_1
\rightarrow
z_2
\rightarrow
\dots
\rightarrow
Answer
\]

模型实际上获得更多：

\[
\boxed{\text{Intermediate Computational Steps}}
\]

因此可以得到一个很有意思的统一视角：

\[
\boxed{
\text{Tokenizer 决定基础计算步有多细；Reasoning 决定模型愿意使用多少计算步。}
}
\]

Tokenization 控制：

\[
\text{step granularity}
\]

Reasoning length 控制：

\[
\text{number of steps}
\]

二者共同影响：

\[
\boxed{\text{Test-time Computation}}
\]

## 第六部分：Token 的多重身份

### 到这里，我们终于可以回答：Token 到底是什么？

如果从语言学看：

\[
\text{Token}\approx\text{linguistic unit}
\]

如果从压缩看：

\[
\text{Token}\approx\text{frequent code}
\]

如果从概率建模看：

\[
\text{Token}\approx\text{autoregressive factorization unit}
\]

如果从 RL-like sequential process 看：

\[
\text{Token}\approx\text{action}
\]

如果从 Transformer 工程看：

\[
\text{Token}\approx\text{unit receiving expensive computation}
\]

于是所谓 Token，其实是多个抽象层的交叉点。这也是为什么围绕 Tokenizer 的争论经常鸡同鸭讲：大家讨论的根本不是同一个层级。

### 从「战略—战役—战术」看 Tokenizer，会清晰很多

我们可以重新整理整个问题。

| 层级 | Tokenizer 中真正的问题 |
|---|---|
| 战略 | 模型应该以什么粒度观察并操作信息？ |
| 战役 | Word / Subword / Byte / Neural Token / Dynamic Token |
| 战术 | BPE / WordPiece / Unigram / VQ / RVQ / Entropy Patching |
| 工程 | vocab、序列长度、显存、KV、延迟、吞吐 |
| 行为 | 模型最终学到了语义、组合结构还是 shortcut？ |

因此：

> BPE 是不是比 WordPiece 高级？

属于战术问题。而：

> 我们究竟应该让模型处理固定 Token，还是直接处理 Byte？

已经是战役问题。再往上：

> 模型为什么一定需要固定的离散时间尺度？

这才是战略问题。

## 第七部分：动态计算与多模态统一

### 然后历史开始出现一个有意思的反转：也许根本不需要传统 Tokenizer

既然所有文本最终都是 bytes，那么最自然的问题就是：

> 为什么不直接训练 Byte-level Language Model？

ByT5 就是这个方向的重要代表。它直接处理 UTF-8 bytes，而不使用传统 subword vocabulary。这带来几个明显好处：

- 任意语言天然可表示；
- 没有传统 OOV；
- 对 noise 更鲁棒；
- 减少复杂 tokenizer preprocessing。

但代价也非常明确：

\[
N_{\text{byte}}
\gg
N_{\text{subword}}
\]

论文也明确将长 byte sequence 带来的训练 FLOPs 和 inference speed 视为核心 trade-off。

所以问题再次回到：

\[
\boxed{\text{计算粒度}}
\]

### BLT 的关键突破：为什么 Token Boundary 必须在训练前固定？

传统 BPE：

\[
T(x)
\]

基本由一个固定 vocabulary 和 merge rule 决定。训练完 tokenizer 后，某段字符串在哪里切，基本就固定了。

Byte Latent Transformer 提出了一个非常漂亮的新问题：

> 如果一段信息非常简单，为什么要和困难区域获得一样细的计算粒度？

BLT 从 raw bytes 出发，但不会把每一个 byte 都交给昂贵的 Transformer 主干。它根据下一 byte 的 entropy 动态形成不同长度 patch。

容易预测的区域：

\[
H(b_{t+1}|b_{\le t})\downarrow
\]

可以形成更长 patch。复杂区域：

\[
H(b_{t+1}|b_{\le t})\uparrow
\]

切得更细。也就是：

\[
\text{Predictable Region}
\rightarrow
\text{Less Compute}
\]

\[
\text{Difficult Region}
\rightarrow
\text{More Compute}
\]

BLT 把 dynamically sized patches 作为主要计算单位，并通过 entropy 分配计算资源。

这里发生了一次非常重要的概念跃迁：

\[
\boxed{
\text{Tokenizer 从 Representation Problem 开始变成 Compute Allocation Problem}
}
\]

### 这不是文本领域独有的变化，多模态几乎走了一模一样的路

现在把视角从文字移到图片。一张图片：

\[
I\in\mathbb R^{H\times W\times3}
\]

并没有 word boundary，也没有 character。那么 Transformer 怎么处理图片？

ViT 给出了一个极其简单甚至有点「暴力」的答案：切格子。

例如 \(224\times224\) 图片，切成 \(16\times16\) patch。那么：

\[
14\times14=196
\]

个 patch。每个 \(16\times16\times3\) 的像素块经过线性投影以后变成一个 embedding。

ViT 的原论文正是直接把 image patches 当作 sequence 输入 Transformer。

### 但是：世界真的天然由 16×16 小方块组成吗？

当然不是。所以：

\[
\boxed{\text{ViT Patch 不是视觉中的「词」}}
\]

它只是：

\[
\boxed{\text{Engineering Discretization}}
\]

为什么是 \(16\times16\) 而不是 \(8\times8\) 或者 \(32\times32\)？本质仍然是：

\[
\text{Information Granularity}
\leftrightarrow
\text{Compute Cost}
\]

因为：

\[
N=\frac{HW}{P^2}
\]

Patch 越小：

\[
P\downarrow
\Rightarrow
N\uparrow
\]

细节更多，但计算更贵。Patch 越大：

\[
P\uparrow
\Rightarrow
N\downarrow
\]

计算便宜，但更多局部结构被提前压缩。

你会发现，这和 `character vs word` 几乎是同一个问题。

### 但 ViT 的 Patch 仍然不是真正意义上的「离散 Token」

文本 Token：

\[
x_t\in\{1,\dots,|V|\}
\]

例如：

```text
token_id = 31872
```

而 ViT Patch 经过 projection 后是：

\[
h_i\in\mathbb R^d
\]

这是连续向量。因此：

\[
\boxed{
\text{Visual Token}
\neq
\text{Discrete Visual Token}
}
\]

很多多模态讨论里会把两者混在一起。LLaVA、CLIP+LLM 这类架构中的 visual tokens，大量属于 continuous embeddings，而不是 vocabulary IDs。

### VQ-VAE 才真正让图片获得了类似「词表」的东西

VQ-VAE 的关键思想，是加入：

\[
\boxed{\text{Codebook}}
\]

假设 encoder：

\[
E(x)=z
\]

然后准备：

\[
\mathcal C=
\{e_1,e_2,\dots,e_K\}
\]

对于每个 latent \(z_i\)，寻找最近的 code：

\[
k_i = \arg\min_k \|z_i-e_k\|^2
\]

于是：

\[
z_i
\rightarrow
e_{k_i}
\]

图片终于可以表示成：

```text
[327] [91] [91] [812] [15] ...
```

VQ-VAE 2017 的核心之一，正是通过 vector quantization 学习离散 latent representations，并进一步用 autoregressive prior 对这些离散 code 建模。

现在：

\[
k_i\in\{1,\dots,K\}
\]

形式上已经和文字完全一样。

### 这里要再次警惕人类的「过度解释」

假设我们观察 `visual token 327`，发现它经常出现在 `grass` 附近。很容易说：

> 327 就是「草地 token」。

这句话过头了。更准确的是：

\[
P(\text{grass-like patterns}|k=327)
\]

可能很高。Code 327 之所以存在，是因为 quantization：

\[
\arg\min_k\|z-e_k\|^2
\]

而不是人类事先规定 `327 = grass`。

所以多模态领域再次重复了 BPE 的故事：

\[
\boxed{
\text{Code Identity}
\neq
\text{Human Concept}
}
\]

### 文本和视觉 Tokenizer 的一个重大区别：文本通常要求无损，视觉天然允许有损

文本：

```text
hello
```

Tokenize 再 Decode 通常希望：

\[
D(T(x))=x
\]

精确恢复。图片不一样。一张 \(1024\times1024\) 图片如果被压缩成几百个 discrete visual tokens：

\[
T(I)
\]

Decoder：

\[
D(T(I))=\hat I
\]

一般只能希望：

\[
\hat I\approx I
\]

所以视觉 Tokenizer 同时面对：

\[
\boxed{
\text{Compression}
+
\text{Reconstruction}
+
\text{Semantic Preservation}
}
\]

三个目标。而且 Pixel Fidelity 和 Semantic Utility 并不完全一致。

### 这就出现了视觉领域一个非常核心的矛盾：理解和生成，到底需要同一种 Token 吗？

做 image understanding 时，重要的是：

```text
这是猫
猫坐在桌子上
桌子在窗户左边
```

微小 JPEG 噪声未必重要。也就是更关注：

\[
\text{semantic information}
\]

但是 image generation 要还原纹理、颜色、毛发、光影、局部边缘。需要：

\[
\text{low-level fidelity}
\]

因此：

\[
\boxed{
\text{Understanding Representation}
\neq
\text{Generation Representation}
}
\]

这也是为什么多模态统一并不像「把所有东西都强行变成一个 Tokenizer」那么简单。

### 但是统一 Next-Token Prediction 仍然具有巨大的诱惑

因为一旦 Text、Image、Video、Audio 都可以变成：

\[
[s_1,s_2,\dots,s_N]
\]

那么整个训练目标可以统一成：

\[
\boxed{ \mathcal L = -\sum_t \log P_\theta(s_t|s_{\lt t}) }
\]

这正是 Emu3 一类工作的核心吸引力。Emu3 将 text、image 和 video tokenize 到离散空间，然后从头训练单一 Transformer，仅依赖 next-token prediction，同时完成生成和感知任务。

于是文字、图片、视频从模型角度都逐渐变成：

\[
\boxed{\text{Sequence}}
\]

### 视觉 Tokenizer 的质量甚至可能决定 Autoregressive Vision 能不能成立

MAGVIT-v2 的标题非常直接：**Language Model Beats Diffusion: Tokenizer is Key to Visual Generation**

它的核心判断就是：

> 如果视觉 Tokenizer 能产生足够 concise、expressive 的离散 Token，那么 Language Model 式的生成范式可以变得非常强。

MAGVIT-v2 构建了能够同时服务 image 和 video 的 visual tokenizer，并强调 tokenizer quality 对 autoregressive visual generation 的关键作用。

这其实再次说明：

\[
\boxed{\text{Tokenizer 不是模型前面一个无关紧要的 preprocessing}}
\]

它决定：主模型究竟在什么表示空间里学习。

### 然后视觉领域也开始走向「动态 Token 数」

传统 \(256\times256\) image，无论内容是什么，可能都固定：

\[
N=256
\]

tokens。但一张图片可能只是：

```text
白色背景 + 一个红色圆
```

另一张则是：

```text
纽约街景
+ 几十个人
+ 汽车
+ 广告牌文字
+ 建筑
+ 复杂光影
```

为什么它们都应该获得 256 个 token？FlexTok 给出的答案是：不应该。

FlexTok 可以把一张 \(256\times256\) 图片表示为 1 到 256 个可变长度的离散 token，并形成 coarse-to-fine 的有序 1D visual token sequence。

于是：

\[
\text{Simple Image}
\rightarrow
N\downarrow
\]

\[
\text{Complex Image}
\rightarrow
N\uparrow
\]

这和 BLT 几乎形成了完美呼应。

### 文本 Tokenizer 与视觉 Tokenizer，正在走向同一个终点

文本的发展：

```text
Word
↓
Character
↓
Subword
↓
Byte
↓
Dynamic Byte Patch
```

视觉的发展：

```text
Pixel
↓
Fixed Patch
↓
Learned Latent
↓
Discrete VQ Token
↓
Variable-length Token
```

起点完全不同。但终点却越来越像：

\[
\boxed{
\text{Information-dependent Compute Allocation}
}
\]

换句话说，过去的问题是：

> 什么是一个 Token？

现在的问题开始变成：

> **这部分信息值得多少个 Token？**

更进一步：

> **这部分信息值得多少计算？**

## 第八部分：技术史、哲学与未来

### 所以我们需要重新理解整个 Tokenizer 的技术史

如果按照算法名字看，历史很乱：

```text
WordPiece
BPE
Unigram
SentencePiece
Byte-BPE
ByT5
VQ-VAE
MAGVIT
BLT
FlexTok
...
```

如果按照「解决什么根本矛盾」来看，则非常清晰。

第一阶段：

\[
\boxed{\text{Linguistic Unit}}
\]

人类问：什么是词？什么是词素？

第二阶段：

\[
\boxed{\text{Statistical Unit}}
\]

机器问：什么 pattern 经常一起出现？

第三阶段：

\[
\boxed{\text{Universal Code}}
\]

工程师问：能不能用 byte 覆盖所有输入？

第四阶段：

\[
\boxed{\text{Learned Discrete Representation}}
\]

神经网络问：图像、声音这些连续信号能不能自己学出 codebook？

第五阶段：

\[
\boxed{\text{Adaptive Compute Unit}}
\]

模型开始问：不同区域为什么一定获得一样细的计算粒度？

### 如果再向上抽象，可以看到三种完全不同的 Token 世界观

第一种，可以称为：

\[
\boxed{\text{Linguistic Token View}}
\]

Token 应该尽量对应 word、morpheme、radical，即 Token 是意义单位。这是早期 NLP 最自然的思想。

第二种是：

\[
\boxed{\text{Compression Token View}}
\]

Token 不需要人类可解释。只要 efficient code 就可以。BPE、Byte-BPE、VQ codebook 都明显具有这一特征。

第三种是：

\[
\boxed{\text{Computation Token View}}
\]

Token 最重要的意义是：它是进入昂贵模型主体的一次计算机会。从这个视角：

\[
\text{Token}
\approx
\text{Unit of Expensive Computation}
\]

BLT、动态视觉 Tokenizer 尤其体现了这种变化。

### 这也是为什么「一个 Token 是不是一个完整的词」越来越不是最重要的问题

例如 `人工智能`。如果整体变成一个 Token：人类觉得很符合语义，工程上：

\[
N\downarrow
\]

也很好。但模型层面却必须继续问：是否丢掉了有价值的内部 compositional structure？

反过来 `人 工 智 能`，虽然序列变长，但：

\[
\text{substructure is explicit}
\]

所以并不存在：

\[
\boxed{ \text{Human Interpretability} = \text{Machine Utility} }
\]

Haslett 的中文实验恰恰提供了一个非常漂亮的证据：人类看起来怪异的多 Token / byte-level 切分，有时候反而暴露了模型可以利用的信息。

### 因此一个好的 Tokenizer 至少需要从五个不同维度评价

最底层是：

\[
\text{Coding Correctness}
\]

文本能不能可靠编码、解码：

\[
D(T(x))=x
\]

第二层：

\[
\text{Coverage}
\]

有没有 `<UNK>`。

第三层：

\[
\text{Compression Efficiency}
\]

相同信息需要多少 token。

第四层：

\[
\text{Statistical / Representation Quality}
\]

这些 token 是否获得足够学习，同时又保留有价值结构。

第五层：

\[
\text{Task Utility}
\]

最终 Accuracy、Perplexity、Reasoning、Robustness 到底怎么样。

而这几个目标：

\[
\boxed{\text{并不保证同向}}
\]

### 再往哲学层走一步：世界到底有没有天然的「基本单位」？

这是整个问题最有意思的地方。

对于 `New York`，你可以认为基本单位是 `New`、`York`，也可以认为 `New York` 作为一个实体更合理。

对于 `人工智能`，你可以讨论 `人工 / 智能`，也可以讨论 `人 / 工 / 智 / 能`，甚至 `偏旁 / 笔画`。

每个层次都有自己的规律。所以：

\[
\boxed{\text{世界往往不存在唯一正确的离散尺度}}
\]

机器学习所做的，是选择一种：

\[
\boxed{\text{Inductive Bias}}
\]

把原始世界转换成某个计算坐标系。

### 换句话说：Tokenizer 不只是在表示世界，它也在规定模型能够怎样操作世界

如果 `人工智能` 永远是一个不可拆的 Token：

\[
t_{\text{AI}}
\]

那么模型第一层得到的是：

\[
e_{\text{AI}}
\]

它无法直接在 Token 层操作 `人`、`工`、`智`、`能` 内部结构。这些信息只能通过更间接的学习恢复。

反过来，如果全部字符级 `人 工 智 能`，那么模型必须自己学习：

\[
\text{Composition}
\]

才能形成 `人工智能` 这个高层概念。因此：

\[
\boxed{
\text{Representation determines what distinctions are explicit}
}
\]

Tokenization 本身就是一种归纳偏置。

### 到这里，我们可以给 Tokenizer 一个比「分词器」更准确的定义

传统定义：

\[
\text{Tokenizer}:
\text{Text}\rightarrow\text{Token IDs}
\]

已经不足以描述今天的问题。更一般地：

\[
\boxed{
T:x_{\text{raw}}
\rightarrow
(z_1,z_2,\dots,z_N)
}
\]

其中 Tokenizer 实际同时决定：

\[
\boxed{N}
\]

需要多少计算步；决定：

\[
\boxed{\text{Granularity}}
\]

每一个单位包含多少信息；决定：

\[
\boxed{\mathcal A}
\]

自回归模型每一步可以选择的动作集合；还决定：

\[
\boxed{\text{Information exposed to the model}}
\]

哪些结构一开始就显式存在。

所以我更愿意把 Tokenizer 定义成：

\[
\boxed{ \text{Tokenizer} = \text{Representation Interface} + \text{Compression Interface} + \text{Compute Interface} }
\]

### 这也解释了为什么 Tokenizer 看似是「小技术」，其实处在整个大模型最底层的战略位置

我们今天经常讨论：

```text
Scaling Law
RL
Reasoning
Agent
Memory
Tool Use
World Model
```

这些当然重要。但所有这些能力最终都建立在：

\[
\text{model can perceive and manipulate certain units}
\]

之上。Tokenization 决定的是：

\[
\boxed{\text{模型世界的坐标系}}
\]

如果坐标系设计得不好：

- 一个语言被碎成大量 token；
- 数字被奇怪切分；
- spelling information 被隐藏；
- 一个汉字内部结构被压没；
- 一张图片被固定切成数千个 patch；
- 视频产生几十万 token；

那么后面再强大的 Transformer，都需要在这个坐标系之上付出代价。

### 所以 Tokenizer 的发展史，其实也是 AI 对「基本计算单位」认识不断变化的历史

1950 年代，人类说：基本单位就是词、语法和意义。

后来统计方法说：不需要完全相信语言学，让数据决定什么 pattern 有用。

BPE 进一步说：我甚至不要求这个单位有人类意义，高频就可以。

Byte-level 方法说：连字符都不需要是基本单位，256 个 byte 就足够描述所有文本。

VQ-VAE 说：连图片也可以学习一个离散 codebook。

Emu3 一类模型说：文本、图片和视频可以统一成为 sequence，然后做 next-token prediction。

BLT 和 FlexTok 又进一步说：连 Token boundary、Token 数量都未必应该提前固定。

因此整条历史可以浓缩成：

\[
\boxed{
\text{人定义基本单位}
\rightarrow
\text{数据发现基本单位}
\rightarrow
\text{模型学习基本单位}
\rightarrow
\text{模型动态决定基本单位}
}
\]

### 如果要再向未来推一步，我认为真正的问题已经不是「BPE 会不会被淘汰」

这个问题太小了。真正的问题是：

\[
\boxed{\text{未来模型还会不会拥有固定 Token Clock？}}
\]

今天绝大多数 Transformer 相当于拥有一个固定时钟：

```text
读一个 token
算一次
再读一个 token
再算一次
```

无论是：

```text
aaaaaaaaaaaaaaaa
```

还是：

```text
一个复杂数学证明中的关键推导
```

只要 Tokenizer 给出一个 token，它们都获得一次基本计算机会。但信息的复杂度显然并不均匀。

未来更自然的系统可能是：

\[
\text{easy information}
\rightarrow
\text{coarse representation}
\rightarrow
\text{little compute}
\]

\[
\text{hard information}
\rightarrow
\text{fine representation}
\rightarrow
\text{more compute}
\]

于是：

\[
\boxed{
\text{Tokenization}
\rightarrow
\text{Adaptive Computation}
}
\]

可能才是更加重要的长期方向。

### 这时候 Tokenizer、Reasoning 与 Agent 甚至开始出现统一解释

Tokenizer 在最底层决定：

\[
\text{micro-step}
\]

Reasoning 决定：

\[
\text{需要产生多少 intermediate steps}
\]

Agent Loop 则进一步决定：

\[
\text{是否需要跨模型调用、工具调用和环境交互继续计算}
\]

于是形成一个有趣的层级：

\[
\boxed{
\text{Token}
\rightarrow
\text{Reasoning Step}
\rightarrow
\text{Tool/Agent Step}
}
\]

都在回答同一个更加抽象的问题：

> **面对当前的信息状态，系统下一步应该投入多少计算，并采取多大的动作？**

从这个角度看，Tokenizer 并不是一个孤立的 NLP 历史遗留组件。它是：

\[
\boxed{\text{AI Compute Granularity}}
\]

问题最底层的一种体现。

### 回到最开始的问题：大模型眼里的世界，到底由什么组成？

答案已经发生了很多次变化。

早期语言学系统认为：

\[
\text{World}
\rightarrow
\text{Words + Grammar}
\]

Subword 模型认为：

\[
\text{World}
\rightarrow
\text{Statistical Pieces}
\]

Byte 模型认为：

\[
\text{Text}
\rightarrow
\text{256 Symbols}
\]

多模态模型开始认为：

\[
\text{Text/Image/Audio/Video}
\rightarrow
\text{Tokens}
\]

而下一阶段的模型也许会认为：

\[
\boxed{
\text{世界不是由固定 Token 组成，而是由值得投入不同计算量的信息区域组成。}
}
\]

这可能才是 Tokenizer 历史真正指向的方向。

## 结语：Token 不是意义的原子，而是计算的原子

如果全文只保留一个结论，我希望是这一句：

\[
\boxed{
\text{Token 是模型世界里的「计算原子」，而不是现实世界天然存在的「意义原子」。}
}
\]

人类当然可以赋予 Token 语言学意义。我们看到 `ing` 会想到后缀；看到 `人工智能` 会想到概念；看到视觉 code 327 经常对应草地，会把它解释成「grass token」。

这些解释可能非常有价值。但我们始终需要区分：

\[
\text{Human Interpretation}
\]

\[
\text{Algorithmic Objective}
\]

\[
\text{Engineering Trade-off}
\]

以及：

\[
\text{Emergent Model Representation}
\]

它们不是同一件事。

从这个角度重新看整个历史，我们会发现：BPE 的伟大，并不在于它找到了真正的「词」。Byte-BPE 的意义，也不只是消灭 `<UNK>`。中文 Tokenizer 的问题，也不只是「中文比英文贵」。视觉 Tokenizer 的任务，更不是简单地给图片造一个词表。

这些技术共同研究的是一个非常基础的问题：

\[
\boxed{
\text{我们应该用怎样的离散尺度，让一个有限计算系统去建模一个复杂世界？}
}
\]

从最早的词典和语法规则，到 BPE、Byte、VQ，再到 BLT 和动态视觉 Token，七十年的技术变化其实都围绕这一问题展开。

最开始，人类规定：什么值得成为一个符号。后来，数据统计决定：什么值得合并成一个符号。再后来，神经网络学习：什么信息应该共享一个离散 code。而现在，我们开始让模型自己判断：**哪里值得一个 Token，哪里值得更多计算。**

这也许才是 Tokenizer 真正的终局问题：

\[
\boxed{
\text{Representation}
\rightarrow
\text{Compression}
\rightarrow
\text{Computation}
}
\]

而所谓 Token，只是我们在这条演化道路上，为「一个单位的计算」暂时取的名字。

## 延伸阅读

1. W. John Hutchins, *The Georgetown-IBM Experiment Demonstrated in January 1954*：理解早期词典 + 规则式机器翻译。
2. Rico Sennrich, Barry Haddow, Alexandra Birch, *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016：理解 BPE 为什么真正进入现代 NLP。
3. Taku Kudo, John Richardson, *SentencePiece*, EMNLP 2018：理解 raw text、语言无关 tokenizer 与 BPE/Unigram 的关系。
4. Vilém Zouhar et al., *Tokenization and the Noiseless Channel*, ACL 2023：理解为什么 tokenizer 不能只用压缩率评价。
5. Craig Schmidt et al., *Tokenization Is More Than Compression*, EMNLP 2024：非常直接地挑战「token 越少越好」。
6. David A. Haslett, *Tokenization Changes Meaning in Large Language Models: Evidence from Chinese*, Computational Linguistics 2025：本文最推荐精读的一篇，用中文证明 token boundary 能系统性影响模型表征。
7. Chenglei Si et al., *Sub-Character Tokenization for Chinese Pretrained Language Models*, TACL 2023：进一步理解中文字、字形、语音与 sub-character representation。
8. Linting Xue et al., *ByT5*, TACL 2022：理解为什么有人试图完全抛弃 subword vocabulary，直接回到 bytes。
9. Artidoro Pagnoni et al., *Byte Latent Transformer: Patches Scale Better Than Tokens*, ACL 2025：理解 Tokenizer 从固定表示走向动态计算分配。
10. Alexey Dosovitskiy et al., *An Image is Worth 16×16 Words*, ICLR 2021：理解视觉 patch token 的起点。
11. Aaron van den Oord et al., *Neural Discrete Representation Learning*, 2017：VQ-VAE，理解 visual/audio discrete token 的基础。
12. Lijun Yu et al., *Language Model Beats Diffusion: Tokenizer is Key to Visual Generation*, ICLR 2024：理解视觉 tokenizer 为什么可能决定 AR visual generation 的上限。
13. Xinlong Wang et al., *Emu3: Next-Token Prediction is All You Need*, 2024：理解 text/image/video 统一成离散 Token 后的 next-token paradigm。
14. Roman Bachmann et al., *FlexTok*, ICML 2025：理解视觉 token 从固定数量走向 variable-length、coarse-to-fine 表示。








































