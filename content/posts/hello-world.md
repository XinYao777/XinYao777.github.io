---
title: "第一篇：搭建这个博客"
date: 2026-08-20T10:00:00+08:00
draft: false
tags: ["Hugo", "博客", "工具"]
categories: ["杂谈"]
math: true
summary: "用 Hugo + PaperMod 搭建个人技术博客，并通过 GitHub Actions 自动部署。"
---

## 为什么写博客

把学到的东西写下来，是最好的巩固方式。这里会持续记录 AI、机器学习和工程实践的笔记。

## 代码高亮示例

```python
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))
```

## 数学公式示例

行内公式：交叉熵损失 $\mathcal{L} = -\sum_i y_i \log \hat{y}_i$。

块级公式（Softmax）：

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

## 引用与列表

> 好的笔记不是抄写，而是重新组织。

- 支持标签、归档、全文搜索
- 支持暗黑模式
- `git push` 后自动部署上线
