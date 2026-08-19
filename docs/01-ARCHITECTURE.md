# miniLLM 模型架构：从一句话到每个数组

## 1. 模型到底在学什么

语言模型只做一道分类题：给定前面的内容，预测下一个 token。

本项目把一个 byte 当一个 token，所以共有 256 个类别。对于文本 `hello`：

```text
输入:  h  e  l  l
答案:  e  l  l  o
```

一次输入可以同时产生四道训练题。模型在每个位置输出 256 个分数，正确 byte 的
分数应该逐渐变大。

## 2. 五个形状符号

| 符号 | 含义 | 默认值 |
|---|---|---:|
| V | 词表大小 | 256 |
| T | 本次输入长度 | 最多 32 |
| D | 一个 token 向量的宽度 | 32 |
| H | 注意力头数 | 4 |
| F | MLP 中间宽度 | 64 |

多头注意力中，每个头得到 `Dh=D/H=8` 个通道。

## 3. 为什么 token 要变成向量

byte id 只是一个整数。例如 `A=65` 并不意味着它在数学上比 `B=66` 小 1。
embedding 表可以理解为 256 行可学习词典：

```text
token_embedding[V,D]

第 65 行 → A 当前的 D 维表示
第 66 行 → B 当前的 D 维表示
```

同一个 byte 出现在不同位置时还需要位置信息，因此再取：

```text
x[token] = token_embedding[token_id] + position_embedding[token]
```

得到的 `x` 形状为 `[T,D]`。

## 4. 一个 Transformer block

```text
                 ┌──────── attention branch ────────┐
x [T,D] ──RMSNorm──Q,K,V──causal attention──Wo─────(+ )── r1 [T,D]
  │                                                  ▲
  └──────────────── residual shortcut ───────────────┘

                 ┌────────── MLP branch ────────────┐
r1 [T,D]──RMSNorm──Linear D→F──ReLU──Linear F→D────(+ )── output [T,D]
  │                                                  ▲
  └──────────────── residual shortcut ───────────────┘
```

“block”不是一个新数学操作，只是把几种基础操作按固定顺序组合起来。

## 5. 注意力的直觉

对每个 token 生成三种表示：

- Q：我现在想找什么信息？
- K：我这里有什么信息可供匹配？
- V：如果你关注我，真正拿走什么内容？

第 i 个位置与第 j 个位置的匹配分数：

```text
score[i,j] = dot(Q[i], K[j]) / sqrt(Dh)
```

生成时不能偷看未来，所以只允许 `j<=i`。以 T=4 为例：

```text
       被查看位置 j
       0  1  2  3
i=0    ✓  ×  ×  ×
i=1    ✓  ✓  ×  ×
i=2    ✓  ✓  ✓  ×
i=3    ✓  ✓  ✓  ✓
```

softmax 把每一行分数变成概率，然后：

```text
attention[i] = sum_(j<=i) probability[i,j] * V[j]
```

多个头只是把 D 个通道分组并行执行这套过程，使不同头可以学习不同关系。

## 6. MLP 在做什么

注意力负责 token 之间交换信息，MLP 则独立加工每个 token：

```text
hidden = ReLU(norm2 @ W1 + b1)   # [T,D] → [T,F]
mlp    = hidden @ W2 + b2        # [T,F] → [T,D]
```

先扩宽再压回 D 维，给每个位置更强的非线性变换能力。

## 7. 残差连接为什么重要

若一个子层暂时没有学到有用内容，`output=branch(x)` 会完全覆盖旧信息。残差写成：

```text
output = x + branch(x)
```

模型只需要学习“在原信息上增加什么”。反向传播时梯度也有一条直接通路，不必
完全穿过复杂分支。

## 8. 从最终向量到概率

经过 L 个 block 后仍是 `[T,D]`。输出线性层把每个 D 维向量变成 256 个 logits：

```text
logits = final_norm @ output_weight + output_bias   # [T,256]
```

softmax 概率为：

```text
p[id] = exp(logit[id]) / sum_j exp(logit[j])
```

正确答案概率越接近 1，`-log(p[target])` 越接近 0。

## 9. 完整形状流水线

| 步骤 | 输入 | 参数 | 输出 |
|---|---|---|---|
| token embedding | `[T]` ids | `[V,D]` | `[T,D]` |
| RMSNorm 1 | `[T,D]` | `[D]` | `[T,D]` |
| Q/K/V linear | `[T,D]` | 各 `[D,D]` | 各 `[T,D]` |
| attention scores | Q、K | 无 | `[H,T,T]` |
| weighted values | 概率、V | 无 | `[T,D]` |
| output projection | `[T,D]` | `[D,D]` | `[T,D]` |
| RMSNorm 2 | `[T,D]` | `[D]` | `[T,D]` |
| MLP first | `[T,D]` | `[D,F]`、`[F]` | `[T,F]` |
| MLP second | `[T,F]` | `[F,D]`、`[D]` | `[T,D]` |
| LM head | `[T,D]` | `[D,V]`、`[V]` | `[T,V]` |

## 10. backward 在概念上做什么

把模型想成复合函数：

```text
loss = A(B(C(parameters)))
```

链式法则说，一个参数对 loss 的影响等于沿路径上所有局部导数相乘。程序不创建
通用自动微分系统，而是给每个算子手写 backward：

```text
上游给当前算子 dy
当前算子计算自己的参数梯度
当前算子计算 dx，交给前一个算子
```

所以 `src/model/backward.c` 必须严格按照 forward 的相反顺序排列。

## 11. 与 GPT-2 的差异

本项目有意采用教学简化：byte tokenizer、RMSNorm、ReLU、MLP 宽度 2D、单样本
训练。GPT-2 使用 BPE、LayerNorm、GELU、4D MLP、batch，并共享 embedding 与
输出权重。本项目用于先学清计算图；精确 GPT-2 可继续学习 `karpathy/llm.c`。
