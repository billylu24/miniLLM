# miniLLM 分阶段学习路线

不要从头到尾一次读完所有代码。下面每一阶段只回答一个问题；前一阶段没有说清
之前，完全可以不看后一阶段。

## 第 0 阶段：先运行，不读实现

目标：知道程序的输入和输出。

```console
cmake --build cmake-build-debug
cmake-build-debug/minillm.exe train data/tiny.txt tiny.bin 100 0.003
cmake-build-debug/minillm.exe generate tiny.bin "the " 100 0.8 32
```

先记住一句话：训练反复执行“预测 → 计算错误 → 求梯度 → 改参数”；生成只执行
“预测 → 抽样 → 把结果接回输入”。

## 第 1 阶段：模型有哪些数字

阅读：

1. `include/minillm.h` 中的 `MiniLLMConfig`；
2. `src/internal/minillm_internal.h` 中的 `Tensor`、`Block`、`MiniLLM`；
3. `src/core/model.c`。

暂时不要看 forward。你只需回答：

- token embedding 为什么有 `256 × D` 个数？
- 一个 `D × D` 矩阵做什么？
- 为什么每个参数同时有 `value/grad/m/v`？

## 第 2 阶段：只学线性层

阅读 `src/layers/linear.c`。可以拿 `I=2, O=3` 在纸上画：

```text
x = [x0, x1]
W = [[w00,w01,w02],
     [w10,w11,w12]]
y = [x0*w00+x1*w10, x0*w01+x1*w11, x0*w02+x1*w12]
```

先看 forward，确认每个循环对应上式；第二遍再看 backward。

## 第 3 阶段：只学归一化

阅读 `src/layers/README.md` 的 RMSNorm 小节，再读 `src/layers/rmsnorm.c`。

建议给 `x=[3,4]` 手算 RMS。第一遍可以跳过 `minillm_rms_backward()` 的化简式，
只理解它最终需要产出 `dx` 和 `d_gain`。

## 第 4 阶段：只学一个注意力头

先假装 `H=1`，阅读 `src/layers/attention.c` 的 forward：

1. Q 与 K 点积得到相关性分数；
2. 因果 mask 排除未来位置；
3. softmax 把分数变成总和为 1 的权重；
4. 用权重加权 V。

等单头完全清楚后，再把 D 个通道分给 H 个头。第一次不要读 attention backward。

## 第 5 阶段：把层串成模型

先读 `docs/01-ARCHITECTURE.md`，再读 `src/model/forward.c`。此时各算子已经见过，
forward.c 只是在编排：

```text
embedding → [norm → attention → residual → norm → MLP → residual] × L
          → norm → logits → loss
```

在每行函数调用旁写下输入和输出形状，是理解模型最有效的方法。

## 第 6 阶段：反向传播

先把 forward 路线倒着写一遍，再读 `src/model/backward.c`。看到一个算子，就回到
对应 layers 文件看它的 backward。重点不是背导数，而是理解：

- 上游传来 `dy`；
- 当前算子算出参数梯度和 `dx`；
- `dx` 继续传给前一个算子；
- 残差加法会分流梯度，共用输入会汇合梯度。

## 第 7 阶段：参数为什么会学习

阅读 `src/training/optimizer.c`。先把 AdamW 临时理解成“带记忆、会自动调节每个
参数步长的梯度下降”。随后再逐项理解 m、v、bias correction 和 weight decay。

运行 `tests/test_minillm.c`，观察 loss 从约 `ln(256)=5.545` 降到接近 0。

## 第 8 阶段：生成与保存

最后阅读：

- `src/inference/generate.c`：模型怎样一个 byte 一个 byte 地写文本；
- `src/io/checkpoint.c`：训练状态怎样变成磁盘文件。

这两部分不引入新的 Transformer 数学，可以作为较轻松的收尾。

## 每阶段的学习规则

1. 先理解数组形状，再理解数组内容。
2. 先理解 forward，再理解 backward。
3. 先用一个 token、一个 head 思考，再推广到循环。
4. 遇到指针表达式，把它改写成二维下标。
5. 改代码后立即运行梯度检查和过拟合测试。
