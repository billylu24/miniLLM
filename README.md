# miniLLM：只用 C11 从零训练一个小型 Transformer

这是一个面向学习的、完整可运行的 decoder-only Transformer。它不依赖
PyTorch、BLAS、tokenizer 库或任何第三方代码，只使用 C11 标准库和
`<math.h>`。项目包含训练、推理、手写反向传播、AdamW、checkpoint、CLI、
数值梯度检查和端到端测试。

这里的“mini”很重要：默认模型只有 34,400 个参数，目标是让一个人能够从头
读完，而不是追求现代大模型的速度和能力。现代 LLM 只是把相同的核心结构做得
更宽、更深，并配合高度优化的矩阵库、并行硬件和海量数据。

## 1. 最快上手

在 Linux/macOS 或已经把 CMake 放入 `PATH` 的 Windows 终端中：

```console
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

训练示例语料并保存模型：

```console
./build/minillm train data/tiny.txt tiny.bin 1000 0.003
```

Windows 下可执行文件通常是 `build\minillm.exe`：

```console
build\minillm.exe train data\tiny.txt tiny.bin 1000 0.003
build\minillm.exe generate tiny.bin "the " 300 0.8 32
```

继续训练与查看 checkpoint：

```console
build\minillm.exe resume tiny.bin data\tiny.txt 500 0.001
build\minillm.exe info tiny.bin
```

CLion 用户可以直接打开根目录的 `CMakeLists.txt`，选择 `minillm_cli` 或
`minillm_tests` target 运行。本项目要求严格 C11，CMake 已关闭 GNU 扩展。

## 2. 一次训练到底发生了什么

假设语料中有如下 byte：

```text
h e l l o
```

输入是 `h e l l`，监督答案是向右错一位的 `e l l o`。模型在每个位置预测
下一个 byte，并对所有位置的交叉熵取平均。

```text
byte id
   │
   ├── token embedding ──┐
   └── position embedding┴─> x
                              │
                  ┌───────────▼────────────┐
                  │ RMSNorm                │
                  │ causal multi-head      │
                  │ self-attention         │
                  │ + residual             │
                  │ RMSNorm → MLP → +resid │ × n_layers
                  └───────────┬────────────┘
                              │
                         final RMSNorm
                              │
                       linear → 256 logits
                              │
                     softmax + cross entropy
```

一次 `minillm_train_step()` 执行四件事：

1. 从语料随机截取 `context_length + 1` 个 byte。
2. `forward()` 计算 logits 和平均交叉熵。
3. `backward()` 按相反顺序计算每个参数的梯度。
4. `minillm_apply_gradients()` 用 AdamW 更新参数。

推理没有第 2 步的正确答案，也不执行反向传播。它拿最后一个位置的 logits，
经过 temperature 和 top-k 采样得到新 byte，把新 byte 接回上下文，然后重复。

## 3. 模块地图与源码阅读顺序

实现不再集中在一个千行文件中，而是按学习主题组织：

```text
src/
├── core/       参数、模型生命周期、随机数、前向缓存
├── layers/     线性层、RMSNorm、因果多头注意力
├── model/      把所有层串起来的 forward 与 backward
├── training/   梯度裁剪、AdamW、一次训练步骤
├── inference/  temperature、top-k、自回归生成
├── io/         checkpoint 保存与加载
└── internal/   各模块共享的数据结构与内部函数声明
```

先读 [`docs/00-LEARNING-PATH.md`](docs/00-LEARNING-PATH.md)，它把学习过程拆成
九个小阶段。第一次不需要理解反向传播。

| 顺序 | 文件或函数 | 重点 |
|---:|---|---|
| 1 | `include/minillm.h` | 从公开 API 建立整体认识 |
| 2 | `src/main.c` | 看训练、保存、生成如何串起来 |
| 3 | `src/core/model.c` | 参数有哪些、形状多大、如何初始化 |
| 4 | `src/layers/` | 一次只理解线性层、RMSNorm 或注意力 |
| 5 | `src/model/forward.c` | embedding、注意力、MLP、loss 如何串联 |
| 6 | `src/model/backward.c` | 沿计算图反向应用链式法则 |
| 7 | `src/training/optimizer.c` | 梯度裁剪与 AdamW |
| 8 | `src/inference/generate.c` | 滑动上下文和自回归采样 |
| 9 | `tests/test_minillm.c` | 如何证明梯度、训练和存盘是对的 |

建议边读边在纸上记录数组形状。最常用的符号是：

- `T`：当前 token 数量；
- `D`：`d_model`；
- `H`：注意力头数；
- `Dh = D/H`：每个头的宽度；
- `F`：MLP 隐藏宽度；
- `V = 256`：byte 词表大小。

先看图解版 [`docs/01-ARCHITECTURE.md`](docs/01-ARCHITECTURE.md)，再看公式版
[`docs/DERIVATION.md`](docs/DERIVATION.md)。每个 `src` 子目录也有一份 README，
只解释该模块，避免一次接触整个项目。

## 4. 为什么使用 byte tokenizer

词表恰好是 `[0, 255]`，文件中的每个 byte 直接就是 token id。优点是零依赖、
不会出现 unknown token，而且任何文件都能输入。缺点是 UTF-8 中文字符通常占
3 个 byte，序列会比 BPE tokenizer 更长。这个取舍非常适合教学：先理解模型，
以后再单独实现 BPE。

## 5. 默认模型

| 配置 | 默认值 |
|---|---:|
| vocabulary | 256 bytes |
| context length | 32 |
| model width | 32 |
| attention heads | 4 |
| Transformer blocks | 2 |
| MLP width | 64 |
| parameters | 34,400 |

所有矩阵都是行优先的一维 `float` 数组，矩阵乘法是普通的三重循环。这比优化库
慢很多，但你可以逐行跟踪每个乘加。默认模型适合短语料和概念验证，不会生成
接近商业大模型质量的文本。

## 6. 测试为什么可信

测试不只检查“程序没有崩溃”：

- **有限差分梯度检查**：用 loss 的定义近似某个参数的导数，与手写 backward
  比较；这能发现链式法则、缩放和下标错误。
- **微型过拟合**：模型必须把重复的 `abc` 语料 loss 从随机水平显著降下来。
- **checkpoint round-trip**：保存后加载的每个参数必须 bit-identical。
- **确定性生成**：保存随机数状态后，原模型与加载模型应生成相同 byte。
- **非法配置**：`d_model` 不能被头数整除时必须拒绝创建模型。

随机初始化时，256 类均匀预测的理论交叉熵约为 `ln(256) = 5.545`。如果初始
loss 离这个数很远，通常说明 softmax、交叉熵或初始化存在问题。

## 7. 可以亲手完成的扩展

建议一次只改一件事，并保持梯度检查和过拟合测试通过：

1. 把 ReLU 换成 GELU，并推导它的导数。
2. 实现学习率 warmup 与 cosine decay。
3. 让输出权重与 token embedding 权重共享。
4. 增加 validation split，分别报告训练和验证 loss。
5. 实现 batch size，而不是每步只训练一个序列。
6. 写一个最小 BPE tokenizer。
7. 保存为明确规定字节序的跨平台 checkpoint。
8. 最后再做 SIMD、线程或 BLAS 优化，并与这个朴素版本逐项对照。

## 8. 项目边界

“只用标准库”在这里表示源码只包含 ISO C 标准头文件。`<math.h>` 属于 C 标准
库；部分 Unix 工具链要求链接其独立的系统数学库 `libm`，CMake 会自动处理。
构建系统和编译器本身当然不是运行时依赖。

checkpoint 直接写入本机的整数和 `float` 表示，适合同一架构上的学习实验；它
不是面向生产环境的长期文件格式。训练也没有 GPU、batch、混合精度或并行化。
这些限制都是有意的：先得到一个透明、正确、可测试的基线。
