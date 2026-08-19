# layers：一次只学一个数学算子

这里没有完整 Transformer 流程，适合逐文件学习。

1. `linear.c`：矩阵乘法和它的梯度，是最重要的基础。
2. `rmsnorm.c`：控制每个 token 向量的数值尺度。
3. `attention.c`：唯一让不同 token 互相交换信息的层。

建议每次把 `rows=1`、维度设为 2 或 3，在纸上展开循环。详细整体形状见
`docs/01-ARCHITECTURE.md`，完整推导见 `docs/DERIVATION.md`。
