# model：把基础层串成 Transformer

- `forward.c` 按阅读顺序组织，回答“输入怎样变成 loss”。
- `backward.c` 严格反向排列，回答“loss 怎样把误差信号传给所有参数”。

第一次只读 forward。等能独立画出 block 数据流后，再读 backward。backward 中
看到线性层、RMSNorm 或注意力时，回到 `src/layers/` 查看对应局部导数。

残差连接是阅读重点：前向的加法会在反向时把梯度分成两份；多条路径回到同一
变量时，梯度必须累加。
