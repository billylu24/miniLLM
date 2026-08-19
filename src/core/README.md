# core：模型的地基

推荐顺序：`memory.c → random.c → model.c → cache.c`。

- `memory.c`：解释 Tensor 在 C 中只是四个等长 float 数组。
- `random.c`：初始化和采样怎样做到可复现。
- `model.c`：列出所有可训练参数，负责创建和销毁模型。
- `cache.c`：保存 forward 中间结果，供 backward 使用。

读完后应能回答：参数与 activation 有什么区别？为什么 parameter 训练后长期
保留，而 cache 每次 forward 都重新创建？
