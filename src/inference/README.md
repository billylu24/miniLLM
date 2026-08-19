# inference：一个 token 一个 token 地生成

`generate.c` 不修改模型参数。每轮只做：

1. 取最近 context_length 个 byte；
2. forward；
3. 读取最后位置的 256 个 logits；
4. temperature 缩放并保留 top-k；
5. 按概率抽出新 byte，接回历史；
6. 重复。

temperature 为 0 时总选最高分；温度升高会增加随机性。
