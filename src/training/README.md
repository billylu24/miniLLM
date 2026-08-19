# training：从梯度到参数更新

`optimizer.c` 包含两个层次：

- `minillm_train_step()` 组织一次训练：随机取数据、forward、backward、update。
- `minillm_apply_gradients()` 实现全局梯度裁剪和 AdamW。

先把最简单梯度下降记成 `w = w - learning_rate * grad`。AdamW 只是为每个参数
额外记录近期梯度均值 m 和平方均值 v，让步长更稳定。
