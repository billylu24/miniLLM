# miniLLM 公式与代码对照

本文只推导项目实际使用的公式。源码变量名与这里尽量保持一致。

## 1. 线性层

对一行输入 `x`：

```text
y[o] = sum_i x[i] W[i,o] + b[o]
```

收到上游梯度 `dy` 后：

```text
dW[i,o] += x[i] dy[o]
db[o]   += dy[o]
dx[i]   += W[i,o] dy[o]
```

代码位于 `src/layers/linear.c`。多个 token 共享同一组参数，
因此 `dW` 和 `db` 必须对所有行累加。

## 2. RMSNorm

对一个 D 维 token 向量：

```text
r = 1 / sqrt(mean(x²) + epsilon)
y_i = gain_i x_i r
```

令 `u_i = dy_i gain_i`，可以得到：

```text
dx_i = r u_i - x_i (r³/D) sum_j(u_j x_j)
d_gain_i += dy_i x_i r
```

RMSNorm 不减均值，比 LayerNorm 少一部分运算，也让反向公式更紧凑。

## 3. 多头因果注意力

先用三个线性变换得到 Q、K、V，然后把 D 个通道分成 H 组，每组宽度
`Dh = D/H`。对某个 head：

```text
score[i,j] = dot(Q[i], K[j]) / sqrt(Dh)
```

当 `j > i` 时，该位置属于未来，因果 mask 让它完全不参与 softmax：

```text
P[i,:] = softmax(score[i, 0..i])
A[i] = sum_(j<=i) P[i,j] V[j]
```

反向传播先从 `A = P @ V` 得到：

```text
dP[i,j] = dot(dA[i], V[j])
dV[j]  += P[i,j] dA[i]
```

softmax 一行的 Jacobian 可以化简为：

```text
dScore_j = P_j (dP_j - sum_k(P_k dP_k))
```

最后对点积求导，并把缩放 `1/sqrt(Dh)` 乘回来：

```text
dQ[i] += dScore[i,j] K[j] / sqrt(Dh)
dK[j] += dScore[i,j] Q[i] / sqrt(Dh)
```

代码集中在 `src/layers/attention.c`，可以不受完整模型流程干扰地单独阅读。

## 4. 残差连接

```text
y = x + branch(x)
```

加法对两个输入的局部导数都是 1，所以 `dy` 一份直接流向 `x`，另一份进入
branch。这也是源码中多次 `memcpy()` 梯度的原因。之后其他路径的梯度继续用
`+=` 累加。

## 5. ReLU 与 MLP

```text
pre = norm @ W1 + b1
hidden = max(0, pre)
mlp = hidden @ W2 + b2
```

除 `pre = 0` 这个不可导点外，ReLU 的导数为：

```text
dpre = dhidden,  pre > 0
dpre = 0,        pre <= 0
```

## 6. softmax 交叉熵

对正确类别 `target`：

```text
loss = -log( exp(logit[target]) / sum_j exp(logit[j]) )
```

实现先减去最大 logit，以避免 `exp()` 上溢。softmax 和交叉熵合在一起求导后：

```text
dlogit[j] = probability[j] - (j == target ? 1 : 0)
```

项目对 T 个位置取平均，因此还要除以 T。

## 7. AdamW

第 t 步中：

```text
m = beta1 m + (1-beta1) g
v = beta2 v + (1-beta2) g²
m_hat = m / (1-beta1^t)
v_hat = v / (1-beta2^t)
w = w - learning_rate * (m_hat/(sqrt(v_hat)+epsilon) + weight_decay*w)
```

更新前先计算所有参数组成的全局梯度 L2 范数。超过 `grad_clip` 时，所有梯度
乘同一个比例；这样不会改变梯度方向，只缩短它的长度。

## 8. 有限差分为什么能检查反向传播

解析梯度来自上面的链式法则。数值梯度完全不使用这些公式：

```text
df/dw ≈ (f(w+epsilon) - f(w-epsilon)) / (2 epsilon)
```

两种独立方法接近时，反向传播大概率正确。它很慢且存在浮点误差，所以只用于
测试少量参数，不能替代正式训练中的 backward。
