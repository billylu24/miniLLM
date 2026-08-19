#include "internal/minillm_internal.h"

/*
 * 线性层是 Transformer 中工作量最大的基础算子。
 *
 * 一行输入 x 有 I 个数，输出 y 有 O 个数，权重 W 的形状是 [I,O]：
 *
 *   y[o] = b[o] + x[0]W[0,o] + ... + x[I-1]W[I-1,o]
 *
 * rows 表示同时有多少行输入。训练时 rows 通常是 token 数 T。
 * 数组是行优先布局，所以 W[in,out] 的地址是 in*O+out。
 */

void minillm_linear_forward(const float *x, const Tensor *weight,
                            const Tensor *bias, float *y,
                            int rows, int input_dim, int output_dim) {
    int row;
    for (row = 0; row < rows; ++row) {
        int out;
        for (out = 0; out < output_dim; ++out) {
            int in;
            float sum = bias != NULL ? bias->value[out] : 0.0f;
            for (in = 0; in < input_dim; ++in) {
                sum += x[row * input_dim + in] *
                       weight->value[in * output_dim + out];
            }
            y[row * output_dim + out] = sum;
        }
    }
}

/*
 * 对 y=xW+b 使用链式法则：
 *
 *   dW[in,out] += x[in] * dy[out]
 *   db[out]    += dy[out]
 *   dx[in]     += W[in,out] * dy[out]
 *
 * 使用 += 而不是 =，因为共享参数和残差连接会让梯度从多条路径汇合。
 */
void minillm_linear_backward(const float *x, Tensor *weight, Tensor *bias,
                             const float *dy, float *dx,
                             int rows, int input_dim, int output_dim) {
    int row;
    for (row = 0; row < rows; ++row) {
        int out;
        for (out = 0; out < output_dim; ++out) {
            int in;
            const float gradient = dy[row * output_dim + out];
            if (bias != NULL) {
                bias->grad[out] += gradient;
            }
            for (in = 0; in < input_dim; ++in) {
                weight->grad[in * output_dim + out] +=
                    x[row * input_dim + in] * gradient;
                dx[row * input_dim + in] +=
                    weight->value[in * output_dim + out] * gradient;
            }
        }
    }
}
