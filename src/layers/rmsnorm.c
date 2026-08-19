#include "internal/minillm_internal.h"

#include <math.h>

/*
 * 神经网络层层叠加后，向量数值可能忽大忽小。RMSNorm 把每个 token 向量的
 * 均方根缩放到约 1，再乘一个可学习 gain。
 *
 * 对一行 D 维向量 x：
 *   r = 1 / sqrt((x0² + ... + x(D-1)²)/D + epsilon)
 *   y[i] = x[i] * r * gain[i]
 */

void minillm_rms_forward(const float *x, const float *gain, float *y,
                         float *inverse_rms, int rows, int columns) {
    int row;
    for (row = 0; row < rows; ++row) {
        int col;
        float square_sum = 0.0f;
        for (col = 0; col < columns; ++col) {
            const float value = x[row * columns + col];
            square_sum += value * value;
        }
        inverse_rms[row] = 1.0f /
            sqrtf(square_sum / (float)columns + MINILLM_RMS_EPSILON);
        for (col = 0; col < columns; ++col) {
            y[row * columns + col] = x[row * columns + col] *
                                      inverse_rms[row] * gain[col];
        }
    }
}

/*
 * 令 r=inverse_rms，u=dy*gain，可化简出：
 *   dx[i] = r*u[i] - x[i]*(r³/D)*sum_j(u[j]*x[j])
 *   d_gain[i] += dy[i]*x[i]*r
 *
 * 不必第一次阅读就独立推完；先对照 docs/03-rmsnorm.md 的计算图逐项理解。
 */
void minillm_rms_backward(const float *x, Tensor *gain,
                          const float *inverse_rms, const float *dy,
                          float *dx, int rows, int columns) {
    int row;
    for (row = 0; row < rows; ++row) {
        int col;
        float dot = 0.0f;
        const float r = inverse_rms[row];
        for (col = 0; col < columns; ++col) {
            const int index = row * columns + col;
            dot += dy[index] * gain->value[col] * x[index];
            gain->grad[col] += dy[index] * x[index] * r;
        }
        for (col = 0; col < columns; ++col) {
            const int index = row * columns + col;
            const float u = dy[index] * gain->value[col];
            dx[index] += r * u - x[index] * r * r * r * dot / (float)columns;
        }
    }
}
