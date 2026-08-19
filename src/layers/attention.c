#include "internal/minillm_internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

/*
 * 注意力解决的问题：当前位置应该从前文哪些位置取信息？
 *
 * Q（query）表示“我在找什么”，K（key）表示“我有什么标签”，二者点积越大，
 * 当前位置越关注那个历史位置。V（value）是真正被加权取回的信息。
 *
 * probability 的形状是 [H,T,T]：
 *   probability[head, 当前位置 i, 被查看位置 j]
 */

void minillm_attention_forward(const float *q, const float *k, const float *v,
                               float *probability, float *output,
                               int token_count, int d_model, int heads) {
    const int head_dim = d_model / heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    int i;

    for (i = 0; i < token_count; ++i) {
        int head;
        for (head = 0; head < heads; ++head) {
            int j;
            float max_score = -FLT_MAX;
            float denominator = 0.0f;
            const size_t row_base = ((size_t)head * (size_t)token_count +
                                     (size_t)i) * (size_t)token_count;

            /* 第一步：Q[i] 与每个过去的 K[j] 做点积。j<=i 就是因果 mask。 */
            for (j = 0; j <= i; ++j) {
                int c;
                float score = 0.0f;
                for (c = 0; c < head_dim; ++c) {
                    const int channel = head * head_dim + c;
                    score += q[i * d_model + channel] *
                             k[j * d_model + channel];
                }
                score *= scale;
                probability[row_base + (size_t)j] = score;
                if (score > max_score) max_score = score;
            }

            /* 第二步：softmax。先减最大值，避免 exp(score) 上溢。 */
            for (j = 0; j <= i; ++j) {
                const float e = expf(probability[row_base + (size_t)j] - max_score);
                probability[row_base + (size_t)j] = e;
                denominator += e;
            }
            for (j = 0; j <= i; ++j) {
                int c;
                const float p = probability[row_base + (size_t)j] / denominator;
                probability[row_base + (size_t)j] = p;
                /* 第三步：按概率加权 V[j]。 */
                for (c = 0; c < head_dim; ++c) {
                    const int channel = head * head_dim + c;
                    output[i * d_model + channel] +=
                        p * v[j * d_model + channel];
                }
            }
        }
    }
}

int minillm_attention_backward(const float *q, const float *k, const float *v,
                               const float *probability, const float *d_output,
                               float *d_q, float *d_k, float *d_v,
                               int token_count, int d_model, int heads) {
    const int head_dim = d_model / heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float *d_probability = minillm_calloc((size_t)token_count, sizeof(float));
    int i;
    if (d_probability == NULL) return 0;

    for (i = 0; i < token_count; ++i) {
        int head;
        for (head = 0; head < heads; ++head) {
            int j;
            float probability_dot = 0.0f;
            const size_t row_base = ((size_t)head * (size_t)token_count +
                                     (size_t)i) * (size_t)token_count;

            /* output=P@V，所以 dP=d_output@V，dV=P^T@d_output。 */
            for (j = 0; j <= i; ++j) {
                int c;
                float dp = 0.0f;
                const float p = probability[row_base + (size_t)j];
                for (c = 0; c < head_dim; ++c) {
                    const int channel = head * head_dim + c;
                    dp += d_output[i * d_model + channel] *
                          v[j * d_model + channel];
                    d_v[j * d_model + channel] +=
                        p * d_output[i * d_model + channel];
                }
                d_probability[j] = dp;
                probability_dot += p * dp;
            }

            /* softmax 导数：dscore[j]=P[j]*(dP[j]-sum(P*dP))。 */
            for (j = 0; j <= i; ++j) {
                int c;
                const float p = probability[row_base + (size_t)j];
                const float d_score = p * (d_probability[j] - probability_dot) * scale;
                /* score=Q·K/sqrt(Dh)，所以梯度分别乘另一个向量。 */
                for (c = 0; c < head_dim; ++c) {
                    const int channel = head * head_dim + c;
                    d_q[i * d_model + channel] +=
                        d_score * k[j * d_model + channel];
                    d_k[j * d_model + channel] +=
                        d_score * q[i * d_model + channel];
                }
            }
        }
    }
    free(d_probability);
    return 1;
}
