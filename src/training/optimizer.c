#include "internal/minillm_internal.h"

#include <math.h>

/*
 * backward 只回答“参数应该往哪个方向变”；optimizer 决定“实际迈多大一步”。
 * AdamW 为每个参数保存梯度的一阶移动平均 m 和平方梯度移动平均 v。
 */

void minillm_apply_gradients(MiniLLM *model, MiniLLMOptimizer optimizer) {
    size_t tensor;
    double norm_squared = 0.0;
    float gradient_scale = 1.0f;
    float bias_correction1;
    float bias_correction2;
    if (model == NULL || optimizer.learning_rate <= 0.0f ||
        optimizer.beta1 < 0.0f || optimizer.beta1 >= 1.0f ||
        optimizer.beta2 < 0.0f || optimizer.beta2 >= 1.0f ||
        optimizer.epsilon <= 0.0f) {
        return;
    }

    /* 把所有参数梯度看成一个长向量，计算它的 L2 长度。 */
    for (tensor = 0u; tensor < model->tensor_count; ++tensor) {
        size_t i;
        const Tensor *parameter = model->tensors[tensor];
        for (i = 0u; i < parameter->count; ++i) {
            norm_squared += (double)parameter->grad[i] *
                            (double)parameter->grad[i];
        }
    }
    /* 超过阈值时等比例缩小，方向不变。这能降低偶发梯度爆炸的破坏。 */
    if (optimizer.grad_clip > 0.0f && norm_squared > 0.0) {
        const float norm = (float)sqrt(norm_squared);
        if (norm > optimizer.grad_clip) {
            gradient_scale = optimizer.grad_clip / norm;
        }
    }

    ++model->step;
    bias_correction1 = 1.0f - powf(optimizer.beta1, (float)model->step);
    bias_correction2 = 1.0f - powf(optimizer.beta2, (float)model->step);
    for (tensor = 0u; tensor < model->tensor_count; ++tensor) {
        size_t i;
        Tensor *parameter = model->tensors[tensor];
        for (i = 0u; i < parameter->count; ++i) {
            const float g = parameter->grad[i] * gradient_scale;
            float m_hat;
            float v_hat;
            parameter->m[i] = optimizer.beta1 * parameter->m[i] +
                              (1.0f - optimizer.beta1) * g;
            parameter->v[i] = optimizer.beta2 * parameter->v[i] +
                              (1.0f - optimizer.beta2) * g * g;
            m_hat = parameter->m[i] / bias_correction1;
            v_hat = parameter->v[i] / bias_correction2;

            /* Adam 自适应步长 + AdamW 解耦权重衰减。 */
            parameter->value[i] -= optimizer.learning_rate *
                (m_hat / (sqrtf(v_hat) + optimizer.epsilon) +
                 optimizer.weight_decay * parameter->value[i]);
        }
    }
}

float minillm_train_step(MiniLLM *model, const unsigned char *data,
                         size_t data_size, MiniLLMOptimizer optimizer) {
    const size_t context = model != NULL
        ? (size_t)model->config.context_length : 0u;
    size_t offset;
    float loss;
    if (model == NULL || data == NULL || data_size < context + 1u) return NAN;

    /* 输入 data[offset..offset+T-1]，答案向右错一位。 */
    offset = (size_t)(minillm_random_u64(model) %
                      (uint64_t)(data_size - context));
    loss = minillm_loss_and_backward(model, data + offset,
                                     data + offset + 1u, (int)context);
    if (isfinite(loss)) minillm_apply_gradients(model, optimizer);
    return loss;
}
