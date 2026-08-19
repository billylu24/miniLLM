#include "internal/minillm_internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * backward 的唯一原则：把 forward 的操作倒着走，并在每一步应用链式法则。
 *
 * forward:  embedding → block 0 → block 1 → final norm → logits → loss
 * backward: loss → logits → final norm → block 1 → block 0 → embedding
 *
 * d_x 表示“最终 loss 对 x 的偏导数”。它不是新的模型数据，而是误差信号。
 */

static void zero_parameter_gradients(MiniLLM *model) {
    size_t tensor;
    for (tensor = 0u; tensor < model->tensor_count; ++tensor) {
        memset(model->tensors[tensor]->grad, 0,
               model->tensors[tensor]->count * sizeof(float));
    }
}

int minillm_backward(MiniLLM *model, const unsigned char *tokens,
                     const unsigned char *targets, ForwardCache *cache) {
    const int t = cache->token_count;
    const int d = model->config.d_model;
    const int hidden_dim = model->config.hidden_dim;
    const size_t td = (size_t)t * (size_t)d;
    float *d_final_norm = minillm_calloc(td, sizeof(float));
    float *d_layer_output = minillm_calloc(td, sizeof(float));
    int token;
    int layer;

    if (targets == NULL || d_final_norm == NULL || d_layer_output == NULL) {
        free(d_final_norm);
        free(d_layer_output);
        return 0;
    }
    zero_parameter_gradients(model);

    /*
     * 第 1 站：loss → logits → final_norm。
     * softmax 与交叉熵合并后，导数格外简单：
     *   d_logit[id] = (probability[id] - one_hot(target)[id]) / T
     */
    for (token = 0; token < t; ++token) {
        int id;
        float max_logit = -FLT_MAX;
        double denominator = 0.0;
        for (id = 0; id < MINILLM_VOCAB_SIZE; ++id) {
            const float value = cache->logits[token * MINILLM_VOCAB_SIZE + id];
            if (value > max_logit) max_logit = value;
        }
        for (id = 0; id < MINILLM_VOCAB_SIZE; ++id) {
            denominator += exp((double)cache->logits[
                                   token * MINILLM_VOCAB_SIZE + id] -
                               (double)max_logit);
        }
        for (id = 0; id < MINILLM_VOCAB_SIZE; ++id) {
            int channel;
            float gradient = (float)(exp((double)cache->logits[
                                             token * MINILLM_VOCAB_SIZE + id] -
                                         (double)max_logit) / denominator);
            if (id == (int)targets[token]) gradient -= 1.0f;
            gradient /= (float)t;
            model->output_bias.grad[id] += gradient;
            for (channel = 0; channel < d; ++channel) {
                model->output_weight.grad[
                    channel * MINILLM_VOCAB_SIZE + id] +=
                    cache->final_norm[token * d + channel] * gradient;
                d_final_norm[token * d + channel] +=
                    model->output_weight.value[
                        channel * MINILLM_VOCAB_SIZE + id] * gradient;
            }
        }
    }

    {
        const float *final_input =
            cache->blocks[model->config.n_layers - 1].output;
        minillm_rms_backward(final_input, &model->final_rms_gain,
                             cache->final_inv_rms, d_final_norm,
                             d_layer_output, t, d);
    }
    free(d_final_norm);

    /* 第 2 站：从最后一个 Transformer block 向第一个 block 逆序走。 */
    for (layer = model->config.n_layers - 1; layer >= 0; --layer) {
        Block *block = &model->blocks[layer];
        BlockCache *bc = &cache->blocks[layer];
        const float *layer_input = layer == 0 ? cache->embedding
                                              : cache->blocks[layer - 1].output;
        float *d_residual1 = minillm_calloc(td, sizeof(float));
        float *d_norm2 = minillm_calloc(td, sizeof(float));
        float *d_hidden = minillm_calloc((size_t)t * (size_t)hidden_dim,
                                         sizeof(float));
        float *d_hidden_pre = minillm_calloc((size_t)t * (size_t)hidden_dim,
                                             sizeof(float));
        float *d_attention = minillm_calloc(td, sizeof(float));
        float *d_q = minillm_calloc(td, sizeof(float));
        float *d_k = minillm_calloc(td, sizeof(float));
        float *d_v = minillm_calloc(td, sizeof(float));
        float *d_norm1 = minillm_calloc(td, sizeof(float));
        float *d_layer_input = minillm_calloc(td, sizeof(float));
        size_t index;
        int attention_ok;

        if (d_residual1 == NULL || d_norm2 == NULL || d_hidden == NULL ||
            d_hidden_pre == NULL || d_attention == NULL || d_q == NULL ||
            d_k == NULL || d_v == NULL || d_norm1 == NULL ||
            d_layer_input == NULL) {
            free(d_residual1); free(d_norm2); free(d_hidden); free(d_hidden_pre);
            free(d_attention); free(d_q); free(d_k); free(d_v); free(d_norm1);
            free(d_layer_input); free(d_layer_output);
            return 0;
        }

        /*
         * output = residual1 + mlp。
         * 加法的局部导数是 1，所以 d_output 原样复制到两条支路。
         */
        memcpy(d_residual1, d_layer_output, td * sizeof(float));
        minillm_linear_backward(bc->hidden, &block->w2, &block->b2,
                                d_layer_output, d_hidden,
                                t, hidden_dim, d);

        /* ReLU：正数区域斜率 1，负数区域斜率 0。 */
        for (index = 0u; index < (size_t)t * (size_t)hidden_dim; ++index) {
            d_hidden_pre[index] = bc->hidden_pre[index] > 0.0f
                                      ? d_hidden[index] : 0.0f;
        }
        minillm_linear_backward(bc->norm2, &block->w1, &block->b1,
                                d_hidden_pre, d_norm2,
                                t, d, hidden_dim);
        minillm_rms_backward(bc->residual1, &block->rms2_gain,
                             bc->inv_rms2, d_norm2, d_residual1, t, d);

        /* residual1 = layer_input + projection：梯度再次分流。 */
        memcpy(d_layer_input, d_residual1, td * sizeof(float));
        minillm_linear_backward(bc->attention, &block->wo, NULL,
                                d_residual1, d_attention, t, d, d);

        attention_ok = minillm_attention_backward(
            bc->q, bc->k, bc->v, bc->prob, d_attention,
            d_q, d_k, d_v, t, d, model->config.n_heads);
        if (!attention_ok) {
            free(d_residual1); free(d_norm2); free(d_hidden); free(d_hidden_pre);
            free(d_attention); free(d_q); free(d_k); free(d_v); free(d_norm1);
            free(d_layer_input); free(d_layer_output);
            return 0;
        }

        /* Q、K、V 都来自同一个 norm1，所以三份输入梯度累加到 d_norm1。 */
        minillm_linear_backward(bc->norm1, &block->wq, NULL,
                                d_q, d_norm1, t, d, d);
        minillm_linear_backward(bc->norm1, &block->wk, NULL,
                                d_k, d_norm1, t, d, d);
        minillm_linear_backward(bc->norm1, &block->wv, NULL,
                                d_v, d_norm1, t, d, d);
        minillm_rms_backward(layer_input, &block->rms1_gain,
                             bc->inv_rms1, d_norm1, d_layer_input, t, d);

        free(d_layer_output);
        d_layer_output = d_layer_input; /* 交给前一层继续回传。 */
        free(d_residual1);
        free(d_norm2);
        free(d_hidden);
        free(d_hidden_pre);
        free(d_attention);
        free(d_q);
        free(d_k);
        free(d_v);
        free(d_norm1);
    }

    /* 第 3 站：embedding = token_embedding + position_embedding。 */
    for (token = 0; token < t; ++token) {
        int channel;
        const int id = (int)tokens[token];
        for (channel = 0; channel < d; ++channel) {
            const float gradient = d_layer_output[token * d + channel];
            model->token_embedding.grad[id * d + channel] += gradient;
            model->position_embedding.grad[token * d + channel] += gradient;
        }
    }
    free(d_layer_output);
    return 1;
}

float minillm_loss_and_backward(MiniLLM *model, const unsigned char *tokens,
                                const unsigned char *targets, int token_count) {
    ForwardCache *cache;
    float loss;
    if (model == NULL || targets == NULL) return NAN;
    cache = minillm_forward(model, tokens, targets, token_count);
    if (cache == NULL) return NAN;
    loss = cache->loss;
    if (!minillm_backward(model, tokens, targets, cache)) loss = NAN;
    minillm_cache_destroy(model, cache);
    return loss;
}
