#include "internal/minillm_internal.h"

#include <float.h>
#include <math.h>

/*
 * 这个文件只讲“数据怎样从输入流到 loss”，不包含任何梯度公式。
 *
 * 一层 block 的路线：
 *   x ──RMSNorm──QKV──Attention──Wo──┐
 *   └────────────────────────────────+──residual1
 *                                      │
 *              RMSNorm──Linear──ReLU──Linear──┐
 *   residual1 ────────────────────────────────+──output
 *
 * 每层输出会成为下一层输入。最后再 RMSNorm，并投影到 256 个 byte logits。
 */

ForwardCache *minillm_forward(MiniLLM *model, const unsigned char *tokens,
                              const unsigned char *targets, int token_count) {
    const int d = model->config.d_model;
    const int hidden_dim = model->config.hidden_dim;
    ForwardCache *cache;
    const float *layer_input;
    int token;
    int layer;

    if (tokens == NULL || token_count <= 0 ||
        token_count > model->config.context_length) {
        return NULL;
    }
    cache = minillm_cache_create(model, token_count);
    if (cache == NULL) return NULL;

    /*
     * byte tokenizer 没有词表查找过程：unsigned char 的 0..255 就是 token id。
     * token embedding 表示“这是什么”，position embedding 表示“它在哪里”。
     */
    for (token = 0; token < token_count; ++token) {
        int channel;
        const int id = (int)tokens[token];
        for (channel = 0; channel < d; ++channel) {
            cache->embedding[token * d + channel] =
                model->token_embedding.value[id * d + channel] +
                model->position_embedding.value[token * d + channel];
        }
    }

    layer_input = cache->embedding;
    for (layer = 0; layer < model->config.n_layers; ++layer) {
        Block *block = &model->blocks[layer];
        BlockCache *bc = &cache->blocks[layer];
        int index;

        /* 注意力子层：先归一化，再生成 Q/K/V。 */
        minillm_rms_forward(layer_input, block->rms1_gain.value,
                            bc->norm1, bc->inv_rms1, token_count, d);
        minillm_linear_forward(bc->norm1, &block->wq, NULL,
                               bc->q, token_count, d, d);
        minillm_linear_forward(bc->norm1, &block->wk, NULL,
                               bc->k, token_count, d, d);
        minillm_linear_forward(bc->norm1, &block->wv, NULL,
                               bc->v, token_count, d, d);
        minillm_attention_forward(bc->q, bc->k, bc->v, bc->prob,
                                  bc->attention, token_count, d,
                                  model->config.n_heads);
        minillm_linear_forward(bc->attention, &block->wo, NULL,
                               bc->projection, token_count, d, d);

        /* 第一条残差：保留原输入，同时加上注意力取回的信息。 */
        for (index = 0; index < token_count * d; ++index) {
            bc->residual1[index] = layer_input[index] + bc->projection[index];
        }

        /* MLP 子层：每个 token 独立经历 D→F→D，不在 token 之间交换信息。 */
        minillm_rms_forward(bc->residual1, block->rms2_gain.value,
                            bc->norm2, bc->inv_rms2, token_count, d);
        minillm_linear_forward(bc->norm2, &block->w1, &block->b1,
                               bc->hidden_pre, token_count, d, hidden_dim);
        for (index = 0; index < token_count * hidden_dim; ++index) {
            bc->hidden[index] = bc->hidden_pre[index] > 0.0f
                                    ? bc->hidden_pre[index] : 0.0f;
        }
        minillm_linear_forward(bc->hidden, &block->w2, &block->b2,
                               bc->mlp, token_count, hidden_dim, d);
        for (index = 0; index < token_count * d; ++index) {
            bc->output[index] = bc->residual1[index] + bc->mlp[index];
        }
        layer_input = bc->output;
    }

    /* 每个 token 的 D 维表示投影为 V=256 个分数。分数尚不是概率，所以叫 logits。 */
    minillm_rms_forward(layer_input, model->final_rms_gain.value,
                        cache->final_norm, cache->final_inv_rms, token_count, d);
    minillm_linear_forward(cache->final_norm, &model->output_weight,
                           &model->output_bias, cache->logits,
                           token_count, d, MINILLM_VOCAB_SIZE);

    if (targets != NULL) {
        double loss_sum = 0.0;
        for (token = 0; token < token_count; ++token) {
            int id;
            float max_logit = -FLT_MAX;
            double sum_exp = 0.0;
            for (id = 0; id < MINILLM_VOCAB_SIZE; ++id) {
                const float value = cache->logits[token * MINILLM_VOCAB_SIZE + id];
                if (value > max_logit) max_logit = value;
            }
            for (id = 0; id < MINILLM_VOCAB_SIZE; ++id) {
                sum_exp += exp((double)cache->logits[
                                   token * MINILLM_VOCAB_SIZE + id] -
                               (double)max_logit);
            }
            /* -log softmax(target)，写成 log-sum-exp 形式以保证数值稳定。 */
            loss_sum += log(sum_exp) + (double)max_logit -
                (double)cache->logits[token * MINILLM_VOCAB_SIZE +
                                      (int)targets[token]];
        }
        cache->loss = (float)(loss_sum / (double)token_count);
    }
    return cache;
}

float minillm_loss(MiniLLM *model, const unsigned char *tokens,
                   const unsigned char *targets, int token_count) {
    ForwardCache *cache;
    float loss;
    if (model == NULL || targets == NULL) return NAN;
    cache = minillm_forward(model, tokens, targets, token_count);
    if (cache == NULL) return NAN;
    loss = cache->loss;
    minillm_cache_destroy(model, cache);
    return loss;
}
