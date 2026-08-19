#include "internal/minillm_internal.h"

#include <stdlib.h>
#include <string.h>

/*
 * 反向传播需要前向时的中间值，例如 ReLU 需要知道输入是否大于 0，注意力需要
 * softmax 概率。ForwardCache 就是这本“草稿簿”。训练结束后整本释放。
 *
 * 为了让第一版易读，这里给每种中间结果单独分配数组。高性能实现通常会规划
 * 一整块 activation memory 并复用空间，llm.c 就采用了那种方式。
 */

static int allocate_block_cache(BlockCache *cache, int t, int d, int h, int f) {
    const size_t td = (size_t)t * (size_t)d;
    const size_t tf = (size_t)t * (size_t)f;
    const size_t htt = (size_t)h * (size_t)t * (size_t)t;
    memset(cache, 0, sizeof(*cache));

#define ALLOCATE(field, count)                                             \
    do {                                                                   \
        cache->field = minillm_calloc((count), sizeof(float));             \
        if (cache->field == NULL) return 0;                                \
    } while (0)
    ALLOCATE(norm1, td);
    ALLOCATE(inv_rms1, (size_t)t);
    ALLOCATE(q, td);
    ALLOCATE(k, td);
    ALLOCATE(v, td);
    ALLOCATE(prob, htt);
    ALLOCATE(attention, td);
    ALLOCATE(projection, td);
    ALLOCATE(residual1, td);
    ALLOCATE(norm2, td);
    ALLOCATE(inv_rms2, (size_t)t);
    ALLOCATE(hidden_pre, tf);
    ALLOCATE(hidden, tf);
    ALLOCATE(mlp, td);
    ALLOCATE(output, td);
#undef ALLOCATE
    return 1;
}

static void free_block_cache(BlockCache *cache) {
    free(cache->norm1);
    free(cache->inv_rms1);
    free(cache->q);
    free(cache->k);
    free(cache->v);
    free(cache->prob);
    free(cache->attention);
    free(cache->projection);
    free(cache->residual1);
    free(cache->norm2);
    free(cache->inv_rms2);
    free(cache->hidden_pre);
    free(cache->hidden);
    free(cache->mlp);
    free(cache->output);
    memset(cache, 0, sizeof(*cache));
}

ForwardCache *minillm_cache_create(const MiniLLM *model, int token_count) {
    ForwardCache *cache = minillm_calloc(1u, sizeof(*cache));
    int layer;
    if (cache == NULL) {
        return NULL;
    }
    cache->token_count = token_count;
    cache->blocks = minillm_calloc((size_t)model->config.n_layers,
                                   sizeof(*cache->blocks));
    cache->embedding = minillm_calloc((size_t)token_count *
                                      (size_t)model->config.d_model, sizeof(float));
    cache->final_norm = minillm_calloc((size_t)token_count *
                                       (size_t)model->config.d_model, sizeof(float));
    cache->final_inv_rms = minillm_calloc((size_t)token_count, sizeof(float));
    cache->logits = minillm_calloc((size_t)token_count * MINILLM_VOCAB_SIZE,
                                   sizeof(float));
    if (cache->blocks == NULL || cache->embedding == NULL ||
        cache->final_norm == NULL || cache->final_inv_rms == NULL ||
        cache->logits == NULL) {
        minillm_cache_destroy(model, cache);
        return NULL;
    }
    for (layer = 0; layer < model->config.n_layers; ++layer) {
        if (!allocate_block_cache(&cache->blocks[layer], token_count,
                                  model->config.d_model, model->config.n_heads,
                                  model->config.hidden_dim)) {
            minillm_cache_destroy(model, cache);
            return NULL;
        }
    }
    return cache;
}

void minillm_cache_destroy(const MiniLLM *model, ForwardCache *cache) {
    int layer;
    if (cache == NULL) {
        return;
    }
    if (cache->blocks != NULL) {
        for (layer = 0; layer < model->config.n_layers; ++layer) {
            free_block_cache(&cache->blocks[layer]);
        }
    }
    free(cache->blocks);
    free(cache->embedding);
    free(cache->final_norm);
    free(cache->final_inv_rms);
    free(cache->logits);
    free(cache);
}
