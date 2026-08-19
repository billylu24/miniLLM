#include "internal/minillm_internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * 训练是“看答案并改参数”，推理是“没有答案，只反复预测下一个 token”。
 * 本文件完全不调用 backward，模型参数在生成期间保持不变。
 */

static unsigned char sample_logits(MiniLLM *model, const float *logits,
                                   float temperature, int top_k) {
    int id;
    int best_id = 0;
    float best_value = logits[0];
    float probabilities[MINILLM_VOCAB_SIZE];
    unsigned char selected[MINILLM_VOCAB_SIZE] = {0};
    float maximum = -FLT_MAX;
    double total = 0.0;

    /* temperature<=0 表示始终选择最高分，也叫 greedy decoding。 */
    for (id = 1; id < MINILLM_VOCAB_SIZE; ++id) {
        if (logits[id] > best_value) {
            best_value = logits[id];
            best_id = id;
        }
    }
    if (temperature <= 0.0f) return (unsigned char)best_id;

    /* top-k 只允许分数最高的 k 个 byte 参加抽样。V 很小，朴素选择足够清楚。 */
    if (top_k <= 0 || top_k > MINILLM_VOCAB_SIZE) {
        top_k = MINILLM_VOCAB_SIZE;
    }
    for (id = 0; id < top_k; ++id) {
        int candidate;
        int candidate_id = -1;
        float candidate_value = -FLT_MAX;
        for (candidate = 0; candidate < MINILLM_VOCAB_SIZE; ++candidate) {
            if (selected[candidate] == 0u &&
                logits[candidate] > candidate_value) {
                candidate_value = logits[candidate];
                candidate_id = candidate;
            }
        }
        selected[candidate_id] = 1u;
        if (candidate_value / temperature > maximum) {
            maximum = candidate_value / temperature;
        }
    }

    /* temperature 越小，logit 差距被放大，分布越确定；越大则越随机。 */
    for (id = 0; id < MINILLM_VOCAB_SIZE; ++id) {
        if (selected[id] != 0u) {
            probabilities[id] = expf(logits[id] / temperature - maximum);
            total += (double)probabilities[id];
        } else {
            probabilities[id] = 0.0f;
        }
    }
    {
        double needle = (double)minillm_random_uniform(model) * total;
        for (id = 0; id < MINILLM_VOCAB_SIZE; ++id) {
            needle -= (double)probabilities[id];
            if (needle <= 0.0 && selected[id] != 0u) return (unsigned char)id;
        }
    }
    return (unsigned char)best_id;
}

int minillm_generate(MiniLLM *model, const unsigned char *prompt,
                     size_t prompt_size, unsigned char *continuation,
                     size_t continuation_size, float temperature, int top_k) {
    unsigned char *history;
    size_t history_size;
    size_t generated;
    if (model == NULL || continuation == NULL ||
        (prompt_size > 0u && prompt == NULL)) return 0;

    history = minillm_calloc(prompt_size + continuation_size + 1u,
                             sizeof(unsigned char));
    if (history == NULL) return 0;
    if (prompt_size > 0u) {
        memcpy(history, prompt, prompt_size);
        history_size = prompt_size;
    } else {
        history[0] = (unsigned char)'\n';
        history_size = 1u;
    }

    for (generated = 0u; generated < continuation_size; ++generated) {
        const size_t context = (size_t)model->config.context_length;
        const size_t window_size = history_size < context ? history_size : context;
        const unsigned char *window = history + history_size - window_size;
        ForwardCache *cache = minillm_forward(model, window, NULL,
                                              (int)window_size);
        unsigned char next;
        if (cache == NULL) {
            free(history);
            return 0;
        }
        /* 只使用最后一个位置的 logits，因为它代表“接下来是什么”。 */
        next = sample_logits(model,
            cache->logits + (window_size - 1u) * MINILLM_VOCAB_SIZE,
            temperature, top_k);
        minillm_cache_destroy(model, cache);
        continuation[generated] = next;
        history[history_size++] = next;
    }
    free(history);
    return 1;
}
