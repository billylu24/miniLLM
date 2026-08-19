#include "internal/minillm_internal.h"

#include <math.h>
#include <stdlib.h>

/*
 * 本文件回答三个入门问题：
 *   1. 模型到底有哪些可学习参数？
 *   2. 每个参数数组有多大？
 *   3. 创建和销毁模型时，内存如何成对管理？
 *
 * 它不做任何神经网络计算。建议在读 forward.c 前先读完本文件。
 */

static int register_tensor(MiniLLM *model, Tensor *tensor, size_t count) {
    if (!minillm_tensor_allocate(tensor, count)) {
        minillm_tensor_free(tensor);
        return 0;
    }
    if (model->tensor_count >= model->tensor_capacity) {
        minillm_tensor_free(tensor);
        return 0;
    }
    model->tensors[model->tensor_count++] = tensor;
    model->parameter_count += count;
    return 1;
}

static void initialize_normal(MiniLLM *model, Tensor *tensor, float scale) {
    size_t i;
    for (i = 0u; i < tensor->count; ++i) {
        tensor->value[i] = minillm_random_normal(model) * scale;
    }
}

static void initialize_ones(Tensor *tensor) {
    size_t i;
    for (i = 0u; i < tensor->count; ++i) {
        tensor->value[i] = 1.0f;
    }
}

MiniLLMConfig minillm_default_config(void) {
    MiniLLMConfig config;
    config.vocab_size = MINILLM_VOCAB_SIZE;
    config.context_length = 32;
    config.d_model = 32;
    config.n_heads = 4;
    config.n_layers = 2;
    config.hidden_dim = 64;
    config.seed = UINT64_C(42);
    return config;
}

MiniLLMOptimizer minillm_default_optimizer(void) {
    MiniLLMOptimizer optimizer;
    optimizer.learning_rate = 3.0e-3f;
    optimizer.beta1 = 0.9f;
    optimizer.beta2 = 0.99f;
    optimizer.epsilon = 1.0e-8f;
    optimizer.weight_decay = 1.0e-4f;
    optimizer.grad_clip = 1.0f;
    return optimizer;
}

static int valid_config(MiniLLMConfig c) {
    /* 每个 head 必须得到整数个通道，所以 D 必须能被 H 整除。 */
    return c.vocab_size == MINILLM_VOCAB_SIZE && c.context_length > 0 &&
           c.d_model > 0 && c.n_heads > 0 && c.d_model % c.n_heads == 0 &&
           c.n_layers > 0 && c.hidden_dim > 0;
}

MiniLLM *minillm_create(MiniLLMConfig config) {
    MiniLLM *model;
    int layer;
    const size_t d = (size_t)config.d_model;
    const size_t f = (size_t)config.hidden_dim;

    if (!valid_config(config)) {
        return NULL;
    }
    model = minillm_calloc(1u, sizeof(*model));
    if (model == NULL) {
        return NULL;
    }
    model->config = config;
    model->rng_state = config.seed != 0u ? config.seed : UINT64_C(1);

    /* 全局参数 5 组；每个 block 10 组。这里只数“数组”，不是 float 数量。 */
    model->tensor_capacity = 5u + (size_t)config.n_layers * 10u;
    model->tensors = minillm_calloc(model->tensor_capacity,
                                    sizeof(*model->tensors));
    model->blocks = minillm_calloc((size_t)config.n_layers,
                                   sizeof(*model->blocks));
    if (model->tensors == NULL || model->blocks == NULL) {
        minillm_destroy(model);
        return NULL;
    }

    /* 每个 byte id 对应一个 D 维向量；每个位置也对应一个 D 维向量。 */
    if (!register_tensor(model, &model->token_embedding,
                         (size_t)MINILLM_VOCAB_SIZE * d) ||
        !register_tensor(model, &model->position_embedding,
                         (size_t)config.context_length * d)) {
        minillm_destroy(model);
        return NULL;
    }

    for (layer = 0; layer < config.n_layers; ++layer) {
        Block *b = &model->blocks[layer];
        if (!register_tensor(model, &b->rms1_gain, d) ||
            !register_tensor(model, &b->wq, d * d) ||
            !register_tensor(model, &b->wk, d * d) ||
            !register_tensor(model, &b->wv, d * d) ||
            !register_tensor(model, &b->wo, d * d) ||
            !register_tensor(model, &b->rms2_gain, d) ||
            !register_tensor(model, &b->w1, d * f) ||
            !register_tensor(model, &b->b1, f) ||
            !register_tensor(model, &b->w2, f * d) ||
            !register_tensor(model, &b->b2, d)) {
            minillm_destroy(model);
            return NULL;
        }
    }

    if (!register_tensor(model, &model->final_rms_gain, d) ||
        !register_tensor(model, &model->output_weight,
                         d * (size_t)MINILLM_VOCAB_SIZE) ||
        !register_tensor(model, &model->output_bias, MINILLM_VOCAB_SIZE)) {
        minillm_destroy(model);
        return NULL;
    }

    /* 权重从小正态分布开始；RMSNorm 的 gain 从 1 开始；bias 由 calloc 保持 0。 */
    initialize_normal(model, &model->token_embedding, 0.02f);
    initialize_normal(model, &model->position_embedding, 0.02f);
    for (layer = 0; layer < config.n_layers; ++layer) {
        Block *b = &model->blocks[layer];
        initialize_ones(&b->rms1_gain);
        initialize_normal(model, &b->wq, 0.02f);
        initialize_normal(model, &b->wk, 0.02f);
        initialize_normal(model, &b->wv, 0.02f);
        initialize_normal(model, &b->wo, 0.02f);
        initialize_ones(&b->rms2_gain);
        initialize_normal(model, &b->w1, 0.02f);
        initialize_normal(model, &b->w2, 0.02f);
    }
    initialize_ones(&model->final_rms_gain);
    initialize_normal(model, &model->output_weight, 0.02f);
    return model;
}

void minillm_destroy(MiniLLM *model) {
    size_t i;
    if (model == NULL) {
        return;
    }
    for (i = 0u; i < model->tensor_count; ++i) {
        minillm_tensor_free(model->tensors[i]);
    }
    free(model->blocks);
    free(model->tensors);
    free(model);
}

MiniLLMConfig minillm_config(const MiniLLM *model) {
    MiniLLMConfig empty = {0, 0, 0, 0, 0, 0, 0u};
    return model != NULL ? model->config : empty;
}

size_t minillm_parameter_count(const MiniLLM *model) {
    return model != NULL ? model->parameter_count : 0u;
}

static int locate_parameter(const MiniLLM *model, size_t index,
                            Tensor **tensor_out, size_t *local_index) {
    size_t tensor;
    if (model == NULL || index >= model->parameter_count) {
        return 0;
    }
    for (tensor = 0u; tensor < model->tensor_count; ++tensor) {
        if (index < model->tensors[tensor]->count) {
            *tensor_out = model->tensors[tensor];
            *local_index = index;
            return 1;
        }
        index -= model->tensors[tensor]->count;
    }
    return 0;
}

float minillm_parameter_value(const MiniLLM *model, size_t index) {
    Tensor *tensor = NULL;
    size_t local = 0u;
    return locate_parameter(model, index, &tensor, &local)
               ? tensor->value[local] : NAN;
}

float minillm_parameter_gradient(const MiniLLM *model, size_t index) {
    Tensor *tensor = NULL;
    size_t local = 0u;
    return locate_parameter(model, index, &tensor, &local)
               ? tensor->grad[local] : NAN;
}

int minillm_set_parameter_value(MiniLLM *model, size_t index, float value) {
    Tensor *tensor = NULL;
    size_t local = 0u;
    if (!locate_parameter(model, index, &tensor, &local)) {
        return 0;
    }
    tensor->value[local] = value;
    return 1;
}

uint64_t minillm_training_step(const MiniLLM *model) {
    return model != NULL ? model->step : 0u;
}
