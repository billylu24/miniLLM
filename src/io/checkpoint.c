#include "internal/minillm_internal.h"

#include <stdio.h>
#include <string.h>

/*
 * checkpoint 是训练过程的快照。只保存 value 不够：若要无缝继续 AdamW，还要
 * 保存 m、v、step 和随机数状态。grad 是一次性工作区，不需要保存。
 */

#define CHECKPOINT_MAGIC "MLLMV1\0"
#define CHECKPOINT_VERSION 1u

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t context_length;
    uint32_t d_model;
    uint32_t n_heads;
    uint32_t n_layers;
    uint32_t hidden_dim;
    uint64_t seed;
    uint64_t rng_state;
    uint64_t step;
    uint64_t parameter_count;
    uint64_t tensor_count;
} CheckpointHeader;

static int write_exact(FILE *file, const void *data, size_t size, size_t count) {
    return fwrite(data, size, count, file) == count;
}

static int read_exact(FILE *file, void *data, size_t size, size_t count) {
    return fread(data, size, count, file) == count;
}

int minillm_save(const MiniLLM *model, const char *path) {
    FILE *file;
    CheckpointHeader header;
    size_t tensor;
    int ok = 1;
    if (model == NULL || path == NULL) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;

    memset(&header, 0, sizeof(header));
    memcpy(header.magic, CHECKPOINT_MAGIC, 8u);
    header.version = CHECKPOINT_VERSION;
    header.vocab_size = (uint32_t)model->config.vocab_size;
    header.context_length = (uint32_t)model->config.context_length;
    header.d_model = (uint32_t)model->config.d_model;
    header.n_heads = (uint32_t)model->config.n_heads;
    header.n_layers = (uint32_t)model->config.n_layers;
    header.hidden_dim = (uint32_t)model->config.hidden_dim;
    header.seed = model->config.seed;
    header.rng_state = model->rng_state;
    header.step = model->step;
    header.parameter_count = (uint64_t)model->parameter_count;
    header.tensor_count = (uint64_t)model->tensor_count;

    ok = write_exact(file, &header, sizeof(header), 1u);
    for (tensor = 0u; ok && tensor < model->tensor_count; ++tensor) {
        const Tensor *parameter = model->tensors[tensor];
        const uint64_t count = (uint64_t)parameter->count;
        ok = write_exact(file, &count, sizeof(count), 1u) &&
             write_exact(file, parameter->value, sizeof(float), parameter->count) &&
             write_exact(file, parameter->m, sizeof(float), parameter->count) &&
             write_exact(file, parameter->v, sizeof(float), parameter->count);
    }
    if (fclose(file) != 0) ok = 0;
    return ok;
}

MiniLLM *minillm_load(const char *path) {
    FILE *file;
    CheckpointHeader header;
    MiniLLMConfig config;
    MiniLLM *model;
    size_t tensor;
    int trailing_byte;
    int stream_error;
    int close_result;
    if (path == NULL) return NULL;
    file = fopen(path, "rb");
    if (file == NULL) return NULL;
    if (!read_exact(file, &header, sizeof(header), 1u) ||
        memcmp(header.magic, CHECKPOINT_MAGIC, 8u) != 0 ||
        header.version != CHECKPOINT_VERSION) {
        fclose(file);
        return NULL;
    }

    config.vocab_size = (int)header.vocab_size;
    config.context_length = (int)header.context_length;
    config.d_model = (int)header.d_model;
    config.n_heads = (int)header.n_heads;
    config.n_layers = (int)header.n_layers;
    config.hidden_dim = (int)header.hidden_dim;
    config.seed = header.seed;
    model = minillm_create(config);
    if (model == NULL || header.parameter_count != (uint64_t)model->parameter_count ||
        header.tensor_count != (uint64_t)model->tensor_count) {
        minillm_destroy(model);
        fclose(file);
        return NULL;
    }
    model->rng_state = header.rng_state;
    model->step = header.step;

    for (tensor = 0u; tensor < model->tensor_count; ++tensor) {
        Tensor *parameter = model->tensors[tensor];
        uint64_t count = 0u;
        if (!read_exact(file, &count, sizeof(count), 1u) ||
            count != (uint64_t)parameter->count ||
            !read_exact(file, parameter->value, sizeof(float), parameter->count) ||
            !read_exact(file, parameter->m, sizeof(float), parameter->count) ||
            !read_exact(file, parameter->v, sizeof(float), parameter->count)) {
            minillm_destroy(model);
            fclose(file);
            return NULL;
        }
    }

    trailing_byte = fgetc(file);
    stream_error = ferror(file);
    close_result = fclose(file);
    if (trailing_byte != EOF || stream_error != 0 || close_result != 0) {
        minillm_destroy(model);
        return NULL;
    }
    return model;
}
