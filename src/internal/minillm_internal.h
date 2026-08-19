#ifndef MINILLM_INTERNAL_H
#define MINILLM_INTERNAL_H

/*
 * 这是各实现模块共享的“内部地图”。普通使用者只需要 include/minillm.h；
 * 学习实现时，再来这里查看模型有哪些参数、前向传播缓存了哪些中间结果。
 *
 * 形状记号：
 *   T = token 数量，D = 模型宽度，H = 注意力头数，F = MLP 宽度，V = 256。
 */

#include "minillm.h"

#include <stddef.h>
#include <stdint.h>

#define MINILLM_VOCAB_SIZE 256
#define MINILLM_RMS_EPSILON 1.0e-5f

/* 一个可训练数组。四块内存逐元素一一对应。 */
typedef struct {
    size_t count;
    float *value; /* 参数值 w */
    float *grad;  /* 梯度 dw */
    float *m;     /* Adam 一阶矩 */
    float *v;     /* Adam 二阶矩 */
} Tensor;

/* 一个 pre-norm Transformer block 的全部参数。 */
typedef struct {
    Tensor rms1_gain; /* [D] */
    Tensor wq;        /* [D,D] */
    Tensor wk;        /* [D,D] */
    Tensor wv;        /* [D,D] */
    Tensor wo;        /* [D,D] */
    Tensor rms2_gain; /* [D] */
    Tensor w1;        /* [D,F] */
    Tensor b1;        /* [F] */
    Tensor w2;        /* [F,D] */
    Tensor b2;        /* [D] */
} Block;

struct MiniLLM {
    MiniLLMConfig config;
    uint64_t rng_state;
    uint64_t step;

    Tensor token_embedding;    /* [V,D] */
    Tensor position_embedding; /* [maxT,D] */
    Block *blocks;             /* n_layers 个 block */
    Tensor final_rms_gain;     /* [D] */
    Tensor output_weight;      /* [D,V] */
    Tensor output_bias;        /* [V] */

    /* 统一列表让优化器和 checkpoint 不必了解每个参数的名字。 */
    Tensor **tensors;
    size_t tensor_count;
    size_t tensor_capacity;
    size_t parameter_count;
};

/* 一个 block 为反向传播保留的中间结果。 */
typedef struct {
    float *norm1;       /* [T,D] */
    float *inv_rms1;    /* [T] */
    float *q;           /* [T,D] */
    float *k;           /* [T,D] */
    float *v;           /* [T,D] */
    float *prob;        /* [H,T,T] */
    float *attention;   /* [T,D] */
    float *projection;  /* [T,D] */
    float *residual1;   /* [T,D] */
    float *norm2;       /* [T,D] */
    float *inv_rms2;    /* [T] */
    float *hidden_pre;  /* [T,F] */
    float *hidden;      /* [T,F] */
    float *mlp;         /* [T,D] */
    float *output;      /* [T,D] */
} BlockCache;

typedef struct {
    int token_count;
    BlockCache *blocks;
    float *embedding;     /* [T,D] */
    float *final_norm;    /* [T,D] */
    float *final_inv_rms; /* [T] */
    float *logits;        /* [T,V] */
    float loss;
} ForwardCache;

/* core/memory.c */
void *minillm_calloc(size_t count, size_t size);
int minillm_tensor_allocate(Tensor *tensor, size_t count);
void minillm_tensor_free(Tensor *tensor);

/* core/random.c */
uint64_t minillm_random_u64(MiniLLM *model);
float minillm_random_uniform(MiniLLM *model);
float minillm_random_normal(MiniLLM *model);

/* core/cache.c */
ForwardCache *minillm_cache_create(const MiniLLM *model, int token_count);
void minillm_cache_destroy(const MiniLLM *model, ForwardCache *cache);

/* layers/linear.c */
void minillm_linear_forward(const float *x, const Tensor *weight,
                            const Tensor *bias, float *y,
                            int rows, int input_dim, int output_dim);
void minillm_linear_backward(const float *x, Tensor *weight, Tensor *bias,
                             const float *dy, float *dx,
                             int rows, int input_dim, int output_dim);

/* layers/rmsnorm.c */
void minillm_rms_forward(const float *x, const float *gain, float *y,
                         float *inverse_rms, int rows, int columns);
void minillm_rms_backward(const float *x, Tensor *gain,
                          const float *inverse_rms, const float *dy,
                          float *dx, int rows, int columns);

/* layers/attention.c */
void minillm_attention_forward(const float *q, const float *k, const float *v,
                               float *probability, float *output,
                               int token_count, int d_model, int heads);
int minillm_attention_backward(const float *q, const float *k, const float *v,
                               const float *probability, const float *d_output,
                               float *d_q, float *d_k, float *d_v,
                               int token_count, int d_model, int heads);

/* model/forward.c 与 model/backward.c */
ForwardCache *minillm_forward(MiniLLM *model, const unsigned char *tokens,
                              const unsigned char *targets, int token_count);
int minillm_backward(MiniLLM *model, const unsigned char *tokens,
                     const unsigned char *targets, ForwardCache *cache);

#endif /* MINILLM_INTERNAL_H */
