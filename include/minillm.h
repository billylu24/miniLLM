#ifndef MINILLM_H
#define MINILLM_H

/*
 * miniLLM：一个只依赖 C11 标准库的教学用 decoder-only Transformer。
 *
 * 公开头文件只暴露“如何使用模型”。具体实现按学习主题拆分在 src/core、
 * src/layers、src/model、src/training、src/inference 和 src/io 中。
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MiniLLM MiniLLM;

typedef struct {
    int vocab_size;      /* byte tokenizer 固定为 256。 */
    int context_length;  /* 模型一次最多看到多少个 byte。 */
    int d_model;         /* 每个 token 的向量维度。 */
    int n_heads;         /* 注意力头数；d_model 必须能被它整除。 */
    int n_layers;        /* Transformer block 的数量。 */
    int hidden_dim;      /* MLP 中间层宽度，通常取 2~4 倍 d_model。 */
    uint64_t seed;       /* 参数初始化和采样使用的随机种子。 */
} MiniLLMConfig;

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float grad_clip;     /* 全局梯度范数上限；<= 0 表示不裁剪。 */
} MiniLLMOptimizer;

/* 一套能在普通电脑上快速运行的默认教学配置。 */
MiniLLMConfig minillm_default_config(void);
MiniLLMOptimizer minillm_default_optimizer(void);

MiniLLM *minillm_create(MiniLLMConfig config);
void minillm_destroy(MiniLLM *model);
MiniLLMConfig minillm_config(const MiniLLM *model);

/*
 * 只做前向传播并返回平均交叉熵。token_count 必须位于 [1, context_length]。
 * tokens[i] 的正确答案是 targets[i]。
 */
float minillm_loss(MiniLLM *model,
                   const unsigned char *tokens,
                   const unsigned char *targets,
                   int token_count);

/* 前向 + 手写反向传播。返回 loss，并把每个参数的梯度留在模型中。 */
float minillm_loss_and_backward(MiniLLM *model,
                                const unsigned char *tokens,
                                const unsigned char *targets,
                                int token_count);

/* 使用刚刚算出的梯度执行一次 AdamW 更新。 */
void minillm_apply_gradients(MiniLLM *model, MiniLLMOptimizer optimizer);

/*
 * 从 data 中随机截取 context_length+1 个 byte，完成一次训练更新。
 * data_size 至少应为 context_length+1。
 */
float minillm_train_step(MiniLLM *model,
                         const unsigned char *data,
                         size_t data_size,
                         MiniLLMOptimizer optimizer);

/*
 * 根据 prompt 生成 continuation_size 个新 byte。
 * temperature <= 0 时使用贪心解码；top_k <= 0 表示不做 top-k 截断。
 */
int minillm_generate(MiniLLM *model,
                     const unsigned char *prompt,
                     size_t prompt_size,
                     unsigned char *continuation,
                     size_t continuation_size,
                     float temperature,
                     int top_k);

/* checkpoint 同时保存参数、Adam 一二阶矩、训练步数和随机数状态。 */
int minillm_save(const MiniLLM *model, const char *path);
MiniLLM *minillm_load(const char *path);

/* 以下接口主要服务于测试、梯度检查和学习参数布局。 */
size_t minillm_parameter_count(const MiniLLM *model);
float minillm_parameter_value(const MiniLLM *model, size_t index);
float minillm_parameter_gradient(const MiniLLM *model, size_t index);
int minillm_set_parameter_value(MiniLLM *model, size_t index, float value);
uint64_t minillm_training_step(const MiniLLM *model);

#ifdef __cplusplus
}
#endif

#endif /* MINILLM_H */
