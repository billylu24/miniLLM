#include "internal/minillm_internal.h"

#include <math.h>

/*
 * 随机数有两个用途：初始化参数、从概率分布采样下一个 token。
 * 固定 seed 会得到完全相同的随机序列，因此测试可以复现。
 */

uint64_t minillm_random_u64(MiniLLM *model) {
    /* xorshift64*：三次异或移位更新内部状态。它不适合密码学，但足够教学。 */
    uint64_t x = model->rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    model->rng_state = x;
    return x * UINT64_C(2685821657736338717);
}

float minillm_random_uniform(MiniLLM *model) {
    /* 取高 24 bit，除以 2^24，得到 [0,1) 均匀分布。 */
    return (float)(minillm_random_u64(model) >> 40) / 16777216.0f;
}

float minillm_random_normal(MiniLLM *model) {
    /* Box-Muller：把两个均匀随机数变换为均值 0、方差 1 的正态随机数。 */
    float u1 = minillm_random_uniform(model);
    const float u2 = minillm_random_uniform(model);
    if (u1 < 1.0e-7f) {
        u1 = 1.0e-7f; /* 防止 log(0)。 */
    }
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853071795864769f * u2);
}
