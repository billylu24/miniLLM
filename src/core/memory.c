#include "internal/minillm_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * 为什么单独放一个内存模块？
 * 神经网络的“张量”在最底层其实就是连续数组。先看懂这里，再读矩阵运算时，
 * 就不会把 Tensor 想成某种神秘对象。
 */

void *minillm_calloc(size_t count, size_t size) {
    /* 先检查乘法溢出，否则 count*size 可能绕回一个很小的数。 */
    if (count == 0u || size == 0u || count > SIZE_MAX / size) {
        return NULL;
    }
    return calloc(count, size);
}

int minillm_tensor_allocate(Tensor *tensor, size_t count) {
    memset(tensor, 0, sizeof(*tensor));
    tensor->count = count;
    tensor->value = minillm_calloc(count, sizeof(float));
    tensor->grad = minillm_calloc(count, sizeof(float));
    tensor->m = minillm_calloc(count, sizeof(float));
    tensor->v = minillm_calloc(count, sizeof(float));
    return tensor->value != NULL && tensor->grad != NULL &&
           tensor->m != NULL && tensor->v != NULL;
}

void minillm_tensor_free(Tensor *tensor) {
    free(tensor->value);
    free(tensor->grad);
    free(tensor->m);
    free(tensor->v);
    memset(tensor, 0, sizeof(*tensor));
}
