#include "minillm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition, message)                                      \
    do {                                                               \
        if (!(condition)) {                                            \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,  \
                    (message));                                        \
            ++failures;                                                \
        }                                                              \
    } while (0)

static MiniLLMConfig test_config(void) {
    MiniLLMConfig config;
    config.vocab_size = 256;
    config.context_length = 4;
    config.d_model = 8;
    config.n_heads = 2;
    config.n_layers = 1;
    config.hidden_dim = 16;
    config.seed = 12345u;
    return config;
}

/*
 * 数值梯度检查不相信我们的反向传播公式。它只用定义：
 *   df/dx ~= (f(x+epsilon)-f(x-epsilon))/(2*epsilon)
 * 再把这个近似值与 backward() 算出的解析梯度比较。
 */
static void test_gradient_check(void) {
    MiniLLM *model = minillm_create(test_config());
    const unsigned char tokens[4] = {'a', 'b', 'c', 'a'};
    const unsigned char targets[4] = {'b', 'c', 'a', 'b'};
    const float epsilon = 1.0e-3f;
    size_t index;
    size_t chosen = (size_t)-1;
    float analytical = 0.0f;
    float original;
    float loss_plus;
    float loss_minus;
    float numerical;
    float relative_error;

    CHECK(model != NULL, "model creation for gradient check");
    if (model == NULL) return;
    CHECK(isfinite(minillm_loss_and_backward(model, tokens, targets, 4)),
          "backward loss is finite");
    for (index = 0u; index < minillm_parameter_count(model); ++index) {
        const float gradient = minillm_parameter_gradient(model, index);
        if (fabsf(gradient) > 1.0e-4f) {
            chosen = index;
            analytical = gradient;
            break;
        }
    }
    CHECK(chosen != (size_t)-1, "found a non-zero gradient");
    if (chosen != (size_t)-1) {
        original = minillm_parameter_value(model, chosen);
        minillm_set_parameter_value(model, chosen, original + epsilon);
        loss_plus = minillm_loss(model, tokens, targets, 4);
        minillm_set_parameter_value(model, chosen, original - epsilon);
        loss_minus = minillm_loss(model, tokens, targets, 4);
        minillm_set_parameter_value(model, chosen, original);
        numerical = (loss_plus - loss_minus) / (2.0f * epsilon);
        relative_error = fabsf(analytical - numerical) /
                         fmaxf(1.0e-4f, fabsf(analytical) + fabsf(numerical));
        printf("gradient: analytical=%g numerical=%g relative_error=%g\n",
               (double)analytical, (double)numerical, (double)relative_error);
        CHECK(relative_error < 3.0e-2f, "analytical gradient matches finite difference");
    }
    minillm_destroy(model);
}

static void test_training_reduces_loss(void) {
    MiniLLM *model = minillm_create(test_config());
    const unsigned char data[] = "abcabcabcabcabcabcabcabcabcabcabcabcabcabc";
    const unsigned char tokens[4] = {'a', 'b', 'c', 'a'};
    const unsigned char targets[4] = {'b', 'c', 'a', 'b'};
    MiniLLMOptimizer optimizer = minillm_default_optimizer();
    float initial;
    float final;
    int step;
    CHECK(model != NULL, "model creation for training test");
    if (model == NULL) return;
    optimizer.learning_rate = 1.0e-2f;
    optimizer.weight_decay = 0.0f;
    initial = minillm_loss(model, tokens, targets, 4);
    for (step = 0; step < 160; ++step) {
        const float loss = minillm_train_step(model, data, sizeof(data) - 1u,
                                              optimizer);
        CHECK(isfinite(loss), "training loss remains finite");
        if (!isfinite(loss)) break;
    }
    final = minillm_loss(model, tokens, targets, 4);
    printf("overfit: initial_loss=%g final_loss=%g\n",
           (double)initial, (double)final);
    CHECK(final < initial * 0.35f, "tiny repeated corpus can be overfit");
    CHECK(minillm_training_step(model) == 160u, "optimizer step counter advances");
    minillm_destroy(model);
}

static void test_checkpoint_and_generation(void) {
    static const char *path = "minillm_test_checkpoint.bin";
    MiniLLM *model = minillm_create(test_config());
    MiniLLM *loaded;
    const unsigned char data[] = "hello hello hello hello hello hello";
    const unsigned char prompt[] = "hel";
    unsigned char first[24];
    unsigned char second[24];
    MiniLLMOptimizer optimizer = minillm_default_optimizer();
    size_t i;
    CHECK(model != NULL, "model creation for checkpoint test");
    if (model == NULL) return;
    optimizer.learning_rate = 5.0e-3f;
    for (i = 0u; i < 8u; ++i) {
        (void)minillm_train_step(model, data, sizeof(data) - 1u, optimizer);
    }
    CHECK(minillm_save(model, path), "checkpoint save succeeds");
    loaded = minillm_load(path);
    CHECK(loaded != NULL, "checkpoint load succeeds");
    if (loaded != NULL) {
        CHECK(minillm_parameter_count(model) == minillm_parameter_count(loaded),
              "parameter count survives checkpoint");
        CHECK(minillm_training_step(model) == minillm_training_step(loaded),
              "training step survives checkpoint");
        for (i = 0u; i < minillm_parameter_count(model); ++i) {
            if (minillm_parameter_value(model, i) !=
                minillm_parameter_value(loaded, i)) {
                CHECK(0, "all checkpoint parameters are bit-identical");
                break;
            }
        }
        CHECK(minillm_generate(model, prompt, sizeof(prompt) - 1u,
                               first, sizeof(first), 0.8f, 16),
              "generation from original model succeeds");
        CHECK(minillm_generate(loaded, prompt, sizeof(prompt) - 1u,
                               second, sizeof(second), 0.8f, 16),
              "generation from loaded model succeeds");
        CHECK(memcmp(first, second, sizeof(first)) == 0,
              "saved RNG state makes generation reproducible");
    }
    minillm_destroy(loaded);
    minillm_destroy(model);
    CHECK(remove(path) == 0, "temporary checkpoint removed");
}

static void test_invalid_inputs(void) {
    MiniLLMConfig config = test_config();
    MiniLLM *model;
    config.d_model = 7; /* 不能被 2 个 head 均分。 */
    model = minillm_create(config);
    CHECK(model == NULL, "invalid head dimension is rejected");
    minillm_destroy(model);
}

int main(void) {
    test_invalid_inputs();
    test_gradient_check();
    test_training_reduces_loss();
    test_checkpoint_and_generation();
    if (failures != 0) {
        fprintf(stderr, "%d test assertion(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    printf("all miniLLM tests passed\n");
    return EXIT_SUCCESS;
}
