#include "minillm.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * CLI 只负责文件、参数解析和进度显示；模型逻辑按主题拆分在其他子目录。
 * 这种分层让库既能被命令行调用，也能嵌入你自己的 C 程序。
 */

static void print_usage(const char *program) {
    fprintf(stderr,
        "miniLLM - pure C11 educational Transformer\n\n"
        "Usage:\n"
        "  %s train <corpus.txt> <model.bin> [steps] [learning_rate]\n"
        "  %s resume <model.bin> <corpus.txt> [steps] [learning_rate]\n"
        "  %s generate <model.bin> [prompt] [new_bytes] [temperature] [top_k]\n"
        "  %s info <model.bin>\n\n"
        "Examples:\n"
        "  %s train data/tiny.txt tiny.bin 1000 0.003\n"
        "  %s generate tiny.bin \"the \" 300 0.8 32\n",
        program, program, program, program, program, program);
}

static unsigned char *read_file(const char *path, size_t *size_out) {
    FILE *file = fopen(path, "rb");
    long length;
    unsigned char *data;
    if (file == NULL) {
        fprintf(stderr, "error: cannot open '%s'\n", path);
        return NULL;
    }
    if (fseek(file, 0L, SEEK_END) != 0 || (length = ftell(file)) < 0L ||
        fseek(file, 0L, SEEK_SET) != 0) {
        fprintf(stderr, "error: cannot determine size of '%s'\n", path);
        fclose(file);
        return NULL;
    }
    data = malloc((size_t)length + 1u);
    if (data == NULL || fread(data, 1u, (size_t)length, file) != (size_t)length) {
        fprintf(stderr, "error: cannot read '%s'\n", path);
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *size_out = (size_t)length;
    return data;
}

static int parse_int(const char *text, int fallback) {
    char *end = NULL;
    long value;
    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0L ||
        value > 100000000L) {
        return fallback;
    }
    return (int)value;
}

static float parse_float(const char *text, float fallback) {
    char *end = NULL;
    float value;
    errno = 0;
    value = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) {
        return fallback;
    }
    return value;
}

static int train_model(MiniLLM *model, const char *corpus_path,
                       const char *checkpoint_path, int steps,
                       float learning_rate) {
    size_t data_size = 0u;
    unsigned char *data = read_file(corpus_path, &data_size);
    MiniLLMOptimizer optimizer = minillm_default_optimizer();
    MiniLLMConfig config;
    int step;
    double running_loss = 0.0;
    const int report_every = steps < 20 ? 1 : steps / 20;
    if (data == NULL) {
        return 0;
    }
    config = minillm_config(model);
    if (data_size < (size_t)config.context_length + 1u) {
        fprintf(stderr, "error: corpus needs at least %d bytes (got %zu)\n",
                config.context_length + 1, data_size);
        free(data);
        return 0;
    }
    optimizer.learning_rate = learning_rate;
    printf("training: %d steps, %zu corpus bytes, %zu parameters\n",
           steps, data_size, minillm_parameter_count(model));
    printf("model: context=%d d_model=%d heads=%d layers=%d hidden=%d\n",
           config.context_length, config.d_model, config.n_heads,
           config.n_layers, config.hidden_dim);

    for (step = 1; step <= steps; ++step) {
        const float loss = minillm_train_step(model, data, data_size, optimizer);
        if (!isfinite(loss)) {
            fprintf(stderr, "error: loss became non-finite at step %d\n", step);
            free(data);
            return 0;
        }
        running_loss += (double)loss;
        if (step % report_every == 0 || step == steps) {
            const int window = step % report_every == 0 ? report_every
                                                        : step % report_every;
            printf("step %6llu | loss %.4f | perplexity %.2f\n",
                   (unsigned long long)minillm_training_step(model),
                   (float)(running_loss / (double)window),
                   exp((double)(running_loss / (double)window)));
            running_loss = 0.0;
        }
    }
    free(data);
    if (!minillm_save(model, checkpoint_path)) {
        fprintf(stderr, "error: failed to save '%s'\n", checkpoint_path);
        return 0;
    }
    printf("saved checkpoint: %s\n", checkpoint_path);
    return 1;
}

static int command_train(int argc, char **argv) {
    MiniLLMConfig config = minillm_default_config();
    MiniLLM *model;
    int steps = argc >= 5 ? parse_int(argv[4], 1000) : 1000;
    float learning_rate = argc >= 6 ? parse_float(argv[5], 3.0e-3f) : 3.0e-3f;
    int ok;
    model = minillm_create(config);
    if (model == NULL) {
        fprintf(stderr, "error: model allocation failed\n");
        return 0;
    }
    ok = train_model(model, argv[2], argv[3], steps, learning_rate);
    minillm_destroy(model);
    return ok;
}

static int command_resume(int argc, char **argv) {
    MiniLLM *model = minillm_load(argv[2]);
    int steps = argc >= 5 ? parse_int(argv[4], 500) : 500;
    float learning_rate = argc >= 6 ? parse_float(argv[5], 1.0e-3f) : 1.0e-3f;
    int ok;
    if (model == NULL) {
        fprintf(stderr, "error: cannot load '%s'\n", argv[2]);
        return 0;
    }
    ok = train_model(model, argv[3], argv[2], steps, learning_rate);
    minillm_destroy(model);
    return ok;
}

static int command_generate(int argc, char **argv) {
    MiniLLM *model = minillm_load(argv[2]);
    const char *prompt = argc >= 4 ? argv[3] : "the ";
    int new_bytes = argc >= 5 ? parse_int(argv[4], 300) : 300;
    float temperature = argc >= 6 ? parse_float(argv[5], 0.8f) : 0.8f;
    int top_k = argc >= 7 ? parse_int(argv[6], 32) : 32;
    unsigned char *output;
    int ok;
    if (model == NULL) {
        fprintf(stderr, "error: cannot load '%s'\n", argv[2]);
        return 0;
    }
    output = malloc((size_t)new_bytes);
    if (output == NULL) {
        minillm_destroy(model);
        return 0;
    }
    ok = minillm_generate(model, (const unsigned char *)prompt, strlen(prompt),
                          output, (size_t)new_bytes, temperature, top_k);
    if (ok) {
        fwrite(prompt, 1u, strlen(prompt), stdout);
        fwrite(output, 1u, (size_t)new_bytes, stdout);
        fputc('\n', stdout);
    }
    free(output);
    minillm_destroy(model);
    return ok;
}

static int command_info(const char *path) {
    MiniLLM *model = minillm_load(path);
    MiniLLMConfig config;
    if (model == NULL) {
        fprintf(stderr, "error: cannot load '%s'\n", path);
        return 0;
    }
    config = minillm_config(model);
    printf("checkpoint: %s\n", path);
    printf("parameters: %zu\n", minillm_parameter_count(model));
    printf("training step: %llu\n",
           (unsigned long long)minillm_training_step(model));
    printf("vocab=%d context=%d d_model=%d heads=%d layers=%d hidden=%d\n",
           config.vocab_size, config.context_length, config.d_model,
           config.n_heads, config.n_layers, config.hidden_dim);
    minillm_destroy(model);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1], "train") == 0 && argc >= 4) {
        return command_train(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (strcmp(argv[1], "resume") == 0 && argc >= 4) {
        return command_resume(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (strcmp(argv[1], "generate") == 0 && argc >= 3) {
        return command_generate(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (strcmp(argv[1], "info") == 0 && argc >= 3) {
        return command_info(argv[2]) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    print_usage(argv[0]);
    return EXIT_FAILURE;
}
