#include "supports.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(bool result, const char * message) {
    if (!result) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

static ggml_tensor tensor() {
    ggml_tensor result{};
    result.type = GGML_TYPE_F32;
    result.ne[0] = 4;
    result.ne[1] = result.ne[2] = result.ne[3] = 1;
    result.nb[0] = 4;
    result.nb[1] = result.nb[2] = result.nb[3] = 16;
    return result;
}

static void test_matmul_support() {
    const auto base = tensor();
    auto weights = base;
    auto input = base;
    auto product = base;
    weights.ne[0] = input.ne[0] = 64;
    weights.ne[1] = input.ne[1] = 32;
    product.op = GGML_OP_MUL_MAT;
    product.src[0] = &weights;
    product.src[1] = &input;
    ggml_cannge_options options;
    auto expect = [&](const char * expected) {
        const char * reason = "stale";
        check(ggml_cannge_supports_op(&product, options, &reason) == (expected == nullptr), "matmul support outcome");
        check(expected == nullptr ? reason == nullptr : reason != nullptr && std::strcmp(reason, expected) == 0,
              "matmul constraint diagnostic");
    };
    expect(nullptr);
    product.src[0] = nullptr;
    expect("matmul requires weights and input tensors");
    product.src[0] = &weights; product.type = GGML_TYPE_I32;
    expect("matmul weights, input and output must be F16/F32");
    product.type = GGML_TYPE_F32; input.ne[0] = 32;
    expect("matmul contraction dimensions must match");
    input.ne[0] = 64; input.type = GGML_TYPE_F16;
    expect("batched matmul requires matching types or F16 weights with F32 input");
    input.type = GGML_TYPE_F32; weights.type = GGML_TYPE_F16;
    expect(nullptr);
    for (auto * dimension : {&weights.ne[0], &weights.ne[1], &input.ne[1]}) {
        const auto saved = *dimension;
        *dimension = 17;
        if (dimension == &weights.ne[0]) { input.ne[0] = 17; }
        expect("batched matmul M/N/K dimensions do not meet GE alignment requirements");
        *dimension = saved;
        input.ne[0] = weights.ne[0];
    }
    weights.ne[2] = 2;
    expect("batched matmul requires matching ranks from two through four");
    input.ne[2] = 3;
    expect("batched matmul batch dimensions must match");
    input.ne[2] = weights.ne[2] = 1;
    weights.ne[0] = input.ne[0] = 512; weights.ne[1] = 512;
    options.limit_matmul = true;
    expect(nullptr);
    weights.ne[1] += 16;
    expect("matmul weight matrix exceeds GGML_CANNGE_BIGMM limit");
    options.limit_matmul = false;
    expect(nullptr);
    options.limit_matmul = true; input.ne[1] = 1;
    expect(nullptr); // Chunked GEMV bypasses the BatchMatMul size cap.
    weights.type = GGML_TYPE_F32; input.type = GGML_TYPE_F16;
    weights.ne[0] = input.ne[0] = 17;
    expect(nullptr); // GEMV casts mixed types and accepts unaligned contractions.
}

static void test_rope_support() {
    const auto base = tensor();
    auto data = base;
    auto positions = base;
    auto rope = base;
    data.ne[1] = 2;
    data.nb[2] = data.nb[3] = 32;
    positions.type = GGML_TYPE_I32;
    rope.op = GGML_OP_ROPE;
    rope.src[0] = &data;
    rope.src[1] = &positions;
    rope.op_params[1] = 4;
    auto set_float = [&](int index, float value) {
        std::memcpy(rope.op_params + index, &value, sizeof(value));
    };
    set_float(6, 1.0f);
    set_float(8, 1.0f);
    auto expect = [&](const char * expected) {
        const char * reason = "stale";
        check(ggml_cannge_supports_op(&rope, {}, &reason) == (expected == nullptr), "RoPE support outcome");
        check(expected == nullptr ? reason == nullptr : reason != nullptr && std::strcmp(reason, expected) == 0,
              "RoPE constraint diagnostic");
    };
    for (int mode : {GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX}) {
        rope.op_params[2] = mode;
        expect(nullptr);
    }
    rope.src[0] = nullptr;
    expect("RoPE requires data and position tensors");
    rope.src[0] = &data; rope.src[1] = nullptr;
    expect("RoPE requires data and position tensors");
    rope.src[1] = &positions; rope.src[2] = &data;
    expect("RoPE frequency factors are unsupported");
    rope.src[2] = nullptr; rope.type = GGML_TYPE_F16;
    expect("RoPE data and output must have matching F16/F32 types");
    data.type = GGML_TYPE_F16; data.nb[0] = 2; data.nb[1] = 8; data.nb[2] = data.nb[3] = 16;
    expect(nullptr);
    positions.type = GGML_TYPE_I64;
    expect("RoPE positions must be I32");
    positions.type = GGML_TYPE_I32; rope.op_params[2] = -1;
    expect("RoPE supports NORMAL and NEOX modes only");
    rope.op_params[2] = GGML_ROPE_TYPE_NORMAL; rope.op_params[1] = 2;
    expect("RoPE requires full-head rotation with an even width");
    data.ne[0] = rope.op_params[1] = 3;
    expect("RoPE requires full-head rotation with an even width");
    data.ne[0] = rope.op_params[1] = 4; rope.op_params[15] = 1;
    expect("RoPE offsets are unsupported");
    rope.op_params[15] = 0; set_float(6, 0.5f);
    expect("RoPE frequency scaling is unsupported");
    set_float(6, 1.0f); set_float(7, 1.0f);
    expect("RoPE YaRN scaling is unsupported");
    set_float(7, 0.0f); set_float(8, 0.5f);
    expect("RoPE YaRN scaling is unsupported");
    set_float(8, 1.0f); set_float(9, 32.0f); set_float(10, 1.0f);
    expect(nullptr); // Beta parameters are inactive without YaRN.
    data.nb[1] *= 2;
    expect("RoPE requires contiguous data with rank two through four");
    data.nb[1] /= 2; data.ne[1] = 1;
    expect("RoPE requires contiguous data with rank two through four");
}

static void test_gemv_support() {
    auto base = tensor();
    ggml_cannge_options options;
    auto weights = base;
    auto activation = base;
    auto product = base;
    weights.ne[1] = 32; weights.ne[2] = 2;
    activation.ne[2] = 8;
    product.op = GGML_OP_MUL_MAT;
    product.src[0] = &weights; product.src[1] = &activation;
    for (int64_t contraction : {64, 128}) {
        weights.ne[0] = activation.ne[0] = contraction;
        for (const char * mode : {"", "kq", "v", "all", "invalid"}) {
            options = {};
            options.gqa_present = true;
            options.gqa_mode = mode;
            const bool expected = std::strcmp(mode, "all") == 0 ||
                                  std::strcmp(mode, contraction == 128 ? "kq" : "v") == 0;
            check(ggml_cannge_use_gemv(&weights, &activation, options) == expected, "GEMV GQA mode");
            check(ggml_cannge_supports_op(&product, options) == expected, "scheduler matches GEMV selection");
            options.gqa_present = false;
            check(!ggml_cannge_use_gemv(&weights, &activation, options), "GQA requires opt-in");
        }
    }
    options = {}; options.gqa_present = true; options.gqa_mode = "all";
    activation.ne[2] = 7;
    check(!ggml_cannge_use_gemv(&weights, &activation, options), "GQA requires integral head ratio");
    activation.ne[2] = weights.ne[2];
    options = {};
    check(ggml_cannge_use_gemv(&weights, &activation, options), "equal heads require no GQA option");
    activation.ne[3] = 2;
    check(!ggml_cannge_use_gemv(&weights, &activation, options), "GEMV batch dimensions must match");
    activation.ne[3] = 1; activation.ne[1] = 2;
    check(!ggml_cannge_use_gemv(&weights, &activation, options), "matrix input is not GEMV");

}

static void test_attention_support() {
    auto base = tensor();
    ggml_cannge_options options;
    auto query = base;
    auto key_tensor = base;
    auto value_tensor = base;
    auto attention = base;
    query.ne[0] = key_tensor.ne[0] = value_tensor.ne[0] = 64;
    query.ne[2] = 8;
    key_tensor.ne[2] = value_tensor.ne[2] = 2;
    key_tensor.ne[1] = value_tensor.ne[1] = 16;
    key_tensor.type = value_tensor.type = GGML_TYPE_F16;
    attention.op = GGML_OP_FLASH_ATTN_EXT;
    attention.src[0] = &query; attention.src[1] = &key_tensor; attention.src[2] = &value_tensor;
    options = {}; options.flash_attention = true;
    auto attention_reason = [&](const char * expected) {
        const char * reason = "stale";
        const bool supported = ggml_cannge_supports_op(&attention, options, &reason);
        check(supported == (expected == nullptr), "attention support outcome");
        check(expected == nullptr ? reason == nullptr : reason != nullptr && std::strcmp(reason, expected) == 0,
              "attention constraint diagnostic");
    };
    attention_reason(nullptr);
    key_tensor.ne[2] = 0;
    attention_reason("flash attention requires positive matching KV heads and an integral query/KV head ratio");
    key_tensor.ne[2] = 2; value_tensor.ne[2] = 1;
    attention_reason("flash attention requires positive matching KV heads and an integral query/KV head ratio");
    value_tensor.ne[2] = 2; query.ne[2] = 7;
    attention_reason("flash attention requires positive matching KV heads and an integral query/KV head ratio");
    query.ne[2] = 8;
    for (auto * operand : {&query, &key_tensor, &value_tensor}) {
        operand->ne[3] = 2;
        attention_reason("flash attention supports batch size one only");
        operand->ne[3] = 1;
    }
    query.ne[1] = 2;
    attention_reason("flash attention supports decode only (query sequence length must be one)");
    query.ne[1] = 1; query.ne[0] = 0;
    attention_reason("flash attention head widths must match and be positive multiples of 16");
    query.ne[0] = 64; value_tensor.ne[1] = 15;
    attention_reason("flash attention key and value sequence lengths must match and be positive");
    value_tensor.ne[1] = 16; attention.src[4] = &base;
    attention_reason("flash attention sinks are unsupported");
    attention.src[4] = nullptr;
    float unsupported_bias = 1.0f;
    std::memcpy(attention.op_params + 1, &unsupported_bias, sizeof(unsupported_bias));
    attention_reason("flash attention ALiBi is unsupported");
    attention.op_params[1] = 0;
    std::memcpy(attention.op_params + 2, &unsupported_bias, sizeof(unsupported_bias));
    attention_reason("flash attention softcapping is unsupported");
    attention.op_params[2] = 0;
    attention_reason(nullptr);

}

int main() {
    test_matmul_support();
    test_rope_support();
    test_gemv_support();
    test_attention_support();
    std::puts("PASS: CANNGE capability constraints and selection policy");
}
