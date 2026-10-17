/*
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "elementwise.h"
#include "graph-build-priv.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

bool ggml_cannge_build_elementwise(ggml_cannge_build_context & ctx, ggml_tensor * node) {
    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];

    cannge_es::EsTensorHolder es0;
    cannge_es::EsTensorHolder es1;

    auto binary = [&](cannge_es::EsTensorHolder (*op)(const cannge_es::EsTensorLike &,
                                                       const cannge_es::EsTensorLike &)) {
        if (!ctx.get(src0, es0) || !ctx.get(src1, es1)) {
            return false;
        }
        // pre-flight broadcast check on the full-rank GE shapes
        // (trailing-aligned, like GE/NumPy): a wrong view-base resolution
        // surfaces here as an incompatible Mul instead of a nondeterministic
        // GE parallel-precompile failure (Mul_2, note-46)
        {
            int64_t a[4] = { src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0] };
            int64_t b[4] = { src1->ne[3], src1->ne[2], src1->ne[1], src1->ne[0] };
            for (int d = 3; d >= 0; d--) {
                if (a[d] != b[d] && a[d] != 1 && b[d] != 1) {
                    fprintf(stderr,
                            "[DBG-BROADCAST] %s %s a(ge)=%lld,%lld,%lld,%lld b(ge)=%lld,%lld,%lld,%lld "
                            "incompatible at ge-dim %d\n",
                            ggml_op_name(node->op), node->name, (long long) a[0], (long long) a[1], (long long) a[2],
                            (long long) a[3], (long long) b[0], (long long) b[1], (long long) b[2], (long long) b[3],
                            d);
                    return ctx.fail("incompatible binary broadcast at GE dimension " + std::to_string(d));
                }
            }
        }
        ctx.tensors[node] = op(es0, es1);
        return true;
    };

    auto binary_dbg = [](const char * nm, ggml_tensor * s0, ggml_tensor * s1) {
        if (s0 != nullptr && s1 != nullptr && getenv("GGML_CANNGE_DEBUG_BUILD")) {
            fprintf(stderr, "[DBG-BIN] %s(%s) a=%lld,%lld,%lld,%lld b=%lld,%lld,%lld,%lld\n", nm,
                    s0->name, (long long) s0->ne[0], (long long) s0->ne[1], (long long) s0->ne[2],
                    (long long) s0->ne[3], (long long) s1->ne[0], (long long) s1->ne[1],
                    (long long) s1->ne[2], (long long) s1->ne[3]);
        }
    };
    switch (node->op) {
        case GGML_OP_ADD:
            binary_dbg("ADD", src0, src1);
            return binary(cannge_es::Add);
        case GGML_OP_SUB:
            binary_dbg("SUB", src0, src1);
            return binary(cannge_es::Sub);
        case GGML_OP_MUL:
            binary_dbg("MUL", src0, src1);
            return binary(cannge_es::Mul);
        case GGML_OP_DIV:
            binary_dbg("DIV", src0, src1);
            return binary(cannge_es::Div);

        case GGML_OP_SCALE: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            float s;
            float b;
            std::memcpy(&s, (const float *) node->op_params + 0, sizeof(s));
            std::memcpy(&b, (const float *) node->op_params + 1, sizeof(b));
            // ggml: dst = src0 * s + b (ggml-cpu ops.cpp ggml_compute_forward_scale_f32)
            auto scaled = cannge_es::Mul(es0, ctx.builder.CreateScalar(s));
            ctx.tensors[node] = b == 0.0f ? scaled : cannge_es::Add(scaled, ctx.builder.CreateScalar(b));
            return true;
        }

        case GGML_OP_UNARY: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_SILU:
                    ctx.tensors[node] = cannge_es::Swish(es0, 1.0f); // silu == swish(scale = 1)
                    return true;
                case GGML_UNARY_OP_ABS:
                    ctx.tensors[node] = cannge_es::Abs(es0); // include/es/es_math/es_Abs.h
                    return true;
                case GGML_UNARY_OP_SGN:
                    ctx.tensors[node] = cannge_es::Sign(es0); // include/es/es_math/es_Sign.h
                    return true;
                case GGML_UNARY_OP_NEG:
                    ctx.tensors[node] = cannge_es::Neg(es0); // include/es/es_math/es_Neg.h
                    return true;
                case GGML_UNARY_OP_TANH:
                    ctx.tensors[node] = cannge_es::Tanh(es0); // include/es/es_math/es_Tanh.h
                    return true;
                case GGML_UNARY_OP_RELU:
                    ctx.tensors[node] = cannge_es::Relu(es0); // include/es/es_nn/es_Relu.h
                    return true;
                case GGML_UNARY_OP_SIGMOID:
                    ctx.tensors[node] = cannge_es::Sigmoid(es0); // include/es/es_nn/es_Sigmoid.h
                    return true;
                case GGML_UNARY_OP_GELU:
                    // ggml GELU is the tanh approximation (ggml-cpu vec.h ggml_gelu_f32)
                    ctx.tensors[node] = cannge_es::GeluV2(es0, "tanh"); // include/es/es_nn/es_GeluV2.h
                    return true;
                case GGML_UNARY_OP_GELU_ERF:
                    ctx.tensors[node] = cannge_es::Gelu(es0); // include/es/es_nn/es_Gelu.h
                    return true;
                case GGML_UNARY_OP_GELU_QUICK:
                    // ggml: x*sigmoid(1.702x) (ggml-cpu vec.h ggml_gelu_quick_f32) == FastGelu
                    ctx.tensors[node] = cannge_es::FastGelu(es0); // include/es/es_nn/es_FastGelu.h
                    return true;
                case GGML_UNARY_OP_EXP:
                    ctx.tensors[node] = cannge_es::Exp(es0); // include/es/es_math/es_Exp.h
                    return true;
                case GGML_UNARY_OP_FLOOR:
                    ctx.tensors[node] = cannge_es::Floor(es0); // include/es/es_math/es_Floor.h
                    return true;
                case GGML_UNARY_OP_CEIL:
                    ctx.tensors[node] = cannge_es::Ceil(es0); // include/es/es_math/es_Ceil.h
                    return true;
                case GGML_UNARY_OP_ROUND:
                    // NB: Ascend Round is half-to-even, ggml uses roundf (half away from zero)
                    ctx.tensors[node] = cannge_es::Round(es0); // include/es/es_math/es_Round.h
                    return true;
                case GGML_UNARY_OP_TRUNC:
                    ctx.tensors[node] = cannge_es::Trunc(es0); // include/es/es_math/es_Trunc.h
                    return true;
                default:
                    return ctx.fail("unsupported unary op");
            }
        }

        case GGML_OP_GLU: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            const enum ggml_glu_op glu_op = ggml_get_glu_op(node);
            if (src1 != nullptr) {
                // two-input swiglu split: y = silu(src0) * src1
                // (ggml_vec_swiglu_f32; silu == Swish with scale 1)
                cannge_es::EsTensorHolder es1;
                if (!ctx.get(src1, es1)) {
                    return false;
                }
                ctx.tensors[node] = cannge_es::Mul(cannge_es::Swish(es0, 1.0f), es1);
                return true;
            }
            if (glu_op == GGML_GLU_OP_SWIGLU) {
                // es SwiGlu = swish(first half) * second half along dim, the
                // ggml single-input swiglu (swapped=0) semantics
                // spec: https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.swiglu.html
                ctx.tensors[node] = cannge_es::SwiGlu(es0, (int64_t) ggml_n_dims(src0) - 1); // include/es/es_nn/es_SwiGlu.h
                return true;
            }
            // remaining modes from primitives: split ne[0] (ggml dim 0 = last
            // GE axis) with Slice, y = act(first half) * second half (ggml-cpu
            // ops.cpp ggml_compute_forward_reglu_f32 with swapped=0)
            const int            nd   = ggml_n_dims(src0);
            const int64_t        half = src0->ne[0] / 2;
            std::vector<int64_t> sz   = ggml_cannge_ge_shape(src0);
            sz[nd - 1]                = half;
            std::vector<int64_t> off(nd, 0);
            auto                 first = cannge_es::Slice(es0, off, sz);
            off[nd - 1]                = half;
            auto                       second = cannge_es::Slice(es0, off, sz);
            switch (glu_op) {
                case GGML_GLU_OP_REGLU:
                    ctx.tensors[node] = cannge_es::Mul(cannge_es::Relu(first), second); // include/es/es_nn/es_Relu.h
                    return true;
                case GGML_GLU_OP_GEGLU:
                    // ggml GEGLU uses the tanh gelu (ggml-cpu vec.h ggml_gelu_f32)
                    ctx.tensors[node] = cannge_es::Mul(cannge_es::GeluV2(first, "tanh"), second); // include/es/es_nn/es_GeluV2.h
                    return true;
                case GGML_GLU_OP_GEGLU_ERF:
                    ctx.tensors[node] = cannge_es::Mul(cannge_es::Gelu(first), second); // include/es/es_nn/es_Gelu.h
                    return true;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ctx.tensors[node] = cannge_es::Mul(cannge_es::FastGelu(first), second); // include/es/es_nn/es_FastGelu.h
                    return true;
                default:
                    return ctx.fail("unsupported glu op");
            }
        }

        default:
            return ctx.fail("unsupported elementwise operation");
    }
}
