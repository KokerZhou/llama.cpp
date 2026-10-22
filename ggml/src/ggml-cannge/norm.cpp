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

#include "norm.h"
#include "graph-build-priv.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

bool ggml_cannge_build_norm(ggml_cannge_build_context & ctx, ggml_tensor * node) {
    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];

    cannge_es::EsTensorHolder es0;
    cannge_es::EsTensorHolder es1;

    switch (node->op) {
        case GGML_OP_SOFT_MAX: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            float scale    = 1.0f;
            float max_bias = 0.0f;
            std::memcpy(&scale, (const float *) node->op_params + 0, sizeof(scale));
            std::memcpy(&max_bias, (const float *) node->op_params + 1, sizeof(max_bias));
            // mirrors the L0 gate: attention sinks and ALiBi slopes unsupported
            if (node->src[2] != nullptr || max_bias != 0.0f) {
                return ctx.fail("soft_max with sinks or ALiBi unsupported");
            }
            // ggml: softmax(x*scale + mask) over the innermost dim
            // (ggml-cpu ops.cpp ggml_compute_forward_soft_max_f32)
            auto es = scale == 1.0f ? es0 : cannge_es::Mul(es0, ctx.scalar_cached(scale));
            if (src1 != nullptr) {
                if (!ctx.get(src1, es1)) {
                    return false;
                }
                if (src1->type != src0->type) {
                    ge::DataType dt;
                    if (!ggml_cannge_ge_dtype(src0->type, dt)) {
                        return ctx.fail(std::string("unsupported soft_max mask dtype ") + src1->name);
                    }
                    es1 = cannge_es::Cast(es1, (int64_t) dt);
                }
                es = cannge_es::Add(es, es1);
            }
            ctx.tensors[node] = cannge_es::SoftmaxV2(es, { -1 });
            return true;
        }

        case GGML_OP_RMS_NORM: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            float eps;
            std::memcpy(&eps, node->op_params, sizeof(eps));
            // the fused RmsNorm kernel drops rows of the output on small
            // shapes (seen on 310P3), build it from primitives instead:
            // y = x * rsqrt(mean(x*x, axis=-1, keepdims) + eps)
            // ReduceMean: x/axes are required inputs, keep_dims an optional
            // attr; axes is a 1-D int32/int64 tensor
            // ge 仓 tests/st/graph/compiler/testcase/eager_style_graph_builder/all_ops.cpp:39019-39048 @00ecb5c
            // include/es/es_ReduceMean.h
            const int nd = ggml_n_dims(src0);
            std::vector<int64_t> axes = { (int64_t) nd - 1 }; // GE: last axis
            auto ms = cannge_es::ReduceMean(cannge_es::Square(es0), ctx.vector_cached(axes), true);
            auto den = cannge_es::Rsqrt(cannge_es::Add(ms, ctx.scalar_cached(eps)));
            if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
                fprintf(stderr, "[DBG-RMS] %s ne=%lld,%lld,%lld,%lld\n", node->name, (long long) src0->ne[0],
                        (long long) src0->ne[1], (long long) src0->ne[2], (long long) src0->ne[3]);
            }
            auto out = cannge_es::Mul(es0, den);
            ctx.tensors[node] = out;
            return true;
        }

        case GGML_OP_NORM: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            float eps;
            std::memcpy(&eps, node->op_params, sizeof(eps));
            // ggml NORM is layer norm without affine params: feed unit gamma and
            // zero beta of the normalized size (ggml ne[0] = last GE axis). The
            // fused kernel may share the 310P row-drop issue of RmsNorm
            // (note-10), fall back to the RMS_NORM primitive shape if seen
            const int64_t      n = src0->ne[0];
            std::vector<float> ones(n, 1.0f);
            std::vector<float> zeros(n, 0.0f);
            auto gamma = ctx.const_cached(ones, { n });
            auto beta  = ctx.const_cached(zeros, { n });
            auto ln = cannge_es::LayerNorm(es0, gamma, beta, (int64_t) ggml_n_dims(src0) - 1,
                                           (int64_t) ggml_n_dims(src0) - 1, eps); // include/es/es_nn/es_LayerNorm.h
            ctx.tensors[node] = ln.y;
            return true;
        }

        default:
            return ctx.fail("unsupported norm operation");
    }
}
