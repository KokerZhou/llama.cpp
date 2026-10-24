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

#include "rope.h"
#include "graph-build-priv.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

bool ggml_cannge_build_rope(ggml_cannge_build_context & ctx, ggml_tensor * node) {
    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];

    cannge_es::EsTensorHolder es0;
    cannge_es::EsTensorHolder es1;

    switch (node->op) {
        case GGML_OP_ROPE: {
            if (!ctx.get(src0, es0) || !ctx.get(src1, es1)) {
                return false;
            }
            // minimal subset (enforced by the L0 gate): no YaRN, no freq
            // factors, no n_offs, full-head rotation. theta[t, k] =
            // pos[t] * freq_base^(-2k/n_dims) (ggml-cpu ops.cpp
            // ggml_rope_cache_init theta recurrence, ggml.h rope_ext comment)
            const int     n_dims    = ggml_get_op_params_i32(node, 1);
            const int     mode      = ggml_get_op_params_i32(node, 2);
            const int64_t n_tok     = src1->ne[0];
            const int64_t half      = n_dims / 2;
            float         freq_base = 10000.0f;
            std::memcpy(&freq_base, (const float *) node->op_params + 5, sizeof(freq_base));

            std::vector<float> freqs(half);
            for (int64_t k = 0; k < half; k++) {
                freqs[k] = std::pow(freq_base, -2.0f * (float) k / (float) n_dims);
            }
            const std::vector<int64_t> shape_pos = { n_tok, 1 };
            const std::vector<int64_t> shape_fr  = { 1, half };
            auto pos_f = cannge_es::Reshape(cannge_es::Cast(es1, (int64_t) ge::DT_FLOAT), shape_pos);
            auto theta = cannge_es::Mul(pos_f, ctx.const_cached(freqs, shape_fr)); // [n_tok, half] f32
            auto cos_t = cannge_es::Cos(theta); // include/es/es_math/es_Cos.h
            auto sin_t = cannge_es::Sin(theta); // include/es/es_math/es_Sin.h

            // Full-width per-position tables follow the rotation pairing:
            // NORMAL pairs are adjacent so each value is pair-duplicated,
            // NEOX tables duplicate the half
            auto cos_tab = cos_t;
            auto sin_tab = sin_t;
            std::vector<cannge_es::EsTensorHolder> two_cos = { cos_t, cos_t };
            std::vector<cannge_es::EsTensorHolder> two_sin = { sin_t, sin_t };
            if (mode == GGML_ROPE_TYPE_NEOX) {
                cos_tab = cannge_es::Concat(ctx.scalar_cached((int32_t) 1), two_cos, 2);
                sin_tab = cannge_es::Concat(ctx.scalar_cached((int32_t) 1), two_sin, 2);
            } else {
                auto dup_pairs = [&](cannge_es::EsTensorHolder t) {
                    const std::vector<int64_t>          shape3 = { n_tok, half, 1 };
                    const std::vector<int64_t>          shape2 = { n_tok, (int64_t) n_dims };
                    auto e = cannge_es::Reshape(t, shape3);
                    std::vector<cannge_es::EsTensorHolder> two = { e, e };
                    e      = cannge_es::Concat(ctx.scalar_cached((int32_t) 2), two, 2);
                    return cannge_es::Reshape(e, shape2);
                };
                cos_tab = dup_pairs(cos_t);
                sin_tab = dup_pairs(sin_t);
            }
            // Use canonical [B,T,H,D] order for both decode and prefill.
            const int64_t batch = src0->ne[3], heads = src0->ne[1];
            const std::vector<int64_t> shape4 = { batch, n_tok, heads, n_dims };
            auto x4 = cannge_es::Reshape(es0, shape4);
            if (src0->type != GGML_TYPE_F32) {
                x4 = cannge_es::Cast(x4, (int64_t) ge::DT_FLOAT);
            }
            cannge_es::EsTensorHolder rotated;
            if (mode == GGML_ROPE_TYPE_NEOX) {
                const std::vector<int64_t> size = { batch, n_tok, heads, half };
                auto first = cannge_es::Slice(x4, std::vector<int64_t>{0, 0, 0, 0}, size);
                auto second = cannge_es::Slice(x4, std::vector<int64_t>{0, 0, 0, half}, size);
                rotated = cannge_es::Concat(ctx.scalar_cached((int32_t) 3),
                        {cannge_es::Sub(ctx.scalar_cached(0.0f), second), first}, 2);
            } else {
                auto pairs = cannge_es::Reshape(x4, std::vector<int64_t>{batch, n_tok, heads, half, 2});
                const std::vector<int64_t> size = { batch, n_tok, heads, half, 1 };
                auto even = cannge_es::Slice(pairs, std::vector<int64_t>{0, 0, 0, 0, 0}, size);
                auto odd = cannge_es::Slice(pairs, std::vector<int64_t>{0, 0, 0, 0, 1}, size);
                rotated = cannge_es::Concat(ctx.scalar_cached((int32_t) 4),
                        {cannge_es::Sub(ctx.scalar_cached(0.0f), odd), even}, 2);
                rotated = cannge_es::Reshape(rotated, shape4);
            }
            const std::vector<int64_t> table_shape = { 1, n_tok, 1, n_dims };
            auto y = cannge_es::Add(cannge_es::Mul(x4, cannge_es::Reshape(cos_tab, table_shape)),
                                    cannge_es::Mul(rotated, cannge_es::Reshape(sin_tab, table_shape)));
            if (node->type != GGML_TYPE_F32) {
                ge::DataType dtype;
                if (!ggml_cannge_ge_dtype(node->type, dtype)) {
                    return ctx.fail(std::string("unsupported rope output dtype ") + node->name);
                }
                y = cannge_es::Cast(y, (int64_t) dtype);
            }
            y = cannge_es::Reshape(y, ggml_cannge_ge_shape(node));
            ctx.tensors[node] = y;
            return true;
        }

        default:
            return ctx.fail("unsupported rope operation");
    }
}
