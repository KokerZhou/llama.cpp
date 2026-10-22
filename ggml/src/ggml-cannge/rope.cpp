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
            const int     nd        = ggml_n_dims(src0);
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

            // the GE op takes full-width per-position tables (layout 11SD):
            // NORMAL pairs are adjacent so each value is pair-duplicated,
            // NEOX tables duplicate the half
            // spec: https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.rotary_position_embedding.html
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
            if (src0->type != GGML_TYPE_F32) {
                // GE requires cos/sin in x's dtype
                ge::DataType xdt;
                if (!ggml_cannge_ge_dtype(src0->type, xdt)) {
                    return ctx.fail(std::string("unsupported rope dtype ") + node->name);
                }
                cos_tab = cannge_es::Cast(cos_tab, (int64_t) xdt);
                sin_tab = cannge_es::Cast(sin_tab, (int64_t) xdt);
            }

            // x: ggml [D, H, T, B] -> GE [B?, T, H, D]; the op wants BNSD
            // [B, H, T, D], so swap the token and head axes (prepend a dummy
            // batch for 3D inputs). Decode is 2D [D, H] with T=1: (d, h)
            // memory already matches [1, H, 1, D] row-major, plain reshape
            const std::vector<int64_t> to_bnsd =
                nd == 3 ? std::vector<int64_t>{ 1, 0, 2 } : std::vector<int64_t>{ 0, 2, 1, 3 };
            cannge_es::EsTensorHolder x4;
            if (nd == 2) {
                x4 = cannge_es::Reshape(es0, std::vector<int64_t>{ 1, src0->ne[1], 1, (int64_t) n_dims });
            } else {
                x4 = cannge_es::Permute(es0, to_bnsd);
            }
            if (nd == 3) {
                const std::vector<int64_t> shape_bnsd = { 1, src0->ne[1], n_tok, (int64_t) n_dims };
                x4 = cannge_es::Reshape(x4, shape_bnsd);
            }
            // GE mode 0 = rotate_half (NeoX), 1 = rotate_interleaved (GPT-J == ggml NORMAL)
            const std::vector<int64_t> shape_11sd = { 1, 1, n_tok, (int64_t) n_dims };
            auto y = cannge_es::RotaryPositionEmbedding(x4, cannge_es::Reshape(cos_tab, shape_11sd),
                                                        cannge_es::Reshape(sin_tab, shape_11sd), nullptr,
                                                        mode == GGML_ROPE_TYPE_NEOX ? 0 : 1);
            if (nd == 2) {
                y = cannge_es::Reshape(y, ggml_cannge_ge_shape(node));
            } else if (nd == 3) {
                const std::vector<int64_t> shape3 = { src0->ne[1], n_tok, (int64_t) n_dims };
                const std::vector<int64_t> perm3  = { 1, 0, 2 };
                y = cannge_es::Reshape(y, shape3);
                y = cannge_es::Permute(y, perm3);
            } else {
                y = cannge_es::Permute(y, std::vector<int64_t>{ 0, 2, 1, 3 });
            }
            ctx.tensors[node] = y;
            return true;
        }

        default:
            return ctx.fail("unsupported rope operation");
    }
}
