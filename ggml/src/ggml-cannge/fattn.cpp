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

// translates the per-call IO analysis into a ge::Graph via the ES authoring API
// axis order convention (docs/cannge note-7): ggml ne[0] is innermost, GE
// shapes are outermost-first, all conversions go through graph-build.h helpers

#include "fattn.h"
#include "graph-build.h"

#include "ggml-impl.h"
#include "plan.h"

#include <ge/es_graph_builder.h>

#include <es_math/es_math_ops.h>
#include <es_nn/es_nn_ops.h>
#include <es_transformer/es_transformer_ops.h>

#include <graph/graph.h>

#include <cstring>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace cannge_es = ge::es;

#include "graph-build-priv.h"


bool ggml_cannge_build_attention(ggml_cannge_build_context & ctx, ggml_tensor * node) {
    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];
    cannge_es::EsTensorHolder es0;
    cannge_es::EsTensorHolder es1;
    const struct ggml_tensor * src2 = node->src[2];
    const struct ggml_tensor * src3 = node->src[3];
    cannge_es::EsTensorHolder   es2;
    cannge_es::EsTensorHolder   es3;
    if (!ctx.get(src0, es0) || !ctx.get(src1, es1) || !ctx.get((ggml_tensor *) src2, es2)) {
        return false;
    }
    // ggml q/k/v reach FA with (d, n, s) memory order: q is llama's
    // permute(0,2,1,3) view, k/v are KV-cache slice views. With
    // input_layout=BSND a plain reshape suffices. Knobs allow the
    // BNSD convention (pre/post permute) for interop probing
    if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
        fprintf(stderr,
                "[DBG-FA] %s q ne=%lld,%lld,%lld,%lld nb=%lld,%lld,%lld,%lld t=%d | k ne=%lld,%lld,%lld "
                "nb=%lld,%lld,%lld t=%d | v ne=%lld,%lld,%lld nb=%lld,%lld,%lld t=%d\n",
                node->name, (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2],
                (long long) src0->ne[3], (long long) src0->nb[0], (long long) src0->nb[1],
                (long long) src0->nb[2], (long long) src0->nb[3], (int) src0->type, (long long) src1->ne[0],
                (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->nb[0],
                (long long) src1->nb[1], (long long) src1->nb[2], (int) src1->type, (long long) src2->ne[0],
                (long long) src2->ne[1], (long long) src2->ne[2], (long long) src2->nb[0],
                (long long) src2->nb[1], (long long) src2->nb[2], (int) src2->type);
    }
    const char * fa_layout = ctx.io.options.attention_layout.c_str();
    const bool   fa_inperm = ctx.io.options.attention_input_permute;
    const bool   fa_outperm = ctx.io.options.attention_output_permute;
    auto to_bsnd = [&](const ggml_tensor * t, cannge_es::EsTensorHolder & es) {
        const int64_t d = t->ne[0], s = t->ne[1], n = t->ne[2];
        auto r = cannge_es::Reshape(es, std::vector<int64_t>{ 1, s, n, d });
        if (fa_inperm) {
            // genuine (s,n,d) -> (n,s,d) reorder via 3D transpose;
            // a 4D {0,2,1,3} here gets folded into the upstream ggml
            // permute view by the GE optimizer and never materializes
            auto r3 = cannge_es::Reshape(r, std::vector<int64_t>{ s, n, d });
            r3        = cannge_es::Permute(r3, std::vector<int64_t>{ 1, 0, 2 });
            r         = cannge_es::Reshape(r3, std::vector<int64_t>{ 1, n, s, d });
        }
        return r;
    };
    const int64_t d = src0->ne[0];
    auto              q = to_bsnd(src0, es0);
    auto              k = to_bsnd(src1, es1);
    auto              v = to_bsnd(src2, es2);

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) node->op_params + 0, sizeof(scale));

    const int64_t n_heads    = src0->ne[2];
    const int64_t n_kv_heads = src1->ne[2];
    const int64_t s_q        = src0->ne[1];
    const int64_t s_kv       = src1->ne[1];

    // ggml kq_mask is additive [S_kv, S_q]; GE atten_mask is
    // [B, N, S_q, S_kv] broadcastable - transpose and expand
    bool has_mask = false;
    cannge_es::EsTensorHolder mask;
    if (src3 != nullptr) {
        if (!ctx.get((ggml_tensor *) src3, es3)) {
            return false;
        }
        // ggml mask [S_kv, S_q]; a single-query mask ([KV,1]) is 1-D
        // in ggml (trailing dim dropped) - GE still wants [1,1,Q,KV]
        if (s_q == 1) {
            mask = cannge_es::Reshape(es3, std::vector<int64_t>{ 1, 1, 1, s_kv });
        } else {
            auto mt = cannge_es::Permute(es3, std::vector<int64_t>{ 1, 0 }); // [S_q, S_kv]
            mask    = cannge_es::Reshape(mt, std::vector<int64_t>{ 1, 1, s_q, s_kv });
        }
        // GE IFA/PFA atten_mask is BOOL (keep = true; DT_FLOAT is
        // rejected, EZ3002). ggml masks are additive 0/-inf: keep is
        // "greater than any real masked value"
        auto mf32 = cannge_es::Cast(mask, (int64_t) ge::DT_FLOAT);
        mask      = cannge_es::Greater(mf32, ctx.builder.CreateScalar(-1.0e30f));
        has_mask  = true;
    }

    cannge_es::EsTensorHolder y;
    const int64_t fa_pre  = ctx.io.options.attention_pre_tokens;
    const int64_t fa_next = ctx.io.options.attention_next_tokens;
    const bool fa_ifa_all = ctx.io.options.attention_ifa_all;
    const int64_t fa_asl  = ctx.io.options.attention_sequence_length;
    if (s_q == 1 || fa_ifa_all) {
        // decode: incremental FA over the padded KV cache. IFA is the
        // streaming op: kv validity is carried by actual_seq_lengths
        // in serving stacks, not by an atten_mask (probe fa m12: the
        // BOOL mask is ignored for s_q=1; ASL probe, note-42)
        std::vector<cannge_es::EsTensorHolder> ks, vs;
        ks.push_back(k);
        vs.push_back(v);
        if (ctx.has_fa_asl) {
            // kv validity via actual_seq_lengths (refreshed per
            // execute from the mask prefix length); the BOOL mask
            // itself is ignored by IFA for s_q=1 (probe fa m12)
            // tensor slots: pse, mask, asl, deq1, q1, deq2, q2, qoff2, antiq_s, antiq_o, block_table, kv_pad
            y = cannge_es::IncreFlashAttention(q, ks, vs, nullptr, nullptr, ctx.fa_asl_holder, nullptr,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                               nullptr, n_heads, scale, fa_layout, n_kv_heads, 0, 1);
        } else if (fa_asl > 0) {
            auto asl = ctx.builder.CreateConst(std::vector<int64_t>{ fa_asl },
                                               std::vector<int64_t>{ 1 });
            y = cannge_es::IncreFlashAttention(q, ks, vs, nullptr, nullptr, asl, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, n_heads,
                                               scale, fa_layout, n_kv_heads, 0, 1);
        } else if (has_mask) {
            y = cannge_es::IncreFlashAttention(q, ks, vs, nullptr, mask, nullptr, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, n_heads,
                                               scale, fa_layout, n_kv_heads, 0, 1);
        } else {
            y = cannge_es::IncreFlashAttention(q, ks, vs, nullptr, nullptr, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                               n_heads, scale, fa_layout, n_kv_heads, 0, 1);
        }
    } else {
        if (has_mask) {
            y = cannge_es::PromptFlashAttention(q, k, v, nullptr, mask, nullptr, nullptr, nullptr, nullptr,
                                                nullptr, nullptr, nullptr, n_heads, scale, fa_pre,
                                                fa_next, fa_layout, n_kv_heads, 0, 1);
        } else {
            y = cannge_es::PromptFlashAttention(q, k, v, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                nullptr, nullptr, nullptr, nullptr, n_heads, scale,
                                                fa_pre, fa_next, fa_layout, n_kv_heads, 0, 1);
        }
    }
    // PFA/IFA output: with BSND it is [1, S, N, D] whose (s, n, d)
    // memory order matches ggml's (d, h, s) layout of the result
    // exactly - a plain reshape, no permute. With BNSD inputs the
    // output is [1, N, S, D] and needs the swap back
    if (fa_outperm) {
        // y is [1, N, S, D] (n,s,d) under BNSD: genuine reorder back
        auto y3 = cannge_es::Reshape(y, std::vector<int64_t>{ n_heads, s_q, d });
        y3      = cannge_es::Permute(y3, std::vector<int64_t>{ 1, 0, 2 });
        y       = cannge_es::Reshape(y3, std::vector<int64_t>{ 1, s_q, n_heads, d });
    }
    y = cannge_es::Reshape(y, ggml_cannge_ge_shape(node));
    if (node->type != src0->type) {
        // ggml FA output follows the precision flag (F32), GE keeps
        // the q dtype - cast to the declared output dtype
        ge::DataType dt;
        if (!ggml_cannge_ge_dtype(node->type, dt)) {
            return ctx.fail(std::string("unsupported fa output dtype ") + node->name);
        }
        y = cannge_es::Cast(y, (int64_t) dt);
    }
    ctx.tensors[node] = y;
    return true;
}
