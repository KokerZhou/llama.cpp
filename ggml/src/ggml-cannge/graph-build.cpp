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

struct ggml_cannge_build_context {
    ggml_cannge_plan_io &     io;
    cannge_es::EsGraphBuilder builder;
    std::map<ggml_tensor *, cannge_es::EsTensorHolder> tensors;
    std::string & err;

    ggml_cannge_build_context(ggml_cannge_plan_io & io, std::string & err) : io(io), builder("ggml_cannge"), err(err) {
    }

    bool fail(const std::string & msg) {
        err = msg;
        return false;
    }

    bool get(ggml_tensor * t, cannge_es::EsTensorHolder & holder) {
        auto it = tensors.find(t);
        if (it == tensors.end()) {
            if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
                fprintf(stderr,
                        "[DBG-BUILD] no ES tensor for %s op=%s type=%d ne=%lld,%lld,%lld,%lld nb=%lld,%lld,%lld,%lld\n",
                        t->name, ggml_op_name(t->op), (int) t->type, (long long) t->ne[0], (long long) t->ne[1],
                        (long long) t->ne[2], (long long) t->ne[3], (long long) t->nb[0], (long long) t->nb[1],
                        (long long) t->nb[2], (long long) t->nb[3]);
            }
            return fail("no ES tensor for " + std::string(t->name));
        }
        holder = it->second;
        return true;
    }
};

// external input tensors become GE graph inputs, in io.inputs order so the
// execute-time binding indices match CreateInput
static bool ggml_cannge_build_inputs(ggml_cannge_build_context & ctx) {
    for (size_t i = 0; i < ctx.io.inputs.size(); i++) {
        ggml_tensor * t = ctx.io.inputs[i];
        ge::DataType  dt;
        if (!ggml_cannge_ge_dtype(t->type, dt)) {
            return ctx.fail(std::string("unsupported dtype of input ") + t->name);
        }
        ctx.tensors[t] = ctx.builder.CreateInput((int64_t) i, t->name, dt, ge::FORMAT_ND, ggml_cannge_ge_shape(t));
    }
    return true;
}

// ggml GGML_OP_VIEW: alias when offset, strides and the full shape match the
// base (e.g. the first QKV chunk has offset 0 but a smaller ne and needs the
// Slice path); anything non-dense fails the build. The chain walk stops at the
// highest tensor that has an ES holder: in split graphs the scheduler keeps
// the boundary tensor (itself a view) as the graph input, so walking past it
// to a CPU-side root would find no holder at all
static bool ggml_cannge_build_view(ggml_cannge_build_context & ctx, ggml_tensor * node) {
    // holder lookup: exact pointer first, then the registered tensor whose
    // device buffer contains the view's data pointer (the scheduler rebases
    // view chains onto the boundary copies, so the chain object may differ
    // from the registered input even though the buffer is the right one)
    auto has_holder = [&ctx](ggml_tensor * t) -> ggml_tensor * {
        if (t == nullptr) {
            return nullptr;
        }
        auto it = ctx.tensors.find(t);
        if (it != ctx.tensors.end()) {
            return it->first;
        }
        return nullptr;
    };

    ggml_tensor * base = node;
    size_t        offset = 0;
    std::set<ggml_tensor *> visited; // guard against chain cycles
    visited.insert(base);
    while (ggml_is_view(base) && base->view_src != nullptr) {
        ggml_tensor * nxt = has_holder(base->view_src);
        if (nxt == nullptr || visited.count(nxt)) {
            break;
        }
        offset += base->view_offs;
        base   = nxt;
        visited.insert(base);
    }
    if (!ctx.tensors.count(base) && node->data != nullptr) {
        // chain walk dead-ends on an unregistered object: resolve the base in
        // two tiers. Tier 1: registered tensors named like the chain parent
        // (view_src); Tier 2: any registered tensor containing node->data.
        // Same-buffer shapes (per-head vs merged views) share the span, so
        // span alone cannot disambiguate: a candidate only qualifies if the
        // view box actually fits its dims at the data-pointer offset
        const int64_t esize = ggml_element_size(node);
        const int   node_nd = ggml_n_dims(node);
        auto fits = [&](const ggml_tensor * cand) {
            const char * lo = (const char *) cand->data;
            size_t       span = ggml_nbytes(cand);
            if ((const char *) node->data < lo || (const char *) node->data >= lo + span) {
                return false;
            }
            // full-buffer reinterpretation: a dense view covering the whole
            // candidate buffer under a different shape (per-head vs merged)
            const size_t node_span = (size_t) node->ne[0] * node->ne[1] * node->ne[2] * node->ne[3] * esize;
            if ((const char *) node->data == lo && node_span == span) {
                return true;
            }
            int64_t rem = (int64_t) (((const char *) node->data - lo) / esize);
            if (rem * esize != (int64_t) ((const char *) node->data - lo)) {
                return false;
            }
            const int cand_nd = ggml_n_dims(cand);
            for (int d = 0; d < cand_nd; d++) {
                const int64_t cand_ne = cand->ne[d];
                const int64_t off = cand_ne > 0 ? rem % cand_ne : 0;
                rem               = cand_ne > 0 ? rem / cand_ne : 0;
                if (d < node_nd && off + node->ne[d] > cand_ne) {
                    return false;
                }
            }
            for (int d = cand_nd; d < node_nd; d++) {
                // a view cannot extend a base dim with ne > 1 beyond the base
                if (node->ne[d] != 1) {
                    return false;
                }
            }
            return rem == 0;
        };
        // rank candidates: a same-rank base needs only a Slice, a rank-mismatch
        // needs Slice+Reshape (risky for GE infer); span breaks ties
        auto rank_of = [](const ggml_tensor * t) { return ggml_n_dims(t); };
        const ggml_tensor * best = nullptr;
        size_t              best_span = SIZE_MAX;
        int                 best_rank_pen = 2;
        if (node->view_src != nullptr) {
            for (const auto & kv : ctx.tensors) {
                if (strcmp(kv.first->name, node->view_src->name) != 0 || !fits(kv.first)) {
                    continue;
                }
                const size_t span = ggml_nbytes(kv.first);
                const int    pen  = rank_of(kv.first) == node_nd ? 0 : 1;
                if (pen < best_rank_pen || (pen == best_rank_pen && span < best_span)) {
                    best          = kv.first;
                    best_span     = span;
                    best_rank_pen = pen;
                }
            }
        }
        if (best == nullptr) {
            best_rank_pen = 2;
            best_span     = SIZE_MAX;
            for (const auto & kv : ctx.tensors) {
                if (!fits(kv.first)) {
                    continue;
                }
                const size_t span = ggml_nbytes(kv.first);
                const int    pen  = rank_of(kv.first) == node_nd ? 0 : 1;
                if (pen < best_rank_pen || (pen == best_rank_pen && span < best_span)) {
                    best          = kv.first;
                    best_span     = span;
                    best_rank_pen = pen;
                }
            }
        }
        if (getenv("GGML_CANNGE_DEBUG_COMPILE") && best != nullptr) {
            fprintf(stderr, "[DBG-COMPILE]   view %s: dead-end base=%s span=%zu\n", node->name, best->name, best_span);
        }
        if (getenv("GGML_CANNGE_DEBUG_COMPILE") && best == nullptr) {
            fprintf(stderr, "[DBG-COMPILE]   view %s: dead-end NO CANDIDATE node_data=%p ne=%lld,%lld\n", node->name,
                    node->data, (long long) node->ne[0], (long long) node->ne[1]);
            for (const auto & kv : ctx.tensors) {
                fprintf(stderr, "[DBG-COMPILE]     cand %s data=%p span=%zu fits=%d\n", kv.first->name, kv.first->data,
                        ggml_nbytes(kv.first), (int) fits(kv.first));
            }
        }
        if (best != nullptr) {
            base   = (ggml_tensor *) best;
            offset = (size_t) ((const char *) node->data - (const char *) best->data);
        }
    }

    if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
        fprintf(stderr, "[DBG-COMPILE]   view %s: walk done base=%s offset=%zu\n", node->name, base->name, offset);
    }
    cannge_es::EsTensorHolder base_holder;
    if (!ctx.get(base, base_holder)) {
        return false;
    }
    if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
        fprintf(stderr, "[DBG-COMPILE]   view %s: base holder ok\n", node->name);
    }

    bool same_shape = true;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != base->ne[i]) {
            same_shape = false;
            break;
        }
    }

    const bool dense_rel = node->nb[0] == ggml_element_size(node);
    bool     dense = dense_rel;
    for (int i = 1; i < GGML_MAX_DIMS && dense; i++) {
        dense = node->nb[i] == node->nb[i - 1] * (size_t) node->ne[i - 1];
    }

    if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
        fprintf(stderr, "[DBG-COMPILE]   view %s: shape same=%d dense=%d dense_rel=%d base_nd=%d node_nd=%d\n",
                node->name, (int) same_shape, (int) dense, (int) dense_rel, ggml_n_dims(base), ggml_n_dims(node));
    }
    if (offset == 0 && dense && same_shape) {
        ctx.tensors[node] = base_holder;
        return true;
    }
    if (offset == 0 && dense && !same_shape &&
        (size_t) node->ne[0] * node->ne[1] * node->ne[2] * node->ne[3] * ggml_element_size(node) ==
            ggml_nbytes(base)) {
        // full-buffer reinterpretation (per-head vs merged layout of one buffer)
        if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
            fprintf(stderr, "[DBG-COMPILE]   view %s: reinterpret reshape base=%s ne=%lld,%lld,%lld,%lld\n", node->name,
                    base->name, (long long) base->ne[0], (long long) base->ne[1], (long long) base->ne[2],
                    (long long) base->ne[3]);
        }
        ctx.tensors[node] = cannge_es::Reshape(base_holder, ggml_cannge_ge_shape(node));
        return true;
    }
    if (!dense_rel) {
        return ctx.fail(std::string("non-dense view ") + node->name);
    }

    // decompose the view's byte range [data, data + dense footprint) into a
    // base-coordinate box: start/end offsets decomposed along the base dims,
    // the box is valid iff its element count matches the view (a contiguous
    // range is a box only when it is slab-aligned; full-row cache slots are)
    const int64_t elsize = ggml_element_size(node);
    int64_t       rem = (int64_t) (offset / elsize);
    if (rem * elsize != (int64_t) offset) {
        return ctx.fail(std::string("misaligned view offset ") + node->name);
    }
    const int64_t node_nelems = node->ne[0] * node->ne[1] * node->ne[2] * node->ne[3];

    const int base_nd = ggml_n_dims(base);
    const int node_nd = ggml_n_dims(node);

    std::vector<int64_t> offsets_gg(base_nd, 0);
    std::vector<int64_t> sizes_gg(base_nd, 1);
    {
        int64_t       r = rem;
        int64_t       end = rem + node_nelems - 1;
        bool          aligned = true;
        int64_t       box_nelems = 1;
        for (int d = 0; d < base_nd; d++) {
            const int64_t base_ne = base->ne[d];
            if (base_ne <= 0) {
                aligned = false;
                break;
            }
            const int64_t off_s = r % base_ne;
            const int64_t off_e = end % base_ne;
            r /= base_ne;
            end /= base_ne;
            offsets_gg[d] = off_s;
            sizes_gg[d]   = off_e - off_s + 1;
            box_nelems *= sizes_gg[d];
        }
        if (!aligned || r != 0 || end != 0 || box_nelems != node_nelems) {
            return ctx.fail(std::string("view is not a base-dim box ") + node->name);
        }
    }

    // convert both to GE axis order (outermost-first)
    std::vector<int64_t> offsets(offsets_gg.rbegin(), offsets_gg.rend());
    std::vector<int64_t> sizes(sizes_gg.rbegin(), sizes_gg.rend());

    if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
        fprintf(stderr,
                "[DBG-COMPILE]   view %s: slice path same=%d dense=%d nd=%d/%d off=%zu base_ne=%lld,%lld,%lld,%lld "
                "node_ne=%lld,%lld,%lld,%lld base_span=%zu node_span=%zu\n",
                node->name, (int) same_shape, (int) dense, node_nd, base_nd, offset, (long long) base->ne[0],
                (long long) base->ne[1], (long long) base->ne[2], (long long) base->ne[3], (long long) node->ne[0],
                (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3], ggml_nbytes(base),
                (size_t) node->ne[0] * node->ne[1] * node->ne[2] * node->ne[3] * ggml_element_size(node));
    }
    cannge_es::EsTensorHolder sliced = cannge_es::Slice(base_holder, offsets, sizes);
    if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
        fprintf(stderr, "[DBG-COMPILE]   view %s: slice returned\n", node->name);
    }
    if (node_nd != base_nd) {
        sliced = cannge_es::Reshape(sliced, ggml_cannge_ge_shape(node));
    }
    ctx.tensors[node] = sliced;
    return true;
}

static bool ggml_cannge_build_node(ggml_cannge_build_context & ctx, ggml_tensor * node) {
    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];
    ggml_tensor * src2 = node->src[2];

    cannge_es::EsTensorHolder es0;
    cannge_es::EsTensorHolder es1;

    auto binary = [&](cannge_es::EsTensorHolder (*op)(const cannge_es::EsTensorLike &,
                                                       const cannge_es::EsTensorLike &)) {
        if (!ctx.get(src0, es0) || !ctx.get(src1, es1)) {
            return false;
        }
        ctx.tensors[node] = op(es0, es1);
        return true;
    };

    switch (node->op) {
        case GGML_OP_ADD:
            return binary(cannge_es::Add);
        case GGML_OP_SUB:
            return binary(cannge_es::Sub);
        case GGML_OP_MUL:
            return binary(cannge_es::Mul);
        case GGML_OP_DIV:
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
            auto es = scale == 1.0f ? es0 : cannge_es::Mul(es0, ctx.builder.CreateScalar(scale));
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
            auto ms = cannge_es::ReduceMean(cannge_es::Square(es0), ctx.builder.CreateVector(axes), true);
            auto den = cannge_es::Rsqrt(cannge_es::Add(ms, ctx.builder.CreateScalar(eps)));
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
            auto gamma = ctx.builder.CreateConst(ones, { n });
            auto beta  = ctx.builder.CreateConst(zeros, { n });
            auto ln = cannge_es::LayerNorm(es0, gamma, beta, (int64_t) ggml_n_dims(src0) - 1,
                                           (int64_t) ggml_n_dims(src0) - 1, eps); // include/es/es_nn/es_LayerNorm.h
            ctx.tensors[node] = ln.y;
            return true;
        }

        case GGML_OP_MUL_MAT: {
            if (!ctx.get(src0, es0) || !ctx.get(src1, es1)) {
                return false;
            }
            // ggml: dst[n, m] = sum_k src0[k, n] * src1[k, m], ggml ne: src0 [K, N], src1 [K, M], dst [N, M]
            // GE row-major: x1 = src1 as [M, K], x2 = src0 as [N, K] with adj_x2 -> [K, N]
            // y = x1 @ x2^T = [M, N] = reverse(dst ne), matches both references
            // BatchMatMulV3 (JittorInfer mapping): MatMulV3 has no registered
            // infer_datatype func on this toolkit and fails to compile
            // ggml mul_mat always outputs F32, even for f16 inputs (ggml.c
            // ggml_mul_mat); GE keeps the input dtype, so an f16 result needs
            // a cast to match the F32 writeback buffer (F13 half-written out)
            auto mm = cannge_es::BatchMatMulV3(es1, es0, nullptr, nullptr, false, true);
            if (node->type != src0->type) {
                ge::DataType dt;
                if (!ggml_cannge_ge_dtype(node->type, dt)) {
                    return ctx.fail(std::string("unsupported mul_mat output dtype ") + node->name);
                }
                mm = cannge_es::Cast(mm, (int64_t) dt);
            }
            ctx.tensors[node] = mm;
            return true;
        }

        case GGML_OP_RESHAPE: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            ctx.tensors[node] = cannge_es::Reshape(es0, ggml_cannge_ge_shape(node));
            return true;
        }

        case GGML_OP_PERMUTE: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            // ggml op_params is a forward src->dst map (out.ne[p[i]] sits at src
            // dim i), ES Permute order is dst->src, so invert first; both use
            // ggml axis numbering, then flip to GE order
            const int nd = ggml_n_dims(node);
            int32_t   inv_p[GGML_MAX_DIMS] = { 0, 1, 2, 3 };
            for (int i = 0; i < nd; i++) {
                inv_p[node->op_params[i]] = i;
            }
            std::vector<int64_t> order(nd);
            for (int k = 0; k < nd; k++) {
                order[k] = nd - 1 - inv_p[nd - 1 - k];
            }
            ctx.tensors[node] = cannge_es::Permute(es0, order);
            return true;
        }

        case GGML_OP_TRANSPOSE: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            // swap the innermost two ggml dims = outermost two GE dims
            const int nd = ggml_n_dims(node);
            std::vector<int64_t> perm(nd);
            for (int i = 0; i < nd; i++) {
                perm[i] = i;
            }
            perm[nd - 1] = nd - 2;
            perm[nd - 2] = nd - 1;
            ctx.tensors[node] = cannge_es::Transpose(es0, perm);
            return true;
        }

        case GGML_OP_VIEW:
            return ggml_cannge_build_view(ctx, node);

        case GGML_OP_CONT: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            if (!ggml_is_contiguous(src0)) {
                return ctx.fail(std::string("non-contiguous CONT input ") + src0->name);
            }
            ctx.tensors[node] = es0; // alias, no data movement needed
            return true;
        }

        case GGML_OP_CPY: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            if (src0->type == node->type) {
                // same dtype: alias; a separate destination buffer gets its own
                // GE output port (registered by analyze in io.output_tensors)
                ctx.tensors[node] = es0;
                return true;
            }
            ge::DataType dt;
            if (!ggml_cannge_ge_dtype(node->type, dt)) {
                return ctx.fail(std::string("unsupported CPY dtype ") + node->name);
            }
            ctx.tensors[node] = cannge_es::Cast(es0, (int64_t) dt);
            return true;
        }

        case GGML_OP_DUP: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            // same dtype by definition: alias like same-dtype CPY; analyze
            // forces a GE output port for the node so the destination buffer
            // receives the copy (plan.cpp output_tensors)
            ctx.tensors[node] = es0;
            return true;
        }

        case GGML_OP_GET_ROWS: {
            if (!ctx.get(src0, es0) || !ctx.get(src1, es1)) {
                return false;
            }
            // ggml gathers rows along ggml dim 1 = GE axis (ndims - 2); 2D inputs give axis 0
            auto gathered = cannge_es::GatherV2(es0, es1, (int64_t) (ggml_n_dims(src0) - 2));
            // ggml_get_rows always outputs F32 (ggml.c "TODO: implement non F32 return"),
            // GE keeps the input dtype: cast an f16 result to match the writeback buffer
            if (node->type != src0->type) {
                ge::DataType dt;
                if (!ggml_cannge_ge_dtype(node->type, dt)) {
                    return ctx.fail(std::string("unsupported get_rows output dtype ") + node->name);
                }
                gathered = cannge_es::Cast(gathered, (int64_t) dt);
            }
            ctx.tensors[node] = gathered;
            return true;
        }

        case GGML_OP_CONCAT: {
            if (!ctx.get(src0, es0) || !ctx.get(src1, es1)) {
                return false;
            }
            const int nd = ggml_n_dims(node);
            const int ge_axis = nd - 1 - (int) node->op_params[0];
            ctx.tensors[node] =
                cannge_es::Concat(ctx.builder.CreateScalar((int32_t) ge_axis), { es0, es1 }, 2);
            return true;
        }

        case GGML_OP_SET_ROWS: {
            // the node is a view of the cache (src[2]); the write itself is
            // applied by the backend after execute, see io.set_rows
            cannge_es::EsTensorHolder es2;
            if (!ctx.get(src2, es2)) {
                return false;
            }
            ctx.tensors[node] = es2;
            return true;
        }

        case GGML_OP_GLU: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            const enum ggml_glu_op glu_op = ggml_get_glu_op(node);
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
            auto theta = cannge_es::Mul(pos_f, ctx.builder.CreateConst(freqs, shape_fr)); // [n_tok, half] f32
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
                cos_tab = cannge_es::Concat(ctx.builder.CreateScalar((int32_t) 1), two_cos, 2);
                sin_tab = cannge_es::Concat(ctx.builder.CreateScalar((int32_t) 1), two_sin, 2);
            } else {
                auto dup_pairs = [&](cannge_es::EsTensorHolder t) {
                    const std::vector<int64_t>          shape3 = { n_tok, half, 1 };
                    const std::vector<int64_t>          shape2 = { n_tok, (int64_t) n_dims };
                    std::vector<cannge_es::EsTensorHolder> two = { t, t };
                    auto e = cannge_es::Reshape(t, shape3);
                    e      = cannge_es::Concat(ctx.builder.CreateScalar((int32_t) 2), two, 2);
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
            // batch for 3D inputs)
            const std::vector<int64_t> to_bnsd =
                nd == 3 ? std::vector<int64_t>{ 1, 0, 2 } : std::vector<int64_t>{ 0, 2, 1, 3 };
            auto x4 = cannge_es::Permute(es0, to_bnsd);
            if (nd == 3) {
                const std::vector<int64_t> shape_bnsd = { 1, src0->ne[1], n_tok, (int64_t) n_dims };
                x4 = cannge_es::Reshape(x4, shape_bnsd);
            }
            // GE mode 0 = rotate_half (NeoX), 1 = rotate_interleaved (GPT-J == ggml NORMAL)
            const std::vector<int64_t> shape_11sd = { 1, 1, n_tok, (int64_t) n_dims };
            auto y = cannge_es::RotaryPositionEmbedding(x4, cannge_es::Reshape(cos_tab, shape_11sd),
                                                        cannge_es::Reshape(sin_tab, shape_11sd), nullptr,
                                                        mode == GGML_ROPE_TYPE_NEOX ? 0 : 1);
            if (nd == 3) {
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
            return ctx.fail("unsupported op " + std::to_string((int) node->op));
    }
}

bool ggml_cannge_graph_build(ggml_cannge_plan & plan, ggml_cannge_plan_io & io, ggml_cgraph * cgraph, void *& graph,
                             std::string & err) {
    ggml_cannge_build_context ctx(io, err);

    if (io.needs_staging) {
        return ctx.fail("non-dense IO requires staging, not implemented yet");
    }

    try {
        if (!ggml_cannge_build_inputs(ctx)) {
            return false;
        }

        // cgraph nodes are topologically ordered; views/reshapes register
        // aliases or slices for their consumers
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (ggml_is_empty(node) || node->op == GGML_OP_NONE) {
                continue;
            }
            if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                fprintf(stderr, "[DBG-COMPILE]   build node[%d] %s op=%s ne=%lld,%lld,%lld,%lld\n", i, node->name,
                        ggml_op_name(node->op), (long long) node->ne[0], (long long) node->ne[1],
                        (long long) node->ne[2], (long long) node->ne[3]);
            }
            if (!ggml_cannge_build_node(ctx, node)) {
                if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
                    fprintf(stderr, "[DBG-BUILD] node %s op=%s failed: %s\n", node->name, ggml_op_name(node->op),
                            err.c_str());
                }
                return false;
            }
        }

        // output ports, in the exact order analyze registered in
        // io.output_tensors (forced CPY destinations first, deduplicated)
        for (size_t i = 0; i < io.output_tensors.size(); i++) {
            cannge_es::EsTensorHolder holder;
            if (!ctx.get(io.output_tensors[i], holder)) {
                return false;
            }
            cannge_es::EsGraphBuilder::SetOutput(holder, (int64_t) i);
        }

        graph = ctx.builder.BuildAndReset().release();
    } catch (const std::exception & e) {
        return ctx.fail(std::string("ES build threw: ") + e.what());
    }

    if (graph == nullptr) {
        return ctx.fail("BuildAndReset returned null graph");
    }

    plan.mem_estimate = io.mem_estimate;
    return true;
}
