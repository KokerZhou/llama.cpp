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

#include <graph/graph.h>

#include <cstring>
#include <map>
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

// ggml GGML_OP_VIEW: pure alias only when offset, strides and the full shape
// match the base (e.g. the first QKV chunk has offset 0 but a smaller ne and
// needs the Slice path); anything non-dense fails the build
static bool ggml_cannge_build_view(ggml_cannge_build_context & ctx, ggml_tensor * node) {
    ggml_cannge_view_info info;
    if (!ggml_cannge_resolve_view(node, info)) {
        return ctx.fail("broken view chain");
    }

    cannge_es::EsTensorHolder base;
    if (!ctx.get(info.base, base)) {
        return false;
    }

    bool same_shape = true;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != info.base->ne[i]) {
            same_shape = false;
            break;
        }
    }

    if (info.offset == 0 && info.is_dense && same_shape) {
        ctx.tensors[node] = base;
        return true;
    }
    if (!info.is_dense) {
        return ctx.fail(std::string("non-dense view ") + node->name);
    }

    // decompose the element offset along the base dims in ggml order; the
    // slice box must not wrap in any dim, else a single Slice cannot express it
    const int64_t elsize = ggml_element_size(node);
    int64_t       rem = (int64_t) (info.offset / elsize);
    if (rem * elsize != (int64_t) info.offset) {
        return ctx.fail(std::string("misaligned view offset ") + node->name);
    }

    const int base_nd = ggml_n_dims(info.base);
    const int node_nd = ggml_n_dims(node);

    std::vector<int64_t> offsets_gg(base_nd, 0);
    std::vector<int64_t> sizes_gg(base_nd, 1);
    for (int d = 0; d < base_nd; d++) {
        const int64_t base_ne = info.base->ne[d];
        const int64_t off = base_ne > 0 ? rem % base_ne : 0;
        rem = base_ne > 0 ? rem / base_ne : 0;
        offsets_gg[d] = off;
        if (d < node_nd) {
            if (off + node->ne[d] > base_ne) {
                return ctx.fail(std::string("view wraps a base dim ") + node->name);
            }
            sizes_gg[d] = node->ne[d];
        }
    }
    if (rem != 0) {
        return ctx.fail(std::string("view offset out of base range ") + node->name);
    }

    // convert both to GE axis order (outermost-first)
    std::vector<int64_t> offsets(offsets_gg.rbegin(), offsets_gg.rend());
    std::vector<int64_t> sizes(sizes_gg.rbegin(), sizes_gg.rend());

    cannge_es::EsTensorHolder sliced = cannge_es::Slice(base, offsets, sizes);
    if (node_nd != base_nd) {
        sliced = cannge_es::Reshape(sliced, ggml_cannge_ge_shape(node));
    }
    ctx.tensors[node] = sliced;
    return true;
}

static bool ggml_cannge_build_node(ggml_cannge_build_context & ctx, ggml_tensor * node) {
    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];

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
            std::memcpy(&s, node->op_params, sizeof(s));
            ctx.tensors[node] = cannge_es::Mul(es0, ctx.builder.CreateScalar(s));
            return true;
        }

        case GGML_OP_SOFT_MAX: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            // softmax over the innermost ggml dim = last GE axis
            ctx.tensors[node] = cannge_es::SoftmaxV2(es0, { -1 });
            return true;
        }

        case GGML_OP_UNARY:
            if (ggml_get_unary_op(node) != GGML_UNARY_OP_SILU) {
                return ctx.fail("unsupported unary op");
            }
            if (!ctx.get(src0, es0)) {
                return false;
            }
            ctx.tensors[node] = cannge_es::Swish(es0, 1.0f); // silu == swish(scale = 1)
            return true;

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

        case GGML_OP_MUL_MAT: {
            if (!ctx.get(src0, es0) || !ctx.get(src1, es1)) {
                return false;
            }
            // ggml: dst[n, m] = sum_k src0[k, n] * src1[k, m], ggml ne: src0 [K, N], src1 [K, M], dst [N, M]
            // GE row-major: x1 = src1 as [M, K], x2 = src0 as [N, K] with adj_x2 -> [K, N]
            // y = x1 @ x2^T = [M, N] = reverse(dst ne), matches both references
            // BatchMatMulV3 (JittorInfer mapping): MatMulV3 has no registered
            // infer_datatype func on this toolkit and fails to compile
            ctx.tensors[node] = cannge_es::BatchMatMulV3(es1, es0, nullptr, nullptr, false, true);
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

        case GGML_OP_GET_ROWS: {
            if (!ctx.get(src0, es0) || !ctx.get(src1, es1)) {
                return false;
            }
            // ggml gathers rows along ggml dim 1 = GE axis (ndims - 2); 2D inputs give axis 0
            ctx.tensors[node] = cannge_es::GatherV2(es0, es1, (int64_t) (ggml_n_dims(src0) - 2));
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
            if (!ggml_cannge_build_node(ctx, node)) {
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
