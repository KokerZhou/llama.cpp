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

#include "tensor.h"
#include "graph-build-priv.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

bool ggml_cannge_build_tensor(ggml_cannge_build_context & ctx, ggml_tensor * node) {
    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];
    ggml_tensor * src2 = node->src[2];

    cannge_es::EsTensorHolder es0;
    cannge_es::EsTensorHolder es1;

    switch (node->op) {
        case GGML_OP_RESHAPE: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            ctx.tensors[node] = cannge_es::Reshape(es0, ggml_cannge_ge_shape(node));
            return true;
        }

        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE: {
            if (!ctx.get(src0, es0) || !ggml_cannge_build_relabel(node, es0, ctx.err)) {
                return false;
            }
            ctx.tensors[node] = es0;
            return true;
        }

        case GGML_OP_VIEW:
            return ggml_cannge_build_view(ctx, node);

        case GGML_OP_CONT: {
            if (!ctx.get(src0, es0)) {
                return false;
            }
            if (!ggml_is_contiguous(src0) && !ctx.io.options.allow_noncontiguous_cont()) {
                return ctx.fail(std::string("non-contiguous CONT is disabled: ") + node->name);
            }
            if (ggml_nelements(src0) != ggml_nelements(node)) {
                return ctx.fail(std::string("element-changing CONT ") + node->name);
            }
            bool same_ne = true;
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (node->ne[i] != src0->ne[i]) {
                    same_ne = false;
                    break;
                }
            }
            // a rearrange-view source (the KV transpose): relabel chain +
            // fresh-row splice instead of a staged materialization
            if (!ggml_is_contiguous(src0) && src0->view_src != nullptr) {
                cannge_es::EsTensorHolder chained;
                const auto result = ggml_cannge_build_cont_chain(ctx, node, chained);
                if (result == ggml_cannge_build_result::failure) {
                    return false;
                }
                if (result == ggml_cannge_build_result::success) {
                    ctx.tensors[node] = chained;
                    return true;
                }
            }
            ctx.tensors[node] = same_ne ? es0 : cannge_es::Reshape(es0, ggml_cannge_ge_shape(node));
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
            // Gather retains singleton index axes; consumers use ggml's trimmed rank.
            ctx.tensors[node] = cannge_es::Reshape(gathered, ggml_cannge_ge_shape(node));
            return true;
        }

        case GGML_OP_CONCAT: {
            if (!ctx.get(src0, es0) || !ctx.get(src1, es1)) {
                return false;
            }
            const int nd = ggml_n_dims(node);
            const int ge_axis = nd - 1 - (int) node->op_params[0];
            ctx.tensors[node] =
                cannge_es::Concat(ctx.scalar_cached((int32_t) ge_axis), { es0, es1 }, 2);
            return true;
        }

        case GGML_OP_SET_ROWS: {
            // in-graph row write (note-48 batch G): GE ScatterUpdate silently
            // no-ops on this toolkit (probe-verified), so select the row with
            // a one-hot mask - all elementwise ops with broadcast, no aliasing.
            // The node keeps its boundary output port, so the fresh cache
            // value is copied back into the cache buffer post-execute (dense
            // async D2D, ordered on the stream) and stays visible across
            // fragments. Downstream views resolve their base through
            // ctx.tensors: swap the base registration to the select output so
            // GE orders in-fragment reads after the write.
            bool in_graph = false;
            for (const auto & sr : ctx.io.set_rows) {
                if (sr.dst == src2 && sr.data == src0 && sr.idx == src1) {
                    in_graph = sr.in_graph;
                    break;
                }
            }
            cannge_es::EsTensorHolder es2;
            if (!ctx.get(src2, es2)) {
                return false;
            }
            if (in_graph) {
                // batch P cache refresh: write the fresh cache in-graph as a
                // one-hot row select over the cache input, and register the
                // result under the cache base so every in-fragment view chain
                // reads the post-write bytes (GE orders via the dataflow).
                // M==1: single select; M>1: m^T @ rows as one BatchMatMulV3.
                ge::DataType dt;
                if (!ggml_cannge_ge_dtype(src2->type, dt)) {
                    return ctx.fail(std::string("unsupported set_rows dtype ") + node->name);
                }
                cannge_es::EsTensorHolder upd;
                cannge_es::EsTensorHolder idxh;
                if (!ctx.get(src0, upd) || !ctx.get(src1, idxh)) {
                    return false;
                }
                ge::DataType f32d;
                if (!ggml_cannge_ge_dtype(GGML_TYPE_F32, f32d)) {
                    return ctx.fail("no GE f32 for cache refresh");
                }
                // batch T: GE TensorScatterUpdate re-verified clean with
                // trustworthy probe reads (the original "no-op" verdict was a
                // probe artifact; note-52). Replaces the one-hot select and
                // the matrix form: var = the cache holder, one row per index,
                // updates cast like the retired host scatter.
                {
                    // direct single-op scatter into the cache holder; the
                    // in-place-on-input concern (batch W) died with the
                    // is_weight root cause - direct form saves 2 ops/cache
                    auto upd0 = src0->type == src2->type ? upd : cannge_es::Cast(upd, (int64_t) dt);
                    upd0      = cannge_es::Reshape(upd0, std::vector<int64_t>{ src1->ne[0], src2->ne[0] });
                    auto idx2 = cannge_es::Reshape(idxh, std::vector<int64_t>{ src1->ne[0], 1 });
                    auto out  = cannge_es::TensorScatterUpdate(es2, idx2, upd0);
                    ctx.tensors[node] = out;
                    ggml_cannge_view_info vinfo;
                    if (ggml_cannge_resolve_view(src2, vinfo)) {
                        ctx.tensors[vinfo.base] = out;
                    }
                    return true;
                }
            }
            // the node is a view of the cache (src[2]); the write itself is
            // applied by the backend after execute, see io.set_rows
            ctx.tensors[node] = es2;
            // the scatter data tensor is both consumed in-graph (fresh-row
            // splice) and forced as an output port for the host scatter. In
            // large merged graphs GE has been observed feeding such a port a
            // different buffer than the in-graph consumers (writeback diverges
            // while in-graph values stay correct; note-52 batch N). Add a
            // broadcast zero to force a private materialization for the port.
            if (src0 != nullptr && !ggml_is_empty(src0) && ctx.io.options.scatter_add_zero) {
                cannge_es::EsTensorHolder datah;
                const auto lookup = ctx.try_get(src0, datah);
                if (lookup == ggml_cannge_build_result::failure) {
                    return false;
                }
                if (lookup == ggml_cannge_build_result::success) {
                    ge::DataType dtype;
                    if (!ggml_cannge_ge_dtype(src0->type, dtype)) {
                        return ctx.fail("unsupported scatter materialization dtype");
                    }
                    auto zero = ctx.const_cached(std::vector<float>{ 0.0f }, std::vector<int64_t>{ 1, 1 });
                    if (src0->type != GGML_TYPE_F32) {
                        zero = cannge_es::Cast(zero, (int64_t) dtype);
                    }
                    ctx.tensors[src0] = cannge_es::Add(datah, zero);
                }
            }
            return true;
        }

        default:
            return ctx.fail("unsupported tensor operation");
    }
}
