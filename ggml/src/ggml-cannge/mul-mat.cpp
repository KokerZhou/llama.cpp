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

#include "mul-mat.h"
#include "supports.h"
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

// emit the cast only when it changes dtype; GE did not fold the f16->f16
// casts we used to emit per gemv (note-52 batch-noise), and each surviving
// cast is a real kernel on the merged graph
static cannge_es::EsTensorHolder cast_if_needed(cannge_es::EsTensorHolder h, enum ggml_type src_type,
                                                ge::DataType dst) {
    ge::DataType sdt;
    if (!ggml_cannge_ge_dtype(src_type, sdt) || sdt == dst) {
        return h;
    }
    return cannge_es::Cast(h, (int64_t) dst);
}


#include "graph-build-priv.h"


bool ggml_cannge_build_matmul(ggml_cannge_build_context & ctx, ggml_tensor * node) {
    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];
    cannge_es::EsTensorHolder es0;
    cannge_es::EsTensorHolder es1;
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
    // GEMV helper: explicit full-rank 4D GE shapes {ne3,ne2,ne1,ne0}
    // for both operands - input holders report trimmed ggml dims while
    // internal producers may be 2D, and GE Mul pre-compile rejects
    // mixed-rank broadcast (E40021 Mul_16, note-46)
    auto gemv_full4 = [&](const ggml_tensor * t, cannge_es::EsTensorHolder & h) {
        const std::vector<int64_t> shape = { t->ne[3], t->ne[2], t->ne[1], t->ne[0] };
        return cannge_es::Reshape(h, shape);
    };
    const bool use_gemv = ggml_cannge_use_gemv(src0, src1, ctx.io.options);
    if (use_gemv && src1->ne[2] == 1 && src1->ne[3] == 1) {
        // M==1 GEMV: y[n] = sum_k w[k,n]*x[k]. f32 casts keep the f32
        // accumulation of ggml-cpu (f16 product+sum loses mantissa);
        // the [K,N] f32 product is HBM-bandwidth-bound, still far
        // cheaper than a CPU split boundary (note-36/37)
        // f16 product (default; GGML_CANNGE_GEMV16=0 opts out): halves
        // the HBM traffic of the materialized [K,N] product - the
        // gemvs are bandwidth-bound (post-sync time halved, note-49 H);
        // GE accumulates the reduction internally, probe-measured
        // max_rel 5.5e-4 vs f32
        const bool f16prod = ctx.io.options.gemv_f16;
        ge::DataType f32;
        if (!ggml_cannge_ge_dtype(f16prod ? GGML_TYPE_F16 : GGML_TYPE_F32, f32)) {
            return ctx.fail("no GE product dtype for gemv");
        }
        std::vector<int64_t> axes = { 1 }; // GE K axis
        if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
            fprintf(stderr, "[DBG-GEMV] %s src0 ne=%lld,%lld,%lld,%lld src1 ne=%lld,%lld,%lld,%lld\n",
                    node->name, (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2],
                    (long long) src0->ne[3], (long long) src1->ne[0], (long long) src1->ne[1],
                    (long long) src1->ne[2], (long long) src1->ne[3]);
        }
        if (ctx.io.options.gemv_bmm) {
            // M==1 via one BatchMatMulV3 (cube unit, f32 internal accumulate)
            // instead of the Mul+ReduceSum decomposition; no broadcast Mul, so
            // the lm-head N chunking does not apply either. Operands must be
            // 3D: the 2D form fails BatchMatMulV3 tiling on this toolkit
            ge::DataType f16d;
            if (!ggml_cannge_ge_dtype(GGML_TYPE_F16, f16d)) {
                return ctx.fail("no GE f16 for bmm gemv");
            }
            const int64_t K = src0->ne[0], N = src0->ne[1];
            auto x1 = cannge_es::Reshape(cast_if_needed(es1, src1->type, f16d), ctx.vector_cached(std::vector<int64_t>{ 1, 1, K }));
            auto x2 = cannge_es::Reshape(cast_if_needed(es0, src0->type, f16d), ctx.vector_cached(std::vector<int64_t>{ 1, N, K }));
            auto y  = cannge_es::BatchMatMulV3(x1, x2, nullptr, nullptr, false, true);
            auto out = cannge_es::Reshape(y, ggml_cannge_ge_shape(node));
            ge::DataType dt;
            if (!ggml_cannge_ge_dtype(node->type, dt)) {
                return ctx.fail("no GE output dtype for bmm gemv");
            }
            ctx.tensors[node] = cannge_es::Cast(out, (int64_t) dt);
            return true;
        }
        auto do_gemv = [&](cannge_es::EsTensorHolder w, int64_t n_off, int64_t n_len,
                           cannge_es::EsTensorHolder x) -> cannge_es::EsTensorHolder {
            std::vector<int64_t> off = { n_off, 0 };
            std::vector<int64_t> sz  = { n_len, src0->ne[0] };
            auto                 wc  = n_off == 0 && n_len == src0->ne[1] ? w : cannge_es::Slice(w, off, sz);
            // logits feed the sampler: keep the f32 product despite
            // the 2x traffic (GGML_CANNGE_GEMV16 does not apply here)
            ge::DataType pf32;
            if (!ggml_cannge_ge_dtype(GGML_TYPE_F32, pf32)) {
                ctx.fail("no GE f32 for chunked gemv");
            }
            auto prod = cannge_es::Mul(cannge_es::Cast(wc, (int64_t) pf32), cannge_es::Cast(x, (int64_t) pf32));
            auto sum  = cannge_es::ReduceSum(prod, ctx.vector_cached(axes), false);
            ge::DataType dt;
            if (!ggml_cannge_ge_dtype(GGML_TYPE_F32, dt)) {
                ctx.fail("no GE f32 for chunked gemv");
            }
            return cannge_es::Cast(sum, (int64_t) dt);
        };
        // huge N (the lm head): GE mul pre-compile rejects the single
        // large broadcast (E40021, note-36) - split the weight into N
        // chunks under the cap, one gemv each, concat along GE dim0
        const int64_t n     = src0->ne[1];
        const int64_t chunk = 13107; // 2560*13107 < 33.5M f32 cap... f16 halves further
        if (n > chunk && src0->ne[2] == 1 && src0->ne[3] == 1) {
            std::vector<cannge_es::EsTensorHolder> parts;
            for (int64_t off = 0; off < n; off += chunk) {
                parts.push_back(do_gemv(es0, off, std::min(chunk, n - off), es1));
            }
            auto out = parts.size() == 1 ? parts[0]
                                         : cannge_es::Concat(ctx.scalar_cached((int64_t) 0), parts,
                                                             (int) parts.size());
            out      = cannge_es::Reshape(out, ggml_cannge_ge_shape(node));
            ctx.tensors[node] = out;
            return true;
        }
        auto prod = cannge_es::Mul(cast_if_needed(es0, src0->type, f32), cast_if_needed(es1, src1->type, f32));
        auto sum  = cannge_es::ReduceSum(prod, ctx.vector_cached(axes), false);
        if (f16prod || node->type != GGML_TYPE_F32) {
            // ggml mul_mat output is always F32 (ggml.c); the f16
            // product path must still present f32 to the writeback
            ge::DataType dt;
            if (!ggml_cannge_ge_dtype(node->type, dt)) {
                return ctx.fail(std::string("unsupported gemv output dtype ") + node->name);
            }
            sum = cannge_es::Cast(sum, (int64_t) dt);
        }
        ctx.tensors[node] = sum;
        return true;
    }
    if (use_gemv) {
        // batched M==1 GEMV (classic attention kq/v over heads):
        // (B3,B2,N,K) x (B3,B2,1,K) -> Mul broadcasts, reduce the K
        // axis, keep dims; kept dims are memory-identical to the
        // node's (B3,B2,1,N) GE shape. GQA: when the contraction
        // operand has fewer heads (8 kv vs 32 q), BroadcastTo expands
        // it to the query head count (v_cont output -> kqv matmul)
        // same f16-product rationale as the M==1 branch (note-50 H);
        // these products are small ({G,R,N,K} <= a few MB) so the win
        // is minor, but it is free
        const bool bf16prod = ctx.io.options.gemv_f16;
        ge::DataType f32;
        if (!ggml_cannge_ge_dtype(bf16prod ? GGML_TYPE_F16 : GGML_TYPE_F32, f32)) {
            return ctx.fail("no GE f32 dtype for gemv");
        }
        // GQA without any expansion op: the Mul broadcasts a
        // size-1 replica axis instead. q-head h = kv-head g * R + r;
        // the reduced {G,R,N} relabels to {H,1,N} with the same
        // memory order. All reshapes are byte-identical relabels
        cannge_es::EsTensorHolder e0 = es0;
        cannge_es::EsTensorHolder e1 = es1;
        if (src0->ne[2] != src1->ne[2]) {
            // ggml GQA maps q-head h -> kv-head g = h / R (ggml-cpu
            // ggml-cpu.c i02 = i12 / r2), i.e. h = g*R + r. The kv
            // side takes g on dim0 and a broadcast 1 on dim1, the q
            // side takes (g, r) on dims (0, 1):
            // Mul({G,1,N,K} x {G,R,1,K}) -> {G,R,N,K} -> reduce K ->
            // {G,R,N}, a memory-identity relabel of {H,1,N}
            // (size-1 dims move freely; no byte ever changes place)
            const int64_t R = src1->ne[2] / src0->ne[2];
            const int64_t G = src0->ne[2];
            const int64_t K = src0->ne[0];
            const int64_t N = src0->ne[1];
            e0 = cannge_es::Reshape(e0, std::vector<int64_t>{ G, 1, N, K });
            e1 = cannge_es::Reshape(e1, std::vector<int64_t>{ G, R, 1, K });
        }
        // K sits at the last GE axis in both layouts
        std::vector<int64_t> axes = { src0->ne[2] != src1->ne[2] ? 3 : (int64_t) ggml_n_dims(src0) - 1 };
        auto prod = cannge_es::Mul(cast_if_needed(e0, src0->type, f32), cast_if_needed(e1, src1->type, f32));
        auto sum  = cannge_es::ReduceSum(prod, ctx.vector_cached(axes), false);
        // fresh-row splice (note-50 batch K): the K view chain reads
        // the cache pre-scatter; when the fragment produces Kcur, add
        // the current slot's scores (one tiny gemv) at the slot index
        // of the pre-relabel sum {B3,B2,N(,K consumed)}
        if (!ctx.io.options.disable_splice) {
            ggml_cannge_view_info kinfo;
            if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
                fprintf(stderr, "[DBG-SPLICE] kq-side src0=%s vsrc=%p\n", src0->name, (void *) src0->view_src);
            }
            if (ggml_cannge_resolve_view(src0, kinfo)) {
                if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
                    fprintf(stderr, "[DBG-SPLICE] kq-side base=%s\n", kinfo.base->name);
                }
                // single-row scatter only: this gemv is M==1, and the
                // prefill multi-row scatter is served by the V-side
                // matrix splice (the M>1 attention goes BatchMatMulV3)
                const ggml_cannge_set_rows_op * sr = ggml_cannge_sr_for_base(ctx, kinfo.base);
                if (sr != nullptr && ggml_nelements(sr->idx) != 1) {
                    sr = nullptr;
                }
                cannge_es::EsTensorHolder rowh, idxh;
                if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
                    fprintf(stderr, "[DBG-SPLICE] kq-side sr=%s dataget=%d idxget=%d\n",
                            sr ? "found" : "none", sr && ctx.tensors.count(sr->data) > 0,
                            sr && ctx.tensors.count(sr->idx) > 0);
                }
                if (sr != nullptr && ctx.get(sr->data, rowh) && ctx.get(sr->idx, idxh)) {
                    const int64_t N     = src0->ne[1];
                    const int64_t K     = src0->ne[0];
                    const int64_t G     = src0->ne[2];
                    const int64_t R     = src1->ne[2] / G;
                    // kq_cur[g,r,1] = sum_k Kcur[k,g] * q[k,r,g]:
                    // Kcur {K,G} -> {G,1,1,K}; e1 is {G,R,1,K}
                    auto kr       = cannge_es::Reshape(rowh, std::vector<int64_t>{ G, 1, 1, K });
                    auto cur_prod = cannge_es::Mul(cast_if_needed(kr, src0->type == GGML_TYPE_F16 ? GGML_TYPE_F32 : src0->type, f32), cast_if_needed(e1, src1->type, f32));
                    auto cur_sum  = cannge_es::ReduceSum(cur_prod, ctx.vector_cached(std::vector<int64_t>{ 3 }), false);
                    // {G,R,1} -> {G,R,1(,1)}; select into {G,R,N(,1)}
                    cannge_es::EsTensorHolder m;
                    if (!ggml_cannge_onehot_slot(ctx, idxh, N, m, sr->idx)) {
                        return false;
                    }
                    auto m3  = cannge_es::Reshape(m, std::vector<int64_t>{ 1, 1, N });
                    ge::DataType f32d;
                    if (!ggml_cannge_ge_dtype(GGML_TYPE_F32, f32d)) {
                        return ctx.fail("no GE f32 for kq splice");
                    }
                    auto one = ctx.const_cached(std::vector<float>(N, 1.0f), std::vector<int64_t>{ 1, 1, N });
                    auto s32 = cannge_es::Cast(sum, (int64_t) f32d);
                    auto c32 = cannge_es::Cast(cur_sum, (int64_t) f32d);
                    sum      = cannge_es::Add(cannge_es::Mul(s32, cannge_es::Sub(one, m3)), cannge_es::Mul(c32, m3));
                }
            }
        }
        if (src0->ne[2] != src1->ne[2]) {
            const int64_t R = src1->ne[2] / src0->ne[2];
            const int64_t G = src0->ne[2];
            const int64_t N = src0->ne[1];
            sum = cannge_es::Reshape(sum, std::vector<int64_t>{ R * G, 1, N });
        }
        if (bf16prod) {
            ge::DataType dt;
            if (!ggml_cannge_ge_dtype(node->type, dt)) {
                return ctx.fail(std::string("unsupported gemv output dtype ") + node->name);
            }
            sum = cannge_es::Cast(sum, (int64_t) dt);
        }
        if (node->type != GGML_TYPE_F32) {
            ge::DataType dt;
            if (!ggml_cannge_ge_dtype(node->type, dt)) {
                return ctx.fail(std::string("unsupported gemv output dtype ") + node->name);
            }
            sum = cannge_es::Cast(sum, (int64_t) dt);
        }
        ctx.tensors[node] = sum;
        return true;
    }
    // mixed f16 weights x f32 activations (weights allocated on this
    // backend after the gate relaxed, note-46 batch 12a): BatchMatMulV3
    // needs one dtype - cast both to f32, matching ggml's f32
    // accumulation and the always-f32 mul_mat result
    bool mixed = src0->type != src1->type;
    if (mixed) {
        // mixed f16 weights x f32 activations (weights allocated on this
        // backend after the gate relaxed, note-46 batch 12a): BatchMatMulV3
        // needs one dtype - cast both to f32, matching ggml's f32
        // accumulation and the always-f32 mul_mat result
        ge::DataType f32;
        if (!ggml_cannge_ge_dtype(GGML_TYPE_F32, f32)) {
            return ctx.fail("no GE f32 dtype for mixed mul_mat");
        }
        es0 = cannge_es::Cast(es0, (int64_t) f32);
        es1 = cannge_es::Cast(es1, (int64_t) f32);
    }
    // rank-2 M>1 with M off the GE tiling multiple: zero-pad M, slice back
    const int64_t M_in  = src1->ne[1];
    const int64_t M_pad = ggml_n_dims(src1) == 2 ? (M_in + 15) / 16 * 16 : M_in;
    if (M_pad != M_in) {
        const enum ggml_type xtype = mixed ? GGML_TYPE_F32 : src1->type;
        ge::DataType         xdt;
        if (!ggml_cannge_ge_dtype(xtype, xdt)) {
            return ctx.fail("no GE dtype for mul_mat pad");
        }
        auto zeros = ctx.const_cached(std::vector<float>((M_pad - M_in) * src0->ne[0], 0.0f),
                                      { M_pad - M_in, src0->ne[0] });
        if (xdt != ge::DT_FLOAT) {
            zeros = cannge_es::Cast(zeros, (int64_t) xdt);
        }
        es1 = cannge_es::Concat(ctx.scalar_cached((int64_t) 0), { es1, zeros }, 2);
    }
    auto mm = cannge_es::BatchMatMulV3(es1, es0, nullptr, nullptr, false, true);
    if (M_pad != M_in) {
        mm = cannge_es::Slice(mm, std::vector<int64_t>{ 0, 0 }, std::vector<int64_t>{ M_in, src0->ne[1] });
    }
    if (node->type != (mixed ? GGML_TYPE_F32 : src0->type)) {
        ge::DataType dt;
        if (!ggml_cannge_ge_dtype(node->type, dt)) {
            return ctx.fail(std::string("unsupported mul_mat output dtype ") + node->name);
        }
        mm = cannge_es::Cast(mm, (int64_t) dt);
    }
    ctx.tensors[node] = mm;
    return true;
}
