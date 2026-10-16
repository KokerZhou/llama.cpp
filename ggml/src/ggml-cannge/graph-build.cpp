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

// one step of a pure-rearrangement view chain (see plan.cpp input
// substitution): relabel the holder per the step's op. PERMUTE/TRANSPOSE
// materialize in-GE; VIEW/RESHAPE are byte-identity reshapes (contiguity and
// zero offset checked by the caller's chain policy)

#include "graph-build-priv.h"


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
                    return false;
                }
            }
        }
        ctx.tensors[node] = op(es0, es1);
        return true;
    };

    auto binary_dbg = [&ctx](const char * nm, ggml_tensor * s0, ggml_tensor * s1) {
        if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
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
            // GEMV helper: explicit full-rank 4D GE shapes {ne3,ne2,ne1,ne0}
            // for both operands - input holders report trimmed ggml dims while
            // internal producers may be 2D, and GE Mul pre-compile rejects
            // mixed-rank broadcast (E40021 Mul_16, note-46)
            auto gemv_full4 = [&](const ggml_tensor * t, cannge_es::EsTensorHolder & h) {
                const std::vector<int64_t> shape = { t->ne[3], t->ne[2], t->ne[1], t->ne[0] };
                return cannge_es::Reshape(h, shape);
            };
            if (src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1) {
                // M==1 GEMV: y[n] = sum_k w[k,n]*x[k]. f32 casts keep the f32
                // accumulation of ggml-cpu (f16 product+sum loses mantissa);
                // the [K,N] f32 product is HBM-bandwidth-bound, still far
                // cheaper than a CPU split boundary (note-36/37)
                // f16 product (default; GGML_CANNGE_GEMV16=0 opts out): halves
                // the HBM traffic of the materialized [K,N] product - the
                // gemvs are bandwidth-bound (post-sync time halved, note-49 H);
                // GE accumulates the reduction internally, probe-measured
                // max_rel 5.5e-4 vs f32
                const char * g16 = getenv("GGML_CANNGE_GEMV16");
                const bool f16prod = g16 == nullptr || strcmp(g16, "0") != 0;
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
                    auto sum  = cannge_es::ReduceSum(prod, ctx.builder.CreateVector(axes), false);
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
                                                 : cannge_es::Concat(ctx.builder.CreateScalar((int64_t) 0), parts,
                                                                     (int) parts.size());
                    out      = cannge_es::Reshape(out, ggml_cannge_ge_shape(node));
                    ctx.tensors[node] = out;
                    return true;
                }
                auto prod = cannge_es::Mul(cannge_es::Cast(es0, (int64_t) f32), cannge_es::Cast(es1, (int64_t) f32));
                auto sum  = cannge_es::ReduceSum(prod, ctx.builder.CreateVector(axes), false);
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
            const bool gqa_ratio = src0->ne[2] != src1->ne[2] && src0->ne[2] > 0 &&
                                   src1->ne[2] % src0->ne[2] == 0;
            const bool gqa_on    = gqa_ratio && getenv("GGML_CANNGE_GQA") != nullptr;
            if (src1->ne[1] == 1 && ggml_n_dims(src0) >= 2 && src0->ne[3] == src1->ne[3] &&
                (src0->ne[2] == src1->ne[2] || gqa_on)) {
                // batched M==1 GEMV (classic attention kq/v over heads):
                // (B3,B2,N,K) x (B3,B2,1,K) -> Mul broadcasts, reduce the K
                // axis, keep dims; kept dims are memory-identical to the
                // node's (B3,B2,1,N) GE shape. GQA: when the contraction
                // operand has fewer heads (8 kv vs 32 q), BroadcastTo expands
                // it to the query head count (v_cont output -> kqv matmul)
                // same f16-product rationale as the M==1 branch (note-50 H);
                // these products are small ({G,R,N,K} <= a few MB) so the win
                // is minor, but it is free
                const char * bg16 = getenv("GGML_CANNGE_GEMV16");
                const bool bf16prod = bg16 == nullptr || strcmp(bg16, "0") != 0;
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
                auto prod = cannge_es::Mul(cannge_es::Cast(e0, (int64_t) f32), cannge_es::Cast(e1, (int64_t) f32));
                auto sum  = cannge_es::ReduceSum(prod, ctx.builder.CreateVector(axes), false);
                // fresh-row splice (note-50 batch K): the K view chain reads
                // the cache pre-scatter; when the fragment produces Kcur, add
                // the current slot's scores (one tiny gemv) at the slot index
                // of the pre-relabel sum {B3,B2,N(,K consumed)}
                if (getenv("GGML_CANNGE_NO_SPLICE") == nullptr) {
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
                            auto cur_prod = cannge_es::Mul(cannge_es::Cast(kr, (int64_t) f32), cannge_es::Cast(e1, (int64_t) f32));
                            auto cur_sum  = cannge_es::ReduceSum(cur_prod, ctx.builder.CreateVector(std::vector<int64_t>{ 3 }), false);
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
                            auto one = ctx.builder.CreateConst(std::vector<float>(N, 1.0f), std::vector<int64_t>{ 1, 1, N });
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
            // backend after the gate relaxed, note-46 批12a): BatchMatMulV3
            // needs one dtype - cast both to f32, matching ggml's f32
            // accumulation and the always-f32 mul_mat result
            bool mixed = src0->type != src1->type;
            if (mixed) {
                // mixed f16 weights x f32 activations (weights allocated on this
                // backend after the gate relaxed, note-46 批12a): BatchMatMulV3
                // needs one dtype - cast both to f32, matching ggml's f32
                // accumulation and the always-f32 mul_mat result
                ge::DataType f32;
                if (!ggml_cannge_ge_dtype(GGML_TYPE_F32, f32)) {
                    return ctx.fail("no GE f32 dtype for mixed mul_mat");
                }
                es0 = cannge_es::Cast(es0, (int64_t) f32);
                es1 = cannge_es::Cast(es1, (int64_t) f32);
            }
            auto mm = cannge_es::BatchMatMulV3(es1, es0, nullptr, nullptr, false, true);
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
            // GGML_CANNGE_CONT_NC: non-dense inputs staged in the view's own
            // shape (isolated-opbench OK, in-model corrupt - opt-in). cont
            // never changes the element count; cont_2d relabels (size-1 merge)
            // reshape to the node's shape - plain alias would carry the SRC
            // shape downstream (Mul_2 E40021, note-46 批11a)
            if (!ggml_is_contiguous(src0) && getenv("GGML_CANNGE_CONT_NC") == nullptr &&
                getenv("GGML_CANNGE_NO_CONT_NC") == nullptr) {
                // default-accepted since the fresh-row splice makes the
                // merged-fragment read correct (note-50 batch K); both envs
                // are kept as escape hatches during the transition
            } else if (!ggml_is_contiguous(src0) && getenv("GGML_CANNGE_NO_CONT_NC") != nullptr) {
                return ctx.fail(std::string("non-contiguous CONT input ") + src0->name);
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
                if (ggml_cannge_build_cont_chain(ctx, node, chained)) {
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
            if (src0 != nullptr && getenv("GGML_CANNGE_SR_ADD0") != nullptr) {
                cannge_es::EsTensorHolder datah;
                if (ctx.get(src0, datah)) {
                    ge::DataType dt0;
                    if (ggml_cannge_ge_dtype(src0->type, dt0)) {
                        ge::DataType f32d;
                        if (ggml_cannge_ge_dtype(GGML_TYPE_F32, f32d)) {
                            auto zc = ctx.builder.CreateConst(std::vector<float>{ 0.0f }, std::vector<int64_t>{ 1, 1 });
                            if (src0->type != GGML_TYPE_F32) {
                                zc = cannge_es::Cast(zc, (int64_t) dt0);
                            }
                            ctx.tensors[src0] = cannge_es::Add(datah, zc);
                        }
                    }
                }
            }
            return true;
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
                    auto e = cannge_es::Reshape(t, shape3);
                    std::vector<cannge_es::EsTensorHolder> two = { e, e };
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

        case GGML_OP_FLASH_ATTN_EXT: {
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
            const char * fa_layout = getenv("GGML_CANNGE_FA_LAYOUT") ? getenv("GGML_CANNGE_FA_LAYOUT") : "BSND";
            const bool   fa_inperm = getenv("GGML_CANNGE_FA_INPERM") != nullptr;
            const bool   fa_outperm = getenv("GGML_CANNGE_FA_OUTPERM") != nullptr;
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
            const int64_t fa_pre  = getenv("GGML_CANNGE_FA_PRE")  ? atoll(getenv("GGML_CANNGE_FA_PRE"))  : 2147483647;
            const int64_t fa_next = getenv("GGML_CANNGE_FA_NEXT") ? atoll(getenv("GGML_CANNGE_FA_NEXT")) : 2147483647;
            const bool fa_ifa_all = getenv("GGML_CANNGE_FA_IFA_ALL") != nullptr;
            const int64_t fa_asl  = getenv("GGML_CANNGE_FA_ASL") ? atoll(getenv("GGML_CANNGE_FA_ASL")) : 0;
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
