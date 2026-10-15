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

// CANNGE backend op gates: device_supports_op and buffer-type support
// (split out of ggml-cannge.cpp). Each accepted op must have a matching build
// case in graph-build.cpp.

#include "ggml-cannge-priv.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>

bool ggml_cannge_float_dtype(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32;
}

// L0 static capability table (docs/cannge D3): doc-supported (op, dtype,
// shape) combos only, conservative rejects are by design; GE adds/Mul broadcast
// dims that are equal or 1, MatMul contraction is ne0 of both operands
bool ggml_backend_cannge_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    if (!ggml_backend_cannge_graph_env_enabled()) {
        return false;
    }

    // leaf tensors carry no compute, every backend accepts them
    if (op->op == GGML_OP_NONE) {
        return true;
    }

    // metadata view ops (alias/reinterpret, resolved by the consuming op's
    // build): required so pre-allocated views in this backend's buffers can
    // be assigned during split pass 1 (ggml-cpu accepts them; without this a
    // KV cache view in a CANNGE buffer aborts the split, e.g. -fa off)
    if (op->op == GGML_OP_VIEW || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_PERMUTE ||
        op->op == GGML_OP_TRANSPOSE) {
        return true;
    }

    // debug knob: reject every compute op, isolates backend-presence effects
    // (buffer assignment, boundary copies) from op-mapping correctness
    if (getenv("GGML_CANNGE_DISABLE_ALL")) {
        return false;
    }

    // in-place variants write the result back into src[i]'s buffer; GE
    // drops aliased output writes (910B silent, note-10 F1). The *_inplace
    // builders also alias through view_src (out is a view of src0), same
    // dropped-write hazard (note-10 F8); metadata view ops are exempt.
    // SET_ROWS also views its target (src[2]), but the write is applied by
    // the backend explicitly after execute, so it is exempt as well
    if (op->op != GGML_OP_VIEW && op->op != GGML_OP_PERMUTE && op->op != GGML_OP_TRANSPOSE &&
        op->op != GGML_OP_RESHAPE && op->op != GGML_OP_CPY && op->op != GGML_OP_SET_ROWS &&
        op->view_src != nullptr) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // ggml encodes the CPY destination as src[1] = the node itself - not
        // an in-place hazard (the gate's CPY case checks the real src0/src1)
        if (op->src[i] == op && op->op != GGML_OP_CPY) {
            return false;
        }
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV: {
            if (getenv("GGML_CANNGE_NO_ADDMUL")) {
                return false; // debug knob
            }
            if (src0 == nullptr || src1 == nullptr || !ggml_cannge_float_dtype(op->type)) {
                return false;
            }
            if (!ggml_cannge_float_dtype(src0->type) || !ggml_cannge_float_dtype(src1->type)) {
                return false;
            }
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                if (src0->ne[d] != src1->ne[d] && src0->ne[d] != 1 && src1->ne[d] != 1) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_SCALE:
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type);

        case GGML_OP_SOFT_MAX: {
            if (getenv("GGML_CANNGE_NO_SOFTMAX")) {
                return false; // debug knob: bisect the frozen-argmax corruption
            }
            if (src0 == nullptr || !ggml_cannge_float_dtype(src0->type) ||
                !ggml_cannge_float_dtype(op->type)) {
                return false;
            }
            float scale = 1.0f;
            float max_bias = 0.0f;
            std::memcpy(&scale, (const float *) op->op_params + 0, sizeof(scale));
            std::memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(max_bias));
            // attention sinks (src[2]) and ALiBi slopes have no GE mapping yet
            if (op->src[2] != nullptr || max_bias != 0.0f) {
                return false;
            }
            // optional mask, added after scaling; row width must match, other
            // dims broadcast like ADD (GE equal-or-1, docs/cannge note-7)
            if (src1 != nullptr) {
                if (!ggml_cannge_float_dtype(src1->type) || src1->ne[0] != src0->ne[0]) {
                    return false;
                }
                for (int d = 1; d < GGML_MAX_DIMS; d++) {
                    if (src1->ne[d] != 1 && src0->ne[d] != src1->ne[d]) {
                        return false;
                    }
                }
            }
            return true;
        }

        case GGML_OP_UNARY: {
            if (getenv("GGML_CANNGE_NO_UNARY")) {
                return false; // debug knob
            }
            if (src0 == nullptr || !ggml_cannge_float_dtype(src0->type) || !ggml_cannge_float_dtype(op->type)) {
                return false;
            }
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return true;
                default:
                    // STEP has no es_math counterpart; ELU and the rest unmapped
                    return false;
            }
        }

        case GGML_OP_NORM:
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type);

        case GGML_OP_RMS_NORM:
            if (getenv("GGML_CANNGE_NO_RMS")) {
                return false; // debug knob
            }
            // the ADD->RMS->(weight MUL) merged-subgraph corruption this gate
            // once worked around was the cross-split materialization bug:
            // tensors consumed both in-split and cross-split got no GE output
            // port, so cross-split consumers read stale ggml buffers (fixed
            // in ggml-backend.cpp split pass, note-44)
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type);

        case GGML_OP_MUL_MAT: {
            if (getenv("GGML_CANNGE_NO_MATMUL")) {
                return false; // debug knob: force matmuls to CPU for bisection
            }
            if (src0 == nullptr || src1 == nullptr || !ggml_cannge_float_dtype(op->type)) {
                return false;
            }
            // both operands must share the contraction dtype for the batched
            // MatMulV3 path; the M==1 GEMV build casts both sides to f32, so
            // it accepts mixed f16 weights x f32 activations (decode norm
            // outputs feeding the QKV/FFN projections)
            if (!ggml_cannge_float_dtype(src0->type) || !ggml_cannge_float_dtype(src1->type)) {
                return false;
            }
            if (src0->ne[0] != src1->ne[0]) {
                return false; // contraction dim mismatch
            }
            // M==1 GEMV (optionally batched over heads): BatchMatMulV3 would
            // need M aligned and caps weights at 256K elements; the build
            // maps it to Mul(broadcast)+ReduceSum instead (f32 for
            // accumulation parity with ggml-cpu, note-37). note: src1 with
            // trailing singletons reports FEWER ggml dims than src0 - match
            // on ne, not on ggml_n_dims
            if (getenv("GGML_CANNGE_DEBUG_COMPILE") && src0 != nullptr && strstr(src0->name, "attn_output") != nullptr) {
                fprintf(stderr, "[DBG-WO] %s src0=%s ne=%lld,%lld src1=%s ne=%lld,%lld,%lld,%lld\n", op->name,
                        src0->name, (long long) src0->ne[0], (long long) src0->ne[1], src1 ? src1->name : "?",
                        src1 ? (long long) src1->ne[0] : -1, src1 ? (long long) src1->ne[1] : -1,
                        src1 ? (long long) src1->ne[2] : -1, src1 ? (long long) src1->ne[3] : -1);
            }
            bool gemv = src1->ne[1] == 1 && ggml_n_dims(src0) >= 2 && src0->ne[2] == src1->ne[2] &&
                        src0->ne[3] == src1->ne[3];
            if (!gemv && src1->ne[1] == 1 && ggml_n_dims(src0) >= 2 && src0->ne[3] == src1->ne[3] &&
                src0->ne[2] > 0 && src1->ne[2] % src0->ne[2] == 0) {
                // opt-in GQA ratio, split by contraction dim: KQ contracts
                // head_dim (ne0 == n_dims), V contracts the sequence (kv)
                const bool is_kq = src0->ne[0] == 128;
                const char * e   = getenv("GGML_CANNGE_GQA");
                gemv = (is_kq && (strcmp(e ? e : "", "kq") == 0 || strcmp(e ? e : "", "all") == 0)) ||
                       (!is_kq && (strcmp(e ? e : "", "v") == 0 || strcmp(e ? e : "", "all") == 0));
            }
            if (gemv) {
                // huge-N gemvs (the lm head) used to stay on CPU because GE's
                // mul pre-compile rejects one large broadcast (E40021); the
                // build now chunks the weight under the cap and concats
                // (note-50 batch I), so they are accepted here
                // non-dense or mixed-dtype operands are fine: analyze stages
                // external views into dense buffers and the build casts both
                // sides to f32 (permuted cache views feeding the attention
                // matmuls, f16 weights x f32 activations, note-45)
                return true;
            }
            if (src0->type != src1->type) {
                // mixed f16 weights x f32 activations: the build casts both
                // sides to f32 before BatchMatMulV3 (ggml mul_mat accumulates
                // in f32 anyway). Without this, weight_buft_supported's
                // loader probe (f32 test activation) fails and llama never
                // allocates weights on this backend's buffer type - the root
                // reason all matmuls stayed on CPU (note-46 批12a)
                if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32) {
                    return false;
                }
            }
            // 32-byte alignment of M/N/K: not in the op spec (the aclnn spec
            // has no alignment constraint and allows non-contiguous inputs),
            // but the GE graph-compile tiling path hard-fails unaligned
            // MatMul shapes (CheckDimsAligned310P, EZ9999) on 310P and 910B
            // alike; reject conservatively, product-independent. 未文档化，
            // 依据见 note-10 F3
            // spec: <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnMatmul.md>
#ifdef ASCEND_310P
            // 310P TBE registers fp16 only, so GE down-converts F32 matmul inputs
            // to fp16 before the tiling check (F7, note-10): the 32B rule applies
            // to the fp16 element size, F32 needs 16 elements (note-16 known precision)
            const size_t align_elems = src0->type == GGML_TYPE_F32 ? 16 : 32 / ggml_type_size(src0->type);
#else
            const size_t align_elems = 32 / ggml_type_size(src0->type);
#endif
            if (src0->ne[0] % align_elems != 0 || src0->ne[1] % align_elems != 0 || src1->ne[1] % align_elems != 0) {
                return false;
            }
            // non-dense operands (permuted cache/weight views) are staged into
            // dense buffers as graph IO; no contiguity requirement here
            // 2D and batched (3D/4D) matmul: leading batch dims must match,
            // GE BatchMatMulV3 batches on the outermost dims; contraction stays ne0
            if (ggml_n_dims(src0) != ggml_n_dims(src1)) {
                return false;
            }
            // GE once silently returned zeros for weights above ~256K elements
            // (note-10, old CANN); rechecked on 9.1.0 with a 10.5M-element
            // weight probe: exact (1.4e-6, probe bigmm). The cap is gone; keep
            // an opt-out knob in case a future GE revision regresses
            if (src0->ne[0] * src0->ne[1] > 262144 && getenv("GGML_CANNGE_BIGMM") != nullptr) {
                return false;
            }
            return ggml_n_dims(src0) >= 2 && ggml_n_dims(src0) <= 4 && src0->ne[2] == src1->ne[2] &&
                   src0->ne[3] == src1->ne[3];
        }

        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type);

        case GGML_OP_VIEW:
            // mirror the build precondition: dense strides only, the Slice path
            // needs standard contiguous layout relative to the view
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type) &&
                   ggml_is_contiguous(op);

        case GGML_OP_CONT:
            // non-contiguous CONT opts into the merged attention
            // (GGML_CANNGE_CONT_NC): the fresh-row splice makes the merged
            // fragment read correct, but the 1432-node merged graph mis-executes
            // at model scale while the isolated probe is exact - root cause
            // under investigation (note-52). Default stays split (coherent).
            if (getenv("GGML_CANNGE_CONT_NC") == nullptr) {
                return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type) &&
                       ggml_is_contiguous(src0);
            }
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type) &&
                   ggml_nelements(src0) == ggml_nelements(op);

        case GGML_OP_CPY: {
            // F16 <-> F32 via Cast; aliased same-dtype copies into views are
            // rejected until the writeback mechanism is enabled
            const bool ok = src0 != nullptr && src1 != nullptr && ggml_cannge_float_dtype(src0->type) &&
                            ggml_cannge_float_dtype(op->type) && ggml_are_same_shape(src0, src1) &&
                            src1->view_src == nullptr && ggml_is_contiguous(src1);
            if (!ok && getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                fprintf(stderr, "[DBG-GATE] cpy %s reject: s0=%p s1=%p t=%d/%d same=%d vsrc=%p contig=%d\n",
                        op->name, (const void *) src0, (const void *) src1, src0 ? (int) src0->type : -1,
                        (int) op->type, src0 && src1 ? (int) ggml_are_same_shape(src0, src1) : -1,
                        src1 ? (void *) src1->view_src : nullptr,
                        src1 ? (int) ggml_is_contiguous(src1) : -1);
            }
            return ok;
        }

        case GGML_OP_DUP:
            // same-dtype copy into the node's own buffer (in-place variants are
            // rejected by the generic view_src check above); the build aliases
            // src0 and analyze forces a GE output port for the copy. a
            // non-dense src0 keeps failing numerics through the alias+port
            // path (unlike ADD/MUL which consume the staged input in a real
            // op) - reject it, contiguous copies stay supported
            return src0 != nullptr && ggml_cannge_float_dtype(src0->type) && ggml_cannge_float_dtype(op->type) &&
                   ggml_is_contiguous(src0);

        case GGML_OP_GET_ROWS:
            if (getenv("GGML_CANNGE_NO_GET_ROWS")) {
                return false; // debug knob
            }
            // single-row tables miscompile on GE: GatherV2 infers a [1,1]
            // output and GE writes one element, the ggml buffer stays zero
            // (probe ggml_probe_gr0, 910B3 repro; embedding gather unaffected)
            if (src0 != nullptr && src0->ne[1] == 1) {
                return false;
            }
            return src0 != nullptr && src1 != nullptr && ggml_cannge_float_dtype(src0->type) &&
                   src1->type == GGML_TYPE_I32 && ggml_cannge_float_dtype(op->type);

        case GGML_OP_SET_ROWS: {
            // KV cache write: a[src[1][i]] = src[0][i] per row, applied by the
            // backend as post-execute D2D row copies (no GE scatter needed);
            // ggml.c ggml_set_rows: src0 = row data, src1 = indices, src2 = cache
            const struct ggml_tensor * data = op->src[0];
            const struct ggml_tensor * idx  = op->src[1];
            const struct ggml_tensor * dst  = op->src[2];
            if (data == nullptr || idx == nullptr || dst == nullptr) {
                return false;
            }
            // dtype mismatch (f32 rope output -> f16 cache) is handled by a
            // host-side cast in the scatter when the data sits in host memory
            if (!ggml_cannge_float_dtype(data->type) || !ggml_cannge_float_dtype(dst->type) ||
                (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64)) {
                return false;
            }
            if (data->ne[0] != dst->ne[0] || data->ne[2] != dst->ne[2] || data->ne[3] != dst->ne[3]) {
                return false;
            }
            return ggml_is_contiguous_rows(data) && ggml_is_contiguous_rows(dst) && ggml_is_contiguous(dst);
        }

        case GGML_OP_CONCAT: {
            if (src0 == nullptr || src1 == nullptr || !ggml_cannge_float_dtype(op->type)) {
                return false;
            }
            if (src0->type != src1->type || !ggml_cannge_float_dtype(src0->type)) {
                return false;
            }
            const int axis = (int) op->op_params[0];
            if (axis < 0 || axis >= GGML_MAX_DIMS) {
                return false;
            }
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                if (d != axis && src0->ne[d] != src1->ne[d]) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_GLU: {
            // single-input: y = act(first half of ne[0]) * second half
            // (ggml-cpu ops.cpp ggml_compute_forward_reglu_f32). two-input
            // swiglu split: y = silu(src0) * src1 (ggml_vec_swiglu_f32,
            // llama ffn_swiglu_split). other swapped/clamp/two-input variants
            // stay on CPU
            if (src0 == nullptr || !ggml_cannge_float_dtype(src0->type) || !ggml_cannge_float_dtype(op->type) ||
                op->type != src0->type) {
                return false;
            }
            if (src1 != nullptr) {
                if (!ggml_cannge_float_dtype(src1->type) || src1->type != src0->type ||
                    ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(op, 1) != 0 ||
                    !ggml_are_same_shape(src0, src1) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                    return false;
                }
                return true;
            }
            if (ggml_get_op_params_i32(op, 1) != 0 || src0->ne[0] % 2 != 0 || !ggml_is_contiguous(src0)) {
                return false;
            }
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return true;
                default:
                    return false;
            }
        }

        case GGML_OP_ROPE: {
            if (getenv("GGML_CANNGE_NO_ROPE")) {
                return false; // debug knob
            }
            // minimal subset: full-head NORMAL/NEOX rotation, no YaRN scaling
            // (freq_scale/ext/attn/betas at defaults), no freq factors (src2),
            // no n_offs; cos/sin tables are built inside the graph from src1
            const struct ggml_tensor * src2 = op->src[2];
            if (src0 == nullptr || src1 == nullptr || src2 != nullptr || !ggml_cannge_float_dtype(src0->type) ||
                op->type != src0->type || src1->type != GGML_TYPE_I32) {
                if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                    fprintf(stderr, "[DBG-GATE] rope %s reject-A: src2=%p t0=%d t_op=%d t1=%d\n", op->name,
                            (const void *) src2, src0 ? (int) src0->type : -1, (int) op->type,
                            src1 ? (int) src1->type : -1);
                }
                return false;
            }
            const int n_dims = ggml_get_op_params_i32(op, 1);
            const int mode   = ggml_get_op_params_i32(op, 2);
            if ((mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) || n_dims != src0->ne[0] ||
                n_dims % 2 != 0 || ggml_get_op_params_i32(op, 15) != 0) {
                if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                    fprintf(stderr, "[DBG-GATE] rope %s reject-B: mode=%d n_dims=%d ne0=%lld p15=%d\n", op->name,
                            mode, n_dims, (long long) src0->ne[0], ggml_get_op_params_i32(op, 15));
                }
                return false;
            }
            // op_params floats: [5]freq_base [6]freq_scale [7]ext_factor
            // [8]attn_factor [9]beta_fast [10]beta_slow (ggml.c ggml_rope_impl)
            // beta_fast/beta_slow only steer the YaRN ramp, dead at ext==0
            // (ggml-cpu rope); Qwen3-2507 ships 32/1 with ext 0. The earlier
            // model corruption from relaxing this was the SET_ROWS scatter
            // default-stream race (fixed by draining ctx->stream first, note-40)
            if (ggml_get_op_params_f32(op, 6) != 1.0f || ggml_get_op_params_f32(op, 7) != 0.0f ||
                ggml_get_op_params_f32(op, 8) != 1.0f) {
                if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                    fprintf(stderr, "[DBG-GATE] rope %s reject-C: fs=%f ext=%f attn=%f\n", op->name,
                            ggml_get_op_params_f32(op, 6), ggml_get_op_params_f32(op, 7),
                            ggml_get_op_params_f32(op, 8));
                }
                return false;
            }
            const int nd = ggml_n_dims(src0);
            const bool ok = nd >= 2 && nd <= 4 && ggml_is_contiguous(src0);
            if (!ok && getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                fprintf(stderr, "[DBG-GATE] rope %s rejected: nd=%d ne=%lld,%lld,%lld,%lld nb0=%zu contig=%d\\n",
                        op->name, nd, (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2],
                        (long long) src0->ne[3], (size_t) src0->nb[0], (int) ggml_is_contiguous(src0));
            }
            return ok;
        }

        case GGML_OP_FLASH_ATTN_EXT: {
            // attention on device via GE IFA (BSND decode); K/V must be f16
            // and stay resident (cache views) - no host round-trip. Opt-in:
            // only s_q == 1 (decode) is claimed - IFA is bit-exact there
            // (probe fa). PFA (s_q > 1) silently reads BNSD despite the
            // layout string (probe fa anatomy, note-42), so prefill falls
            // back to the kq_soft_max path
            if (getenv("GGML_CANNGE_FA") == nullptr) {
                return false;
            }
            const struct ggml_tensor * src2 = op->src[2];
            if (src0 == nullptr || src1 == nullptr || src2 == nullptr || op->src[4] != nullptr) {
                return false; // sinks unsupported
            }
            if (src0->ne[1] != 1) {
                if (getenv("GGML_CANNGE_DEBUG_COMPILE")) fprintf(stderr, "[DBG-GATE] fa reject-A prefill s_q=%lld\n", (long long) src0->ne[1]);
                return false;
            }
            if (src1->type != GGML_TYPE_F16 || src2->type != GGML_TYPE_F16) {
                if (getenv("GGML_CANNGE_DEBUG_COMPILE")) fprintf(stderr, "[DBG-GATE] fa reject-B kv type\n");
                return false;
            }
            if (src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) {
                if (getenv("GGML_CANNGE_DEBUG_COMPILE")) fprintf(stderr, "[DBG-GATE] fa reject-C q type\n");
                return false;
            }
            // ggml FA output dtype follows the precision flag (F32 default),
            // independent of the f16 q/k/v inputs
            if (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32) {
                return false;
            }
            float max_bias = 0.0f, softcap = 0.0f;
            std::memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(max_bias));
            std::memcpy(&softcap, (const float *) op->op_params + 2, sizeof(softcap));
            if (max_bias != 0.0f || softcap != 0.0f) {
                return false; // ALiBi / softcapping unsupported
            }
            const int64_t d = src0->ne[0];
            if (d % 16 != 0 || d != src1->ne[0] || d != src2->ne[0] || src0->ne[2] % src1->ne[2] != 0) {
                if (getenv("GGML_CANNGE_DEBUG_COMPILE")) fprintf(stderr, "[DBG-GATE] fa reject-D dims\n");
                return false; // head dim alignment + GQA ratio
            }
            if (src1->ne[1] != src2->ne[1] || src0->ne[3] != 1) {
                if (getenv("GGML_CANNGE_DEBUG_COMPILE")) fprintf(stderr, "[DBG-GATE] fa reject-E s/ndims\n");
                return false;
            }
            const struct ggml_tensor * src3 = op->src[3];
            if (src3 != nullptr && src3->type != GGML_TYPE_F16 && src3->type != GGML_TYPE_F32) {
                return false;
            }
            if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                fprintf(stderr, "[DBG-GATE] fa %s accept ne=%lld,%lld,%lld,%lld ne1=%lld,%lld ne2=%lld,%lld t=%d/%d/%d mt=%d\n",
                        op->name, (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2],
                        (long long) src0->ne[3], (long long) src1->ne[1], (long long) src1->ne[2],
                        (long long) src2->ne[1], (long long) src2->ne[2], (int) src0->type, (int) src1->type,
                        (int) op->type, src3 ? (int) src3->type : -1);
            }
            return true;
        }

        default:
            return false;
    }
}

bool ggml_backend_cannge_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    if (getenv("GGML_CANNGE_HOST_BUFT")) {
        return buft->iface.get_name == ggml_backend_cannge_buffer_type_name ||
               buft->iface.get_name == ggml_backend_cpu_buffer_type()->iface.get_name;
    }
    return buft->iface.get_name == ggml_backend_cannge_buffer_type_name;
    // NOTE: host-buft support (weight ops assigned to CANNGE at split pass 1,
    // note-46) is the right idea but surfaces successive GE landmines: Mul
    // broadcast pre-compile rejects huge lm-head gemv (capped at 32M elems
    // now), keep-dims ReduceSum folds (output port size mismatch, now
    // keep=false), and a rogue [2560,32]x[2560,4096] Mul in the prefill
    // graph (unresolved). Parked 2026-10-15
}
