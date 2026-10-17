// CANNGE graph inputs and fresh-row splicing.

#include "graph-build-priv.h"

#include <utility>

bool ggml_cannge_build_inputs(ggml_cannge_build_context & ctx) {
    for (size_t i = 0; i < ctx.io.inputs.size(); i++) {
        ggml_tensor * t = ctx.io.inputs[i];
        ge::DataType  dt;
        if (!ggml_cannge_ge_dtype(t->type, dt)) {
            return ctx.fail(std::string("unsupported dtype of input ") + t->name);
        }
        ctx.tensors[t] = ctx.builder.CreateInput((int64_t) i, t->name, dt, ge::FORMAT_ND, ggml_cannge_ge_shape(t));
    }
    if (ctx.io.fa_mask != nullptr && ctx.io.fa_asl_dev != nullptr) {
        ctx.fa_asl_holder = ctx.builder.CreateInput((int64_t) ctx.io.inputs.size(), "fa_asl", ge::DT_INT64,
                                                    ge::FORMAT_ND, std::vector<int64_t>{ 1 });
        ctx.has_fa_asl    = true;
    }
    return true;
}

// --- fresh-row splice (note-50 batch K) -------------------------------------
// The KV cache view chains read the cache INPUT, but the SET_ROWS write lands
// post-execute: an in-fragment consumer sees every slot except the current
// one. When the fragment also produces the scattered rows (merged attention),
// splice them in arithmetically (one-hot select; GE scatter ops no-op on this
// stack, note-49). Falls back silently when the row data is not in-fragment
// (split layout: the staged CPU-side cont already holds the fresh bytes).

const ggml_cannge_set_rows_op * ggml_cannge_sr_for_base(ggml_cannge_build_context & ctx, ggml_tensor * base) {
    for (const auto & sr : ctx.io.set_rows) {
        if (sr.in_graph || sr.data == nullptr || sr.idx == nullptr) {
            continue;
        }
        ggml_cannge_view_info dinfo;
        if (ggml_cannge_resolve_view(sr.dst, dinfo) && dinfo.base == base) {
            return &sr;
        }
    }
    return nullptr;
}

// one-hot f32 {S} with 1 at the slot held by idx (1 element)
bool ggml_cannge_onehot_slot(ggml_cannge_build_context & ctx, cannge_es::EsTensorHolder idxh, int64_t S,
                                    cannge_es::EsTensorHolder & mask, ggml_tensor * idx) {
    ge::DataType f32;
    if (!ggml_cannge_ge_dtype(GGML_TYPE_F32, f32)) {
        return ctx.fail("no GE f32 for one-hot");
    }
    std::vector<float> iota(S);
    for (int64_t i = 0; i < S; i++) {
        iota[i] = (float) i;
    }
    auto arange = ctx.builder.CreateConst(iota, std::vector<int64_t>{ S });
    auto idxf   = idx->type == GGML_TYPE_F32 ? idxh : cannge_es::Cast(idxh, (int64_t) f32);
    auto onef   = ctx.builder.CreateConst(std::vector<float>(S, 1.0f), std::vector<int64_t>{ S });
    mask        = cannge_es::Relu(cannge_es::Sub(onef, cannge_es::Abs(cannge_es::Sub(arange, idxf))));
    return true;
}

// CONT of a pure-rearrangement view (the V cache transpose): the cont bytes
// equal the transpose holder's GE layout, so the cont resolves to the in-GE
// relabel chain; then the fresh V row is spliced in when available.
static bool ggml_cannge_splice_fresh_row(ggml_cannge_build_context & ctx, ggml_tensor * t, ggml_tensor * v,
                                         cannge_es::EsTensorHolder & h);

static ggml_cannge_build_result ggml_cannge_build_cont_chain_impl(ggml_cannge_build_context & ctx, ggml_tensor * t, cannge_es::EsTensorHolder & h) {
    if (t->op != GGML_OP_CONT || t->src[0] == nullptr || t->src[0]->view_src == nullptr) {
        return ggml_cannge_build_result::not_applicable;
    }
    ggml_tensor * v = t->src[0];
    ggml_cannge_view_info info;
    if (!ggml_cannge_resolve_view(v, info)) {
        ctx.fail("invalid view chain for CONT " + std::string(t->name));
        return ggml_cannge_build_result::failure;
    }
    // Lookup uses a registered holder first, then reconstructs a view from its registered base.
    const auto source = ctx.try_get(v, h);
    if (source != ggml_cannge_build_result::success) {
        return source;
    }
    return ggml_cannge_splice_fresh_row(ctx, t, v, h) ? ggml_cannge_build_result::success : ggml_cannge_build_result::failure;
}

ggml_cannge_build_result ggml_cannge_build_cont_chain(ggml_cannge_build_context & ctx, ggml_tensor * t,
                                                     cannge_es::EsTensorHolder & holder) {
    const std::string previous_error = ctx.err;
    ctx.err.clear();
    cannge_es::EsTensorHolder candidate;
    const auto result = ggml_cannge_build_cont_chain_impl(ctx, t, candidate);
    if (result == ggml_cannge_build_result::success) {
        holder = candidate;
    }
    if (result != ggml_cannge_build_result::failure) {
        ctx.err = previous_error;
    } else if (ctx.err.empty()) {
        ctx.fail("CONT chain construction failed for " + std::string(t->name));
    }
    return result;
}

// replace the slot written by the sibling SET_ROWS with the in-fragment row
// (the cache input still holds the pre-scatter bytes). cont ggml {S,D,G} ->
// GE {G,D,S}, slot = last GE dim.
static bool ggml_cannge_splice_fresh_row(ggml_cannge_build_context & ctx, ggml_tensor * t, ggml_tensor * v,
                                         cannge_es::EsTensorHolder & h) {
    ggml_cannge_view_info vinfo;
    if (!ggml_cannge_resolve_view(v, vinfo)) {
        return true;
    }
    const ggml_cannge_set_rows_op * sr =
        !ctx.io.options.disable_splice ? ggml_cannge_sr_for_base(ctx, vinfo.base) : nullptr;
    if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
        fprintf(stderr, "[DBG-SPLICE] cont-side v=%s base=%s sr=%s\n", v->name, vinfo.base->name,
                sr ? "found" : "none");
    }
    if (sr == nullptr) {
        return true;
    }
    cannge_es::EsTensorHolder rowh, idxh;
    const auto row = ctx.try_get(sr->data, rowh);
    if (row == ggml_cannge_build_result::failure) {
        return false;
    }
    if (row == ggml_cannge_build_result::not_applicable) {
        return true;
    }
    const auto index = ctx.try_get(sr->idx, idxh);
    if (index == ggml_cannge_build_result::failure) {
        return false;
    }
    if (index == ggml_cannge_build_result::not_applicable) {
        return true;
    }
    ge::DataType f32, dt;
    if (!ggml_cannge_ge_dtype(GGML_TYPE_F32, f32) || !ggml_cannge_ge_dtype(t->type, dt)) {
        return ctx.fail("unsupported fresh-row splice dtype for " + std::string(t->name));
    }
    const int64_t S = t->ne[0];
    const int64_t M = sr->idx->ne[0];
    auto h32        = cannge_es::Cast(h, (int64_t) f32);
    if (M == 1) {
        // decode: single-row one-hot select
        cannge_es::EsTensorHolder m;
        if (!ggml_cannge_onehot_slot(ctx, idxh, S, m, sr->idx)) {
            return false;
        }
        auto m3  = cannge_es::Reshape(m, std::vector<int64_t>{ 1, 1, S });
        auto r3  = cannge_es::Reshape(rowh, std::vector<int64_t>{ t->ne[2], t->ne[1], 1 });
        auto r32 = cannge_es::Cast(r3, (int64_t) f32);
        auto one = ctx.builder.CreateConst(std::vector<float>(S, 1.0f), std::vector<int64_t>{ 1, 1, S });
        h        = cannge_es::Cast(
            cannge_es::Add(cannge_es::Mul(h32, cannge_es::Sub(one, m3)), cannge_es::Mul(r32, m3)), (int64_t) dt);
        return true;
    }
    // prefill (M > 1 rows): sequential per-row one-hot selects, reusing the
    // M==1-proven helper. Costs ~4 passes over the view per row - acceptable
    // for prefill (decode keeps the single-select path above).
    auto hacc = h32;
    for (int64_t i = 0; i < M; i++) {
        auto idx1 = cannge_es::Slice(idxh, std::vector<int64_t>{ i }, std::vector<int64_t>{ 1 });
        ggml_tensor * idx1t = sr->idx; // dtype from the original idx tensor
        cannge_es::EsTensorHolder m;
        if (!ggml_cannge_onehot_slot(ctx, idx1, S, m, idx1t)) {
            return false;
        }
        auto m3  = cannge_es::Reshape(m, std::vector<int64_t>{ 1, 1, S });
        auto ri  = cannge_es::Slice(rowh, std::vector<int64_t>{ i, 0 }, std::vector<int64_t>{ 1, t->ne[1] * t->ne[2] });
        auto r3  = cannge_es::Reshape(ri, std::vector<int64_t>{ t->ne[2], t->ne[1], 1 });
        auto r32 = cannge_es::Cast(r3, (int64_t) f32);
        auto one = ctx.builder.CreateConst(std::vector<float>(S, 1.0f), std::vector<int64_t>{ 1, 1, S });
        hacc     = cannge_es::Add(cannge_es::Mul(hacc, cannge_es::Sub(one, m3)), cannge_es::Mul(r32, m3));
    }
    h = cannge_es::Cast(hacc, (int64_t) dt);
    return true;
}
