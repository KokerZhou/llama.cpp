// CANNGE graph build helpers: input Data ports, view/relabel chains.
// Split out of graph-build.cpp (note-48 batch G).

#include "graph-build-priv.h"

bool ggml_cannge_build_relabel(ggml_tensor * step, cannge_es::EsTensorHolder & h, std::string & err) {
    if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
        fprintf(stderr,
                "[DBG-RELABEL] step %s op=%s type=%s src0=%s ne=%lld,%lld,%lld,%lld nb=%lld,%lld,%lld,%lld "
                "src0_ne=%lld,%lld,%lld,%lld ge_shape(step)=%zu dims\n",
                step->name, ggml_op_name(step->op), ggml_type_name(step->type),
                step->src[0] ? step->src[0]->name : "?", (long long) step->ne[0], (long long) step->ne[1],
                (long long) step->ne[2], (long long) step->ne[3], (long long) step->nb[0], (long long) step->nb[1],
                (long long) step->nb[2], (long long) step->nb[3],
                step->src[0] ? (long long) step->src[0]->ne[0] : -1, step->src[0] ? (long long) step->src[0]->ne[1] : -1,
                step->src[0] ? (long long) step->src[0]->ne[2] : -1, step->src[0] ? (long long) step->src[0]->ne[3] : -1,
                ggml_cannge_ge_shape(step).size());
    }
    switch (step->op) {
        case GGML_OP_PERMUTE: {
            // ggml flattens view chains, so the holder may carry the BASE
            // shape while the permute expects its (virtual) src0 shape: relabel
            // to src0's GE shape first (element count already matched by the
            // caller's prefix slice), then apply the dst->src inverted order
            if (step->src[0] == nullptr) {
                err = "permute step without src0 " + std::string(step->name);
                return false;
            }
            h           = cannge_es::Reshape(h, ggml_cannge_ge_shape(step->src[0]));
            const int nd = ggml_n_dims(step->src[0]);
            int32_t   inv_p[GGML_MAX_DIMS] = { 0, 1, 2, 3 };
            for (int i = 0; i < nd; i++) {
                inv_p[step->op_params[i]] = i;
            }
            std::vector<int64_t> order(nd);
            for (int k = 0; k < nd; k++) {
                order[k] = nd - 1 - inv_p[nd - 1 - k];
            }
            h = cannge_es::Permute(h, order);
            return true;
        }
        case GGML_OP_TRANSPOSE: {
            if (step->src[0] == nullptr) {
                err = "transpose step without src0 " + std::string(step->name);
                return false;
            }
            h            = cannge_es::Reshape(h, ggml_cannge_ge_shape(step->src[0]));
            const int nd = ggml_n_dims(step->src[0]);
            std::vector<int64_t> perm(nd);
            for (int i = 0; i < nd; i++) {
                perm[i] = i;
            }
            perm[nd - 1] = nd - 2;
            perm[nd - 2] = nd - 1;
            h = cannge_es::Transpose(h, perm);
            return true;
        }
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_NONE:
            if (step->view_offs != 0 || !ggml_is_contiguous(step)) {
                err = "non-identity relabel step " + std::string(step->name);
                return false;
            }
            h = cannge_es::Reshape(h, ggml_cannge_ge_shape(step));
            return true;
        default:
            err = "non-relabel view step " + std::string(step->name);
            return false;
    }
}

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

// GGML_OP_VIEW: resolve the highest registered base with the shared helper,

bool ggml_cannge_build_view(ggml_cannge_build_context & ctx, ggml_tensor * node) {
    std::set<ggml_tensor *> registered;
    for (const auto & kv : ctx.tensors) {
        registered.insert(kv.first);
    }

    size_t        offset = 0;
    ggml_tensor * base   = ggml_cannge_resolve_view_base(node, registered, offset);
    if (base == nullptr || !registered.count(base)) {
        return ctx.fail("no registered base for view " + std::string(node->name));
    }

    if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
        fprintf(stderr, "[DBG-COMPILE]   view %s: resolved base=%s offset=%zu\n", node->name, base->name, offset);
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
    bool       dense = dense_rel;
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

// mirror of plan.cpp's static predicate (not exported there)
static bool ggml_cannge_is_rearrange_view_local(const ggml_tensor * t) {
    for (const ggml_tensor * cur = t; cur != nullptr && cur->view_src != nullptr; cur = cur->view_src) {
        if (cur->op != GGML_OP_RESHAPE && cur->op != GGML_OP_PERMUTE && cur->op != GGML_OP_TRANSPOSE &&
            cur->op != GGML_OP_VIEW && cur->op != GGML_OP_NONE) {
            return false;
        }
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

bool ggml_cannge_build_cont_chain(ggml_cannge_build_context & ctx, ggml_tensor * t, cannge_es::EsTensorHolder & h) {
    if (t->op != GGML_OP_CONT || t->src[0] == nullptr || t->src[0]->view_src == nullptr) {
        return false;
    }
    ggml_tensor * v = t->src[0];
    // preferred: the source was built as a cgraph node in this fragment (its
    // PERMUTE/TRANSPOSE cases carry the correct byte semantics); the view_src
    // walk below is only the cross-fragment fallback, and ggml flattens view
    // chains so the walk cannot be trusted to replay the ops
    if (ctx.get(v, h)) {
        return ggml_cannge_splice_fresh_row(ctx, t, v, h);
    }
    if (!ggml_cannge_is_rearrange_view_local(v)) {
        return false;
    }
    std::vector<ggml_tensor *> chain;
    ggml_tensor *              base = v;
    while (base->view_src != nullptr) {
        chain.push_back(base);
        base = base->view_src;
    }
    auto bit = ctx.tensors.find(base);
    if (bit == ctx.tensors.end()) {
        return false;
    }
    h = bit->second;
    // prefix slice when the view covers fewer elements than the base
    const int64_t t_elems = ggml_nelements(t);
    const int64_t b_elems = ggml_nelements(base);
    if (t_elems < b_elems) {
        std::vector<int64_t> ge = ggml_cannge_ge_shape(base);
        int64_t              inner = 1;
        for (size_t k = 1; k < ge.size(); k++) {
            inner *= ge[k];
        }
        if (inner <= 0 || t_elems % inner != 0) {
            return false;
        }
        std::vector<int64_t> off(ge.size(), 0);
        std::vector<int64_t> sz = ge;
        sz[0]                   = t_elems / inner;
        h                       = cannge_es::Slice(h, off, sz);
    }
    for (auto rit = chain.rbegin(); rit != chain.rend(); ++rit) {
        if (!ggml_cannge_build_relabel(*rit, h, ctx.err)) {
            return false;
        }
    }
    return ggml_cannge_splice_fresh_row(ctx, t, v, h);
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
        getenv("GGML_CANNGE_NO_SPLICE") == nullptr ? ggml_cannge_sr_for_base(ctx, vinfo.base) : nullptr;
    if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
        fprintf(stderr, "[DBG-SPLICE] cont-side v=%s base=%s sr=%s\n", v->name, vinfo.base->name,
                sr ? "found" : "none");
    }
    if (sr == nullptr) {
        return true;
    }
    cannge_es::EsTensorHolder rowh, idxh;
    if (!ctx.get(sr->data, rowh) || !ctx.get(sr->idx, idxh)) {
        if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
            fprintf(stderr, "[DBG-SPLICE] cont-side row/idx holder miss data=%s idx=%s\n",
                    sr->data->name, sr->idx->name);
        }
        return true; // row not produced in this fragment: plain (stale-slot)
                     // chain only happens when no sibling exists at all; with a
                     // sibling but no in-fragment row this consumer reads the
                     // cache input, matching the pre-merge semantics
    }
    ge::DataType f32, dt;
    if (!ggml_cannge_ge_dtype(GGML_TYPE_F32, f32) || !ggml_cannge_ge_dtype(t->type, dt)) {
        return false;
    }
    const int64_t S = t->ne[0];
    cannge_es::EsTensorHolder m;
    if (!ggml_cannge_onehot_slot(ctx, idxh, S, m, sr->idx)) {
        return false;
    }
    auto m3 = cannge_es::Reshape(m, std::vector<int64_t>{ 1, 1, S });
    // row GE: sr.data is the flat {D*G} cache-row view; the cont t is
    // ggml {S,D,G} -> GE {G,D,S}, so the row splices as {G,D,1} (byte-identity)
    auto r3 = cannge_es::Reshape(rowh, std::vector<int64_t>{ t->ne[2], t->ne[1], 1 });
    auto h32 = cannge_es::Cast(h, (int64_t) f32);
    auto r32 = cannge_es::Cast(r3, (int64_t) f32);
    auto one = ctx.builder.CreateConst(std::vector<float>(S, 1.0f), std::vector<int64_t>{ 1, 1, S });
    h        = cannge_es::Cast(cannge_es::Add(cannge_es::Mul(h32, cannge_es::Sub(one, m3)), cannge_es::Mul(r32, m3)),
                               (int64_t) dt);
    return true;
}
