#include "graph-build-priv.h"

#include <utility>

bool ggml_cannge_build_view_prefix(ggml_cannge_build_context & ctx, const ggml_tensor * tensor,
                                   const ggml_tensor * base, cannge_es::EsTensorHolder & holder) {
    int64_t outer;
    if (!ggml_cannge_view_prefix_extent(tensor, base, outer)) {
        return ctx.fail("invalid or unaligned view prefix " + std::string(tensor->name));
    }
    const std::vector<int64_t> shape = ggml_cannge_ge_shape(base);
    if (outer == shape[0]) {
        return true;
    }
    std::vector<int64_t> offsets(shape.size(), 0);
    std::vector<int64_t> sizes = shape;
    sizes[0] = outer;
    holder = cannge_es::Slice(holder, offsets, sizes);
    return true;
}

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
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE: {
            if (step->src[0] == nullptr) {
                err = "axis transform without src0 " + std::string(step->name);
                return false;
            }
            const ggml_tensor * src = step->src[0];
            std::vector<int64_t> shape(GGML_MAX_DIMS);
            std::vector<int64_t> order(GGML_MAX_DIMS);
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                shape[GGML_MAX_DIMS - 1 - i] = src->ne[i];
                order[i] = i;
            }
            if (step->op == GGML_OP_PERMUTE) {
                bool seen[GGML_MAX_DIMS] = {};
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const int axis = step->op_params[i];
                    if (axis < 0 || axis >= GGML_MAX_DIMS || seen[axis]) {
                        err = "invalid permutation " + std::string(step->name);
                        return false;
                    }
                    seen[axis] = true;
                    order[GGML_MAX_DIMS - 1 - axis] = GGML_MAX_DIMS - 1 - i;
                }
            } else {
                std::swap(order[GGML_MAX_DIMS - 1], order[GGML_MAX_DIMS - 2]);
            }
            // Retain singleton axes while permuting, then restore the output rank.
            h = cannge_es::Reshape(h, shape);
            h = cannge_es::Permute(h, order);
            h = cannge_es::Reshape(h, ggml_cannge_ge_shape(step));
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

// Resolve the highest registered base for GGML_OP_VIEW.
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

bool ggml_cannge_build_context::get(ggml_tensor * t, cannge_es::EsTensorHolder & holder) {
    if (t == nullptr) {
        return fail("missing source tensor");
    }
    auto it = tensors.find(t);
    if (it == tensors.end()) {
        // CONT of a rearrange view (merged-attention V path): relabel
        // chain + fresh-row splice (note-50 batch K)
        if (t->op == GGML_OP_CONT) {
            const auto result = ggml_cannge_build_cont_chain(*this, t, holder);
            if (result == ggml_cannge_build_result::success) {
                tensors[t] = holder;
                return true;
            }
            if (result == ggml_cannge_build_result::failure) {
                return false;
            }
        }
        // rearrange-view input substituted by its base (plan.cpp): rebuild
        // the relabel chain in-GE from the registered base holder
        if (t->view_src != nullptr) {
            ggml_cannge_view_info info;
            if (!ggml_cannge_resolve_view(t, info)) {
                return fail("invalid view chain for " + std::string(t->name));
            }
            std::vector<ggml_tensor *> chain;
            ggml_tensor *              base = t;
            while (base->view_src != nullptr) {
                chain.push_back(base);
                base = base->view_src;
            }
            auto bit = tensors.find(base);
            if (bit != tensors.end()) {
                cannge_es::EsTensorHolder h = bit->second;
                if (!ggml_cannge_build_view_prefix(*this, t, base, h)) {
                    return false;
                }
                for (auto rit = chain.rbegin(); rit != chain.rend(); ++rit) {
                    if (!ggml_cannge_build_relabel(*rit, h, err)) {
                        return false;
                    }
                }
                tensors[t] = h;
                holder     = h;
                return true;
            }
        }
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