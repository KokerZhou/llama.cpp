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

#include "plan.h"

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cstring>
#include <set>
#include <unordered_map>

// walk the view_src chain down to the root tensor
bool ggml_cannge_resolve_view(ggml_tensor * t, ggml_cannge_view_info & info) {
    info.offset = 0;

    ggml_tensor * base = t;
    while (ggml_is_view(base)) {
        info.offset += base->view_offs;
        base = base->view_src;
        if (base == nullptr) {
            return false;
        }
    }
    info.base = base;

    // nb of the view itself is already expressed relative to its data pointer
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        info.nb[i] = t->nb[i];
    }

    info.is_dense = info.nb[0] == ggml_element_size(t);
    for (int i = 1; i < GGML_MAX_DIMS && info.is_dense; i++) {
        info.is_dense = info.nb[i] == info.nb[i - 1] * (size_t) t->ne[i - 1];
    }
    return true;
}

// resolve a view to a registered base tensor, mirroring build_view's chain walk
// and tiered candidate search
ggml_tensor * ggml_cannge_resolve_view_base(ggml_tensor * node, const std::set<ggml_tensor *> & registered, size_t & offset) {
    offset = 0;
    if (node == nullptr || !ggml_is_view(node)) {
        return nullptr;
    }

    // chain walk: follow view_src while it is registered (has_holder)
    ggml_tensor * base = node;
    std::set<ggml_tensor *> visited;
    visited.insert(base);
    while (ggml_is_view(base) && base->view_src != nullptr) {
        if (registered.find(base->view_src) == registered.end() || visited.count(base->view_src)) {
            break;
        }
        offset += base->view_offs;
        base    = base->view_src;
        visited.insert(base);
    }
    if (registered.count(base)) {
        return base;
    }

    if (node->data == nullptr) {
        return nullptr;
    }

    const int64_t esize   = ggml_element_size(node);
    const int     node_nd = ggml_n_dims(node);
    const size_t  node_span = (size_t) node->ne[0] * node->ne[1] * node->ne[2] * node->ne[3] * esize;

    auto fits = [&](const ggml_tensor * cand) -> bool {
        const char * lo   = (const char *) cand->data;
        const size_t span = ggml_nbytes(cand);
        if ((const char *) node->data < lo || (const char *) node->data >= lo + span) {
            return false;
        }
        // whole-buffer reinterpretation: same address, same byte span
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
            const int64_t off     = cand_ne > 0 ? rem % cand_ne : 0;
            rem                   = cand_ne > 0 ? rem / cand_ne : 0;
            if (d < node_nd && off + node->ne[d] > cand_ne) {
                return false;
            }
        }
        for (int d = cand_nd; d < node_nd; d++) {
            if (node->ne[d] != 1) {
                return false;
            }
        }
        return rem == 0;
    };

    auto rank_of = [](const ggml_tensor * t) { return ggml_n_dims(t); };

    const ggml_tensor * best = nullptr;
    size_t              best_span = SIZE_MAX;
    int                 best_rank_pen = 2;

    // Tier 1: registered tensor whose name matches the immediate view_src
    if (node->view_src != nullptr) {
        for (ggml_tensor * cand : registered) {
            if (strcmp(cand->name, node->view_src->name) != 0 || !fits(cand)) {
                continue;
            }
            const size_t span = ggml_nbytes(cand);
            const int    pen  = rank_of(cand) == node_nd ? 0 : 1;
            if (pen < best_rank_pen || (pen == best_rank_pen && span < best_span)) {
                best          = cand;
                best_span     = span;
                best_rank_pen = pen;
            }
        }
    }

    // Tier 2: any registered tensor whose buffer contains node->data
    if (best == nullptr) {
        best_rank_pen = 2;
        best_span     = SIZE_MAX;
        for (ggml_tensor * cand : registered) {
            if (!fits(cand)) {
                continue;
            }
            const size_t span = ggml_nbytes(cand);
            const int    pen  = rank_of(cand) == node_nd ? 0 : 1;
            if (pen < best_rank_pen || (pen == best_rank_pen && span < best_span)) {
                best          = cand;
                best_span     = span;
                best_rank_pen = pen;
            }
        }
    }

    if (best != nullptr) {
        base   = (ggml_tensor *) best;
        offset = (size_t) ((const char *) node->data - (const char *) best->data);
        return base;
    }

    return nullptr;
}

static bool ggml_cannge_plan_record_io(ggml_cannge_plan_io & io, ggml_tensor * t, std::string & err) {
    ggml_cannge_view_info info;
    if (!ggml_cannge_resolve_view(t, info)) {
        err = "broken view chain";
        return false;
    }
    if (!info.is_dense) {
        // TODO(needs_staging): staged copies are added by a later phase
        io.needs_staging = true;
    }
    io.io_views.emplace_back(t, info);
    return true;
}

bool ggml_cannge_plan_analyze(ggml_cgraph * cgraph, ggml_cannge_plan_io & io, std::string & err) {
    const int n_nodes = cgraph->n_nodes;

    if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
        fprintf(stderr, "[DBG-ANALYZE] graph n_nodes=%d\n", n_nodes);
        for (int i = 0; i < n_nodes; i++) {
            fprintf(stderr, "[DBG-ANALYZE]   node[%d] %s op=%s\n", i, cgraph->nodes[i]->name,
                    ggml_op_name(cgraph->nodes[i]->op));
        }
    }

    // producers: every node produces its own output tensor; a CPY node also
    // produces its write destination (src[1], which it views): registering
    // the dst as an input would bind it read-only and the cast output port
    // would alias an input buffer, which GE drops on write (910B silent)
    std::unordered_map<ggml_tensor *, bool> produced;
    std::unordered_map<ggml_tensor *, int>  n_consumers;
    for (int i = 0; i < n_nodes; i++) {
        produced[cgraph->nodes[i]] = true;
        if (cgraph->nodes[i]->op == GGML_OP_CPY && cgraph->nodes[i]->src[1] != nullptr) {
            produced[cgraph->nodes[i]->src[1]] = true;
        }
    }
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = node->src[j];
            if (src != nullptr) {
                n_consumers[src]++;
            }
        }
    }

    // external inputs: referenced as src but never produced inside the graph
    // (leafs with op GGML_OP_NONE fall in here naturally), in first reference
    // order; empty tensors are skipped like the graph_compute pre-check
    std::unordered_map<ggml_tensor *, bool> seen_input;
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node)) {
            continue;
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = node->src[j];
            if (src == nullptr) {
                continue;
            }
            // CPY src[1] is the write destination (the node views it), not a
            // read input: registering it as an input would turn the dst into
            // an external read-only buffer and the cast result would get no
            // GE output port. SET_ROWS src[1] is the index tensor, consumed by
            // the backend scatter after execute, never by GE (also keeps I64
            // indices away from the GE input dtype check)
            if (node->op == GGML_OP_CPY && j == 1) {
                continue;
            }
            if (node->op == GGML_OP_SET_ROWS && j == 1) {
                continue;
            }
            if (!ggml_is_empty(src) && !produced.count(src) && !seen_input.count(src)) {
                seen_input[src] = true;
                io.inputs.push_back(src);
            }
        }
    }

    std::unordered_map<ggml_tensor *, bool> is_input;
    for (ggml_tensor * t : io.inputs) {
        is_input[t] = true;
    }

    // boundary outputs: no consumer, or explicitly flagged
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node)) {
            continue;
        }
        // sentinel leafs (op NONE, added by test-backend-ops for overflow checks)
        // have no producing op and no GE port: their buffer is harness-owned and
        // must stay untouched, registering them as outputs breaks the build
        if (node->op == GGML_OP_NONE) {
            continue;
        }
        if (n_consumers[node] == 0 || (node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            // a DENSE view that aliases an external input needs no GE output
            // port: consumers read the input buffer through the view strides,
            // the data is already there (the scheduler avoids splitting at
            // views, so this stays an alias-only case). A non-dense view
            // (PERMUTE/TRANSPOSE strides) must be materialized by GE, its
            // bytes are not present at data in dense order
            if (node->view_src != nullptr) {
                ggml_cannge_view_info info;
                if (ggml_cannge_resolve_view(node, info) && info.is_dense && is_input.count(info.base)) {
                    io.io_views.emplace_back(node, info);
                    continue;
                }
            }
            io.outputs.push_back(node);
        }
    }

    // side effects: GGML_OP_SET writes src[0] in place, keep it observable;
    // GGML_OP_SET_ROWS scatters rows into the KV cache view after execute
    std::unordered_map<ggml_tensor *, bool> seen_side_effect;
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node)) {
            continue;
        }
        if (node->op == GGML_OP_SET && node->src[0] != nullptr && !seen_side_effect.count(node->src[0])) {
            seen_side_effect[node->src[0]] = true;
            io.side_effects.push_back(node->src[0]);
        }
        if (node->op == GGML_OP_SET_ROWS && node->src[0] != nullptr && node->src[1] != nullptr &&
            node->src[2] != nullptr) {
            io.set_rows.push_back({ node->src[2], node->src[0], node->src[1] });
        }
    }

    // GE output ports, in the exact registration order of the build step:
    // same-dtype CPY/DUP with a distinct destination buffer first (node order,
    // the build aliases the node to its source and forces a port so GE writes
    // the destination), then boundary outputs, deduplicated
    std::unordered_map<ggml_tensor *, bool> seen_output;
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->src[0] == nullptr) {
            continue;
        }
        if (node->op != GGML_OP_CPY && node->op != GGML_OP_DUP) {
            continue;
        }
        if (node->type == node->src[0]->type && node->data != node->src[0]->data) {
            seen_output[node] = true;
            io.output_tensors.push_back(node);
        }
    }
    for (ggml_tensor * t : io.outputs) {
        if (!seen_output.count(t)) {
            seen_output[t] = true;
            io.output_tensors.push_back(t);
        }
    }

    io.input_staged.assign(io.inputs.size(), false);
    io.input_base.assign(io.inputs.size(), nullptr);
    io.input_base_offset.assign(io.inputs.size(), 0);
    std::set<ggml_tensor *> registered;
    if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
        for (size_t i = 0; i < io.inputs.size(); i++) {
            fprintf(stderr, "[DBG-ANALYZE]   input[%zu] %s op=%s\n", i, io.inputs[i]->name,
                    ggml_op_name(io.inputs[i]->op));
        }
    }
    for (size_t i = 0; i < io.inputs.size(); i++) {
        ggml_tensor * t = io.inputs[i];
        ggml_cannge_view_info info;
        if (!ggml_cannge_resolve_view(t, info)) {
            err = "broken view chain";
            return false;
        }
        if (!info.is_dense) {
            // strided input (e.g. a PERMUTE consumed across the graph boundary):
            // staged into a dense buffer before execute, see ggml-cannge.cpp
            io.input_staged[i] = true;
        } else if (t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer)) {
            // dense but host-resident (CPU fallback outputs, rope positions):
            // GE inputs must be device pointers, stage it too
            io.input_staged[i] = true;
        }
        // resolve the registered base this view aliases; approximate the build's
        // registered set with inputs processed so far
        io.input_base[i] = ggml_cannge_resolve_view_base(t, registered, io.input_base_offset[i]);
        registered.insert(t);
        io.io_views.emplace_back(t, info);
    }
    io.output_strided.assign(io.outputs.size(), 0);
    for (size_t i = 0; i < io.outputs.size(); i++) {
        ggml_tensor * t = io.outputs[i];
        ggml_cannge_view_info info;
        if (!ggml_cannge_resolve_view(t, info)) {
            err = "broken view chain";
            return false;
        }
        if (!info.is_dense) {
            // strided boundary output: GE emits dense into staging, a post-
            // execute strided copy lands the data in the real view buffer
            io.output_strided[i] = 1;
        }
        io.io_views.emplace_back(t, info);
    }
    for (ggml_tensor * t : io.side_effects) {
        if (!ggml_cannge_plan_record_io(io, t, err)) {
            return false;
        }
    }

    // rough device working memory estimate: outputs plus external inputs
    for (int i = 0; i < n_nodes; i++) {
        io.mem_estimate += ggml_nbytes(cgraph->nodes[i]);
    }
    for (ggml_tensor * t : io.inputs) {
        io.mem_estimate += ggml_nbytes(t);
    }

    return true;
}

std::string ggml_cannge_plan_signature(ggml_cgraph * cgraph) {
    std::string sig;

    // tensor ids by first appearance: srcs of a node before the node itself
    std::unordered_map<ggml_tensor *, uint32_t> ids;
    auto tensor_id = [&ids](ggml_tensor * t) -> uint32_t {
        auto it = ids.find(t);
        if (it != ids.end()) {
            return it->second;
        }
        uint32_t id = (uint32_t) ids.size();
        ids[t] = id;
        return id;
    };
    auto append_i32 = [&sig](int32_t v) { sig.append((const char *) &v, sizeof(v)); };
    auto append_u32 = [&sig](uint32_t v) { sig.append((const char *) &v, sizeof(v)); };
    auto append_i64 = [&sig](int64_t v) { sig.append((const char *) &v, sizeof(v)); };
    auto append_u64 = [&sig](uint64_t v) { sig.append((const char *) &v, sizeof(v)); };
    auto append_meta = [&sig, append_i32, append_i64, append_u64](const ggml_tensor * t) {
        append_i32((int32_t) t->type);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            append_i64(t->ne[i]);
        }
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            append_u64(t->nb[i]);
        }
    };

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        append_i32((int32_t) node->op);
        append_i32(node->flags);
        append_meta(node);
        sig.append((const char *) node->op_params, sizeof(node->op_params));

        int n_srcs = 0;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                n_srcs++;
            }
        }
        append_i32(n_srcs);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = node->src[j];
            if (src == nullptr) {
                continue;
            }
            append_u32(tensor_id(src));
            append_meta(src);
        }
        tensor_id(node);
    }

    return sig;
}
