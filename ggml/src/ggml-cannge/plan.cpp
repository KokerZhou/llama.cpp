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
#include "ggml-cannge-priv.h"

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
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

// every step of the view_src chain only relabels bytes (no element transform):
// such a view of a fresh input buffer already holds the correct content for
// consumers reading through the view strides, so GE needs no output port
static bool ggml_cannge_is_rearrange_view(const ggml_tensor * t) {
    for (const ggml_tensor * cur = t; cur != nullptr && cur->view_src != nullptr; cur = cur->view_src) {
        if (cur->op != GGML_OP_RESHAPE && cur->op != GGML_OP_PERMUTE && cur->op != GGML_OP_TRANSPOSE &&
            cur->op != GGML_OP_VIEW && cur->op != GGML_OP_NONE) {
            return false;
        }
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

    // collect all fits and order them deterministically: tightest span
    // first, then rank penalty, name, address as the last resort. A pointer-
    // ordered tie used to pick an arbitrary base when allocator reuse made
    // several candidates contain the view (rogue shapes, note-46)
    std::vector<const ggml_tensor *> fits_vec;
    auto collect = [&](bool name_match_only) {
        fits_vec.clear();
        for (ggml_tensor * cand : registered) {
            if (name_match_only && (node->view_src == nullptr || strcmp(cand->name, node->view_src->name) != 0)) {
                continue;
            }
            if (fits(cand)) {
                fits_vec.push_back(cand);
            }
        }
        std::sort(fits_vec.begin(), fits_vec.end(), [&](const ggml_tensor * a, const ggml_tensor * b) {
            const size_t sa = ggml_nbytes(a);
            const size_t sb = ggml_nbytes(b);
            if (sa != sb) return sa < sb;
            const int pa = rank_of(a) == node_nd ? 0 : 1;
            const int pb = rank_of(b) == node_nd ? 0 : 1;
            if (pa != pb) return pa < pb;
            const int c = strcmp(a->name, b->name);
            if (c != 0) return c < 0;
            return a < b;
        });
    };

    const ggml_tensor * best = nullptr;
    // Tier 1: registered tensor whose name matches the immediate view_src
    if (node->view_src != nullptr) {
        collect(true);
        if (!fits_vec.empty()) {
            best = fits_vec.front();
        }
    }

    // Tier 2: any registered tensor whose buffer contains node->data
    if (best == nullptr) {
        collect(false);
        if (!fits_vec.empty()) {
            if (fits_vec.size() > 1 && getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                fprintf(stderr, "[DBG-VIEW] view %s: %zu candidate bases, picking %s (span %zu)\n", node->name,
                        fits_vec.size(), fits_vec.front()->name, ggml_nbytes(fits_vec.front()));
            }
            best = fits_vec.front();
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
            fprintf(stderr, "[DBG-ANALYZE]   node[%d] %s op=%s ne=%lld,%lld,%lld,%lld", i, cgraph->nodes[i]->name,
                    ggml_op_name(cgraph->nodes[i]->op), (long long) cgraph->nodes[i]->ne[0],
                    (long long) cgraph->nodes[i]->ne[1], (long long) cgraph->nodes[i]->ne[2],
                    (long long) cgraph->nodes[i]->ne[3]);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * s = cgraph->nodes[i]->src[j];
                if (s != nullptr) {
                    fprintf(stderr, " src%d=%s(%s,%lld)", j, s->name, ggml_op_name(s->op), (long long) s->ne[0]);
                }
            }
            fprintf(stderr, "\n");
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

    // in-graph ScatterUpdate decision (note-48 batch G): KV rows written
    // inside the GE graph, so in-fragment view chains read the fresh row and
    // the post-execute host scatter is skipped. Opt out with
    // GGML_CANNGE_NO_SCATTER=1. I64 indices stay host-side (no GE dtype).
    std::unordered_map<ggml_tensor *, bool> sr_in_graph;
    // GE ScatterUpdate/TensorScatterUpdate silently no-op on this stack (probe-
    // verified, note-49): the in-graph write stays opt-in until a working primitive exists
    const bool scatter_enabled = getenv("GGML_CANNGE_SCATTER") != nullptr;
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op != GGML_OP_SET_ROWS) {
            continue;
        }
        const ggml_tensor * data = node->src[0];
        const ggml_tensor * idx  = node->src[1];
        const ggml_tensor * dst  = node->src[2];
        sr_in_graph[node] = scatter_enabled && data != nullptr && idx != nullptr && dst != nullptr &&
                            ggml_cannge_float_dtype(data->type) && ggml_cannge_float_dtype(dst->type) &&
                            (idx->type == GGML_TYPE_I32 || idx->type == GGML_TYPE_I64) && ggml_nelements(idx) == 1;
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
            // SET_ROWS src[1] (idx) is always registered: the fresh-row
            // splice reads it in-GE and the host scatter reads it post-execute
            // (it stays tiny; keeping I64 out of the GE dtype check no longer
            // applies since the splice casts it)
            // IFA decode carries kv validity in actual_seq_lengths derived
            // from the additive mask at execute time (note-42)
            if (node->op == GGML_OP_FLASH_ATTN_EXT && j == 3 && node->src[0] != nullptr &&
                node->src[0]->ne[1] == 1 && src != nullptr) {
                io.fa_mask = src;
            }
            if (!ggml_is_empty(src) && !produced.count(src) && !seen_input.count(src)) {
                // non-dense rearrange view of a dense base (PERMUTE/TRANSPOSE
                // of the KV cache): register the BASE as the input and let the
                // build reconstruct the relabel chain in-GE. Binds the dense
                // buffer directly and skips the strided staging bounce with
                // its host round-trip and stream drain (note-48).
                if (src->view_src != nullptr) {
                    ggml_cannge_view_info vinfo;
                    if (ggml_cannge_resolve_view(src, vinfo) && !vinfo.is_dense && vinfo.offset == 0 &&
                        ggml_cannge_is_rearrange_view(src) && ggml_is_contiguous(vinfo.base)) {
                        // the build reconstructs the relabel in-GE with a
                        // linear prefix slice on the base: the view span must
                        // align to the base's inner extent (KV cache views
                        // span a slot prefix of the context buffer)
                        const int64_t  b_nd = ggml_n_dims(vinfo.base);
                        const int64_t  t_el = ggml_nelements(src);
                        const int64_t  b_el = ggml_nelements(vinfo.base);
                        int64_t        inner = 1;
                        for (int64_t k = 0; k + 1 < b_nd; k++) {
                            inner *= vinfo.base->ne[k];
                        }
                        if (t_el <= b_el && (b_nd < 2 || (inner > 0 && t_el % inner == 0))) {
                            seen_input[src] = true;
                            src = vinfo.base;
                            if (seen_input.count(src)) {
                                continue;
                            }
                        }
                    }
                }
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
        const char * force_out = getenv("GGML_CANNGE_DUMP");
        if (n_consumers[node] == 0 || (node->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            (force_out != nullptr && strstr(node->name, force_out) != nullptr)) {
            // a DENSE view that aliases an external input needs no GE output
            // port: consumers read the input buffer through the view strides,
            // the data is already there (the scheduler avoids splitting at
            // views, so this stays an alias-only case). A non-dense view
            // (PERMUTE/TRANSPOSE strides) must be materialized by GE, its
            // bytes are not present at data in dense order - EXCEPT when the
            // whole chain is a pure rearrangement of a plan input: the view
            // buffer then already presents the rearranged bytes to consumers
            // (the KV cache transpose; the writeback would rewrite the same
            // bytes, and its host-round-trip copy-back forces a stream drain
            // per layer per step)
            if (node->view_src != nullptr) {
                ggml_cannge_view_info info;
                if (ggml_cannge_resolve_view(node, info) && is_input.count(info.base) &&
                    (info.is_dense || ggml_cannge_is_rearrange_view(node))) {
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
            // same staleness class as SET_ROWS below: the post-execute
            // writeback reads the ggml buffer, so an in-graph producer needs
            // its value materialized there
            if (produced.count(node->src[0])) {
                bool already = false;
                for (ggml_tensor * o : io.outputs) {
                    if (o == node->src[0]) { already = true; break; }
                }
                if (!already) {
                    io.outputs.push_back(node->src[0]);
                }
            }
        }
        if (node->op == GGML_OP_SET_ROWS && node->src[0] != nullptr && node->src[1] != nullptr &&
            node->src[2] != nullptr) {
            io.set_rows.push_back({ node->src[2], node->src[0], node->src[1], sr_in_graph[node] });
            // the scatter runs post-execute and reads the data tensor's ggml
            // buffer. When the data is produced inside this graph (e.g. rope
            // on CANNGE feeding the KV write) GE keeps it in its own workspace
            // and never writes the ggml address - the scatter would copy stale
            // bytes into the cache (rope-on KV-on corruption, note-40). Force
            // a GE output port so the value is materialized at t->data.
            // In-graph scatters read the GE tensor directly: no port needed.
            if (!sr_in_graph[node] && produced.count(node->src[0])) {
                bool already = false;
                for (ggml_tensor * o : io.outputs) {
                    if (o == node->src[0]) { already = true; break; }
                }
                if (!already) {
                    io.outputs.push_back(node->src[0]);
                }
            }
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
