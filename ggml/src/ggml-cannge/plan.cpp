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

#include "supports.h"
#include "plan.h"
#include "common.h"

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cstring>
#include <set>
#include <unordered_map>

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

bool ggml_cannge_plan_analyze(ggml_cgraph * cgraph, ggml_cannge_plan_io & io, std::string & err,
                              const ggml_cannge_options & options) {
    io.options = options;
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
    const bool scatter_enabled = options.scatter;
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
                // CONT of a rearrange view (the V cache transpose): the dense
                // cont bytes already live in the base buffer - bind the base
                // (batch R: kills the 1MB/layer H2D staging in the split
                // config too; the in-GE relabel reads the post-scatter cache)
                if (src->op == GGML_OP_CONT && src->src[0] != nullptr && src->src[0]->view_src != nullptr) {
                    ggml_tensor *            v = src->src[0];
                    ggml_cannge_view_info    vinfo;
                    if (ggml_cannge_resolve_view(v, vinfo) && !vinfo.is_dense && vinfo.offset == 0 &&
                        ggml_cannge_is_rearrange_view(v) && ggml_is_contiguous(vinfo.base)) {
                        const int64_t b_nd = ggml_n_dims(vinfo.base);
                        const int64_t t_el = ggml_nelements(src);
                        const int64_t b_el = ggml_nelements(vinfo.base);
                        int64_t       inner = 1;
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

    // batch P (note-52): merged-attention cache refresh. With the attention
    // merged (CONT_NC), a SET_ROWS whose dst view's base is a device input
    // writes the fresh cache IN-GRAPH (one-hot select, M=1 or matrix form) -
    // the node itself becomes an output port (its ggml buffer IS the cache)
    // and the post-execute host scatter is skipped. This replaces the
    // Kcur/Vcur forced data ports that GE's runtime refresh mechanism
    // collapses into one buffer when many same-shaped outputs share a graph.
    if (options.contiguous_views) {
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (ggml_is_empty(node) || node->op != GGML_OP_SET_ROWS || node->src[2] == nullptr) {
                continue;
            }
            ggml_cannge_view_info dinfo;
            if (ggml_cannge_resolve_view(node->src[2], dinfo) && is_input.count(dinfo.base)) {
                sr_in_graph[node] = true;
                io.outputs.push_back(node);
            }
        }
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
        const char * force_out = options.dump_present ? options.dump_filter.c_str() : nullptr;
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

std::string ggml_cannge_plan_signature(ggml_cgraph * cgraph, const ggml_cannge_plan_io & io) {
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
        append_u64(t->view_offs);
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
            append_i32(j);
            append_u32(tensor_id(src));
            append_meta(src);
        }
        append_u32(tensor_id(node));
    }

    // Port order is part of the compiled graph's binding contract.
    // Walk-assigned ids keep the key stable when tensor addresses change.
    auto append_ports = [&](const std::vector<ggml_tensor *> & ports) {
        append_u64(ports.size());
        for (ggml_tensor * t : ports) {
            append_u32(tensor_id(t));
            append_meta(t);
        }
    };
    append_ports(io.inputs);
    append_ports(io.output_tensors);
    append_u64(io.set_rows.size());
    for (const ggml_cannge_set_rows_op & sr : io.set_rows) {
        append_u32(tensor_id(sr.dst));
        append_u32(tensor_id(sr.data));
        append_u32(tensor_id(sr.idx));
        append_i32(sr.in_graph);
    }

    sig.append(io.options.signature());

    return sig;
}
