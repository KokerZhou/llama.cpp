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

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cstring>
#include <set>
#include <unordered_map>

static bool ggml_cannge_plan_record_io(ggml_cannge_plan_io & io, ggml_tensor * t, std::string & err) {
    ggml_cannge_view_info info;
    if (!ggml_cannge_resolve_view(t, info)) {
        err = "invalid side-effect view chain: " + std::string(t->name);
        return false;
    }
    if (!info.is_dense) {
        // TODO(needs_staging): staged copies are added by a later phase
        io.needs_staging = true;
    }
    io.io_views.emplace_back(t, info);
    return true;
}

// TensorScatterUpdate writes rows into a matrix; batched indices use host scatter.
static bool ggml_cannge_can_scatter_in_graph(const ggml_tensor * node) {
    const ggml_tensor * data = node->src[0];
    const ggml_tensor * idx = node->src[1];
    ggml_tensor * dst = node->src[2];
    if (data == nullptr || idx == nullptr || dst == nullptr ||
        !ggml_cannge_float_dtype(data->type) || !ggml_cannge_float_dtype(dst->type) ||
        (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64)) {
        return false;
    }
    // The builder registers its result under the destination base as well as the node.
    // A partial or reshaped view cannot replace that base holder.
    ggml_cannge_view_info destination;
    if (!ggml_cannge_resolve_view(dst, destination) || destination.offset != 0 ||
        !ggml_are_same_shape(dst, destination.base)) {
        return false;
    }
    return ggml_n_dims(dst) == 2 && data->ne[0] == dst->ne[0] && data->ne[1] == idx->ne[0] &&
           data->ne[0] > 0 && idx->ne[0] > 0 && data->ne[2] == 1 && data->ne[3] == 1 &&
           idx->ne[1] == 1 && idx->ne[2] == 1 && idx->ne[3] == 1;
}

namespace {
class ggml_cannge_plan_analysis {
    ggml_cgraph * cgraph;
    ggml_cannge_plan_io & io;
    std::string & err;
    const ggml_cannge_options & options;
    const int n_nodes;
    std::unordered_map<ggml_tensor *, bool> produced;
    std::unordered_map<ggml_tensor *, int> n_consumers;
    std::unordered_map<ggml_tensor *, bool> sr_in_graph;

    void debug_graph() {
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
    }

    void index_graph() {
        // producers: every node produces its own output tensor; a CPY node also
        // produces its write destination (src[1], which it views): registering
        // the dst as an input would bind it read-only and the cast output port
        // would alias an input buffer, which GE drops on write (910B silent)
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

        // Scatter stays opt-in because GE scatter primitives have failed device probes.
        // A single I32 or I64 index can use the in-graph path.
        const bool scatter_enabled = options.scatter;
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (ggml_is_empty(node) || node->op != GGML_OP_SET_ROWS) {
                continue;
            }
            sr_in_graph[node] = scatter_enabled && ggml_cannge_can_scatter_in_graph(node) &&
                                ggml_nelements(node->src[1]) == 1;
        }
    }

    static ggml_tensor * rearranged_input_base(ggml_tensor * tensor, ggml_tensor * view) {
        ggml_cannge_view_info info;
        int64_t outer;
        if (ggml_cannge_resolve_view(view, info) && !info.is_dense && info.offset == 0 &&
            ggml_cannge_is_rearrange_view(view) && ggml_is_contiguous(info.base) &&
            ggml_cannge_view_prefix_extent(tensor, info.base, outer)) {
            return info.base;
        }
        return nullptr;
    }

    void collect_inputs() {
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
                // GE output port.
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
                    ggml_tensor * base = nullptr;
                    if (src->op == GGML_OP_CONT && src->src[0] != nullptr && src->src[0]->view_src != nullptr) {
                        base = rearranged_input_base(src, src->src[0]);
                    }
                    if (base == nullptr && src->view_src != nullptr) {
                        base = rearranged_input_base(src, src);
                    }
                    if (base != nullptr) {
                        seen_input[src] = true;
                        src = base;
                        if (seen_input.count(src)) {
                            continue;
                        }
                    }
                    seen_input[src] = true;
                    io.inputs.push_back(src);
                }
            }
        }
    }

    void collect_outputs() {
        std::unordered_map<ggml_tensor *, bool> is_input;
        for (ggml_tensor * t : io.inputs) {
            is_input[t] = true;
        }

        // Keep supported merged-cache updates in GE and expose the updated cache as an output.
        // Other layouts retain the host scatter and its row-data output port.
        if (options.allow_noncontiguous_cont()) {
            for (int i = 0; i < n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (ggml_is_empty(node) || node->op != GGML_OP_SET_ROWS || node->src[2] == nullptr) {
                    continue;
                }
                ggml_cannge_view_info dinfo;
                if (ggml_cannge_can_scatter_in_graph(node) &&
                    ggml_cannge_resolve_view(node->src[2], dinfo) && is_input.count(dinfo.base)) {
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
    }

    void collect_side_effects() {
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
    }

    void assign_output_ports() {
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
    }

    bool resolve_layouts() {
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
                err = "invalid input view chain: " + std::string(t->name);
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
                err = "invalid output view chain: " + std::string(t->name);
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
        return true;
    }

    void estimate_memory() {
        // rough device working memory estimate: outputs plus external inputs
        for (int i = 0; i < n_nodes; i++) {
            io.mem_estimate += ggml_nbytes(cgraph->nodes[i]);
        }
        for (ggml_tensor * t : io.inputs) {
            io.mem_estimate += ggml_nbytes(t);
        }
    }

  public:
    ggml_cannge_plan_analysis(ggml_cgraph * graph, ggml_cannge_plan_io & io, std::string & err,
                             const ggml_cannge_options & options)
        : cgraph(graph), io(io), err(err), options(options), n_nodes(graph->n_nodes) {}

    bool run() {
        io.options = options;
        debug_graph();
        index_graph();
        collect_inputs();
        collect_outputs();
        collect_side_effects();
        assign_output_ports();
        if (!resolve_layouts()) {
            return false;
        }
        estimate_memory();
        return true;
    }
};
} // namespace

bool ggml_cannge_plan_analyze(ggml_cgraph * cgraph, ggml_cannge_plan_io & io, std::string & err,
                              const ggml_cannge_options & options) {
    ggml_cannge_plan_io analyzed;
    if (!ggml_cannge_plan_analysis(cgraph, analyzed, err, options).run()) {
        return false;
    }
    io = std::move(analyzed);
    err.clear();
    return true;
}
