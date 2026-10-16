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
#include "graph-build-priv.h"
#include "ops.h"

#include "ggml-impl.h"

#include <exception>
#include <graph/graph.h>

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
