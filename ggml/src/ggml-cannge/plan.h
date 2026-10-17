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

#pragma once

#include "ggml.h"
#include "options.h"
#include "view.h"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

// post-execute copy back to a ggml device address (side-effect writeback):
// after ExecuteGraphWithStreamAsync, enqueue a D2D copy of src into dst on
// the backend stream; both tensors live in this backend's device memory
struct ggml_cannge_writeback {
    ggml_tensor * src = nullptr; // ggml tensor holding the GE-produced value
    ggml_tensor * dst = nullptr; // tensor whose device address must receive it
};

// KV cache row scatter (GGML_OP_SET_ROWS): dst view of the cache, data the new
// rows, idx the I32 slot positions. in_graph: the write is a GE ScatterUpdate
// inside the compiled graph (fresh rows visible to in-fragment view chains;
// no post-execute host scatter); otherwise applied as D2D row copies after
// execute
struct ggml_cannge_set_rows_op {
    ggml_tensor * dst      = nullptr; // src[2] of the node, the cache view
    ggml_tensor * data     = nullptr; // src[0], contiguous rows
    ggml_tensor * idx      = nullptr; // src[1], I32 indices
    bool          in_graph = false;
};

// per-call IO analysis: holds fresh ggml_tensor pointers and is rebuilt on
// every graph_compute, the compute context (and its tensors) may be freed
// between calls
struct ggml_cannge_plan_io {
    ggml_cannge_options options;
    std::vector<ggml_tensor *> inputs;       // external inputs, producerless srcs
    std::vector<ggml_tensor *> outputs;      // boundary outputs plus flagged nodes
    std::vector<ggml_tensor *> side_effects; // in-place written tensors (GGML_OP_SET)

    std::vector<std::pair<ggml_tensor *, ggml_cannge_view_info>> io_views;

    std::vector<bool> input_staged; // parallel to inputs: non-dense input, copied
                                    // into a dense staging buffer before execute

    std::vector<ggml_tensor *> input_base;       // parallel to inputs: registered base a view aliases
    std::vector<size_t>        input_base_offset; // parallel to inputs: byte offset of the view inside base

    std::vector<char> output_strided; // parallel to outputs: non-dense boundary
                                      // output (e.g. a PERMUTE as graph result),
                                      // GE writes dense staging, copied back
                                      // strided after execute

    std::vector<ggml_tensor *> output_tensors;     // one per SetOutput port, bind order
    std::vector<ggml_cannge_writeback> writebacks; // executed after the graph, same stream
    std::vector<ggml_cannge_set_rows_op> set_rows; // KV scatter, after the graph

    // non-dense inputs are staged into dense buffers at execute time; non-dense
    // outputs (a PERMUTE as graph boundary) still fail the build, output-side
    // staging is a later phase

    bool   needs_staging = false; // some IO is not row-major dense
    size_t mem_estimate = 0;

    // IFA decode: kv validity via actual_seq_lengths (int64 [1], GE input
    // port bound to a backend-owned device buffer) instead of the BOOL
    // atten_mask, which IFA ignores for s_q=1 (probe fa m12, note-42). The
    // value is the visible-prefix length of fa_mask, refreshed per execute
    ggml_tensor * fa_mask     = nullptr;
    void *        fa_asl_dev  = nullptr; // owned by the plan, 8 bytes
};

// compiled plan: the cache unit of one cgraph signature, holds no ggml
// pointers so it survives compute context resets; GE owns the compiled graph
// (added via AddGraphWithCopy), the IR is released right after compile
struct ggml_cannge_plan {
    enum state { CREATED, COMPILED, LOADED, FAILED };

    enum state  state = CREATED;
    std::string signature;
    uint32_t    graph_id = 0;
    size_t      mem_estimate = 0;
    uint64_t    seq = 0; // insertion sequence for oldest-first cache eviction

    // staging buffers for output ports whose address range overlaps an input
    // (aliased views, e.g. PERMUTE of an input): GE writes here, a post-
    // execute D2D copy lands the data in the real ggml buffer; one slot per
    // output port, lazily allocated (sizes are fixed per signature)
    std::vector<void *> staging;

    // dense copies of non-dense inputs (strided views), filled by a pre-execute
    // D2D copy on the compute stream; one slot per input, nullptr when direct
    std::vector<void *> input_staging;
    // packed-arena staging for the per-step dense host inputs (merged graphs
    // stage ~150 tiny inputs per execute; per-copy launch cost dominated
    // bind): one pinned host arena, one async H2D per execute
    void *  in_arena       = nullptr;
    void *  in_arena_host  = nullptr;
    size_t  in_arena_size  = 0;
    // Source addresses of staged weight inputs; null means the copy is not reusable.
    std::vector<const void *> input_staged_sources;

    // IFA actual_seq_lengths device buffer (8 bytes, int64 [1]); refreshed
    // from the visible-prefix length of io.fa_mask on every execute
    void * fa_asl_dev = nullptr;

    ~ggml_cannge_plan(); // frees all staging, defined where ACL is available
};

// Replace IO on success; leave it unchanged on failure. Success clears err.
bool ggml_cannge_plan_analyze(ggml_cgraph * cgraph, ggml_cannge_plan_io & io, std::string & err,
                              const ggml_cannge_options & options = ggml_cannge_options::from_env());

std::string ggml_cannge_plan_signature(ggml_cgraph * cgraph, const ggml_cannge_plan_io & io);
