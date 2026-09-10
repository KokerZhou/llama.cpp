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

#include <graph/types.h>

#include <cstdint>
#include <vector>

struct ggml_cannge_plan;
struct ggml_cannge_plan_io;
struct ggml_cgraph;

// ggml -> GE dtype mapping, F32/I32 names follow the ge::DataType enum
static inline bool ggml_cannge_ge_dtype(enum ggml_type type, ge::DataType & dt) {
    switch (type) {
        case GGML_TYPE_F16:
            dt = ge::DT_FLOAT16;
            return true;
        case GGML_TYPE_F32:
            dt = ge::DT_FLOAT;
            return true;
        case GGML_TYPE_I32:
            dt = ge::DT_INT32;
            return true;
        default:
            return false;
    }
}

// axis order convention (docs/cannge note-7): ggml ne[0] is innermost, GE
// shapes are outermost-first, so ge_shape = reverse(ggml ne[:ndims]); this is
// the single conversion helper for all shapes, Permute orders and Slice offsets
static inline std::vector<int64_t> ggml_cannge_ge_shape(const ggml_tensor * t) {
    std::vector<int64_t> shape;
    const int            ndims = ggml_n_dims(t);
    shape.reserve(ndims);
    for (int i = ndims - 1; i >= 0; i--) {
        shape.push_back(t->ne[i]);
    }
    return shape;
}

// translates the analyzed per-call IO into a ge::Graph via the ES authoring
// API; the returned graph is owned by the caller either way (on success the
// session holds an AddGraphWithCopy copy and the IR can be released)
bool ggml_cannge_graph_build(ggml_cannge_plan & plan, ggml_cannge_plan_io & io, ggml_cgraph * cgraph, void *& graph,
                             std::string & err);
