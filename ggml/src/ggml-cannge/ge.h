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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// GE types are kept out of this header; ge.cpp includes the GE headers
namespace ge {
class Graph;
class Session;
}
namespace gert {
class Tensor;
}

// one GE session per backend context; process-wide GEInitialize/GEFinalize
// reference counting, see docs/cannge D6
class ggml_cannge_ge_context {
  public:
    // returns nullptr on failure, err carries the reason
    static ggml_cannge_ge_context * create(int32_t device, std::string & err);
    ~ggml_cannge_ge_context();

    ggml_cannge_ge_context(const ggml_cannge_ge_context &)            = delete;
    ggml_cannge_ge_context & operator=(const ggml_cannge_ge_context &) = delete;

    // graph is an opaque ge::Graph produced by the graph build step
    bool add_and_compile(void * graph, uint32_t & graph_id, std::string & err);
    bool load(uint32_t graph_id, void * stream, std::string & err);
    bool remove(uint32_t graph_id);

    bool execute(uint32_t graph_id, void * stream, const std::vector<gert::Tensor> & inputs,
                 std::vector<gert::Tensor> & outputs, std::string & err);

  private:
    ggml_cannge_ge_context() = default;

    struct impl;
    std::unique_ptr<impl> pimpl;
};
