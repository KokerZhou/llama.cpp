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

#include "common.h"
#include "ge.h"
#include "plan.h"

#include <memory>

// backend context, one per backend instance
struct ggml_backend_cannge_context {
    int32_t     device = -1;
    std::string name;
    aclrtStream stream = nullptr;

    std::unique_ptr<ggml_cannge_ge_context> ge;           // lazy, created on first plan compile
    std::map<std::string, std::unique_ptr<ggml_cannge_plan>> plans;
    uint64_t                                next_plan_seq = 0;

    ggml_backend_cannge_context(int32_t device);

    ~ggml_backend_cannge_context();

    // drop the least recently inserted plan; HBM budget based eviction per D6
    // keeps the oldest-first policy, the GE graph is removed from the session
    void evict_oldest_plan();
};

void ggml_backend_cannge_set_device(int32_t device);
