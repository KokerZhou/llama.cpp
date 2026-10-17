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
    bool initialized = false;

    std::unique_ptr<ggml_cannge_ge_context> ge;           // lazy, created on first plan compile
    std::map<std::string, std::unique_ptr<ggml_cannge_plan>> plans;
    uint64_t                                next_plan_seq = 0;

    ggml_backend_cannge_context(int32_t device);

    // Retain the oldest plan for retry if synchronization or graph removal fails.
    bool evict_oldest_plan();

  private:
    // Linked only after the backend relinquishes this context.
    ggml_backend_cannge_context * next_retired = nullptr;

    ~ggml_backend_cannge_context();
    friend void ggml_cannge_release_context(ggml_backend_cannge_context * ctx);
};

void ggml_backend_cannge_set_device(int32_t device);

// Retain failed drains and retry them on subsequent calls. Null retries without retiring a context.
void ggml_cannge_release_context(ggml_backend_cannge_context * ctx);
