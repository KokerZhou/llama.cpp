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

#include "ggml-cannge-impl.h"

#include <mutex>

ggml_backend_cannge_context::ggml_backend_cannge_context(int32_t device) : device(device), name(GGML_CANNGE_NAME + std::to_string(device)) {
    if (!ggml_cannge_check(aclrtSetDevice(device), "aclrtSetDevice", __func__)) {
        return;
    }
    initialized = ggml_cannge_check(aclrtCreateStream(&stream), "aclrtCreateStream", __func__) && stream != nullptr;
}

ggml_backend_cannge_context::~ggml_backend_cannge_context() {
    // release_context has selected the device and drained the stream.
    ge.reset();
    plans.clear();
    if (stream != nullptr) {
        ggml_cannge_check(aclrtDestroyStream(stream), "aclrtDestroyStream", __func__);
    }
}

bool ggml_backend_cannge_context::evict_oldest_plan() {
    auto oldest = plans.end();
    for (auto it = plans.begin(); it != plans.end(); ++it) {
        if (oldest == plans.end() || it->second->seq < oldest->second->seq) {
            oldest = it;
        }
    }
    if (oldest != plans.end()) {
        // Keep the graph and its buffers until queued work and graph removal both complete.
        if (stream != nullptr &&
            !ggml_cannge_check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream", __func__)) {
            return false;
        }
        if (ge != nullptr && oldest->second->graph_id != 0 &&
            !ge->remove(oldest->second->graph_id)) {
            return false;
        }
        plans.erase(oldest);
    }
    return true;
}

ggml_cannge_plan::~ggml_cannge_plan() {
    for (void * p : staging) {
        aclrtFree(p);
    }
    for (void * p : input_staging) {
        aclrtFree(p);
    }
    if (in_arena != nullptr) {
        aclrtFree(in_arena);
    }
    if (in_arena_host != nullptr) {
        aclrtFreeHost(in_arena_host);
    }
    if (fa_asl_dev != nullptr) {
        aclrtFree(fa_asl_dev);
    }
}

void ggml_cannge_release_context(ggml_backend_cannge_context * ctx) {
    static std::mutex mutex;
    // Raw links deliberately retain failed contexts at process exit rather than running unsafe destructors.
    static ggml_backend_cannge_context * retired = nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    if (ctx != nullptr) {
        ctx->next_retired = retired;
        retired = ctx;
    }
    auto ** link = &retired;
    while (*link != nullptr) {
        auto * candidate = *link;
        if (candidate->stream != nullptr &&
            (!ggml_cannge_check(aclrtSetDevice(candidate->device), "select retired context device", __func__) ||
             !ggml_cannge_check(aclrtSynchronizeStream(candidate->stream), "drain retired context", __func__))) {
            link = &candidate->next_retired;
            continue;
        }
        *link = candidate->next_retired;
        candidate->next_retired = nullptr;
        delete candidate;
    }
}
